#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


class WordRepresenter(nn.Module):
    def __init__(self, spelling, cv_size, cp_idx, we_size,
                 bidirectional=False, dropout=0.3,
                 is_extra_feat_learnable=False,
                 ce_size=50,
                 cr_size=100,
                 char_composition='RNN', pool='Max'):
        super(WordRepresenter, self).__init__()
        self.spelling = spelling
        self.sorted_spellings, self.sorted_lengths, self.unsort_idx, self.freqs = self.init_word2spelling()
        self.v_size = len(self.sorted_lengths)
        self.ce_size = ce_size
        self.we_size = we_size
        self.cv_size = cv_size
        self.cr_size = cr_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.ce_layer = torch.nn.Embedding(self.cv_size, self.ce_size, padding_idx=cp_idx)
        self.vocab_idx = Variable(torch.arange(self.v_size).long(), requires_grad=False)
        self.ce_layer.weight = nn.Parameter(
            torch.FloatTensor(self.cv_size, self.ce_size).uniform_(-0.5 / self.ce_size, 0.5 / self.ce_size))
        char_comp_items = char_composition.split('+')
        self.char_composition = char_comp_items[0]
        if len(char_comp_items) > 1:
            self.use_word_embeddings = char_comp_items[1].lower() == 'word'
        else:
            self.use_word_embeddings = False
        self.pool = pool
        if self.use_word_embeddings:
            self.word_embeddings = nn.Embedding(self.v_size, self.we_size)
            self.merge_weights = nn.Sequential(
                                                nn.Embedding(self.v_size, 1),
                                                torch.nn.Sigmoid()
                                              )
        if self.char_composition == 'RNN':
            self.c_rnn = torch.nn.LSTM(self.ce_size, self.cr_size,
                                       bidirectional=bidirectional, batch_first=True,
                                       dropout=self.dropout)
            if self.cr_size * (2 if bidirectional else 1) != self.we_size:
                self.c_proj = torch.nn.Linear(self.cr_size * (2 if bidirectional else 1), self.we_size)
                print('using Linear c_proj layer')
            else:
                print('no Linear c_proj layer')
                self.c_proj = None
        elif self.char_composition == 'CNN':
            assert self.we_size % 4 == 0
            self.c1d_3g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 3)
            self.c1d_4g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 4)
            self.c1d_5g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 5)
            self.c1d_6g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 6)
            if self.pool == 'Avg':
                self.max_3g = torch.nn.AvgPool1d(self.sorted_spellings.size(1) - 3 + 1)
                self.max_4g = torch.nn.AvgPool1d(self.sorted_spellings.size(1) - 4 + 1)
                self.max_5g = torch.nn.AvgPool1d(self.sorted_spellings.size(1) - 5 + 1)
                self.max_6g = torch.nn.AvgPool1d(self.sorted_spellings.size(1) - 6 + 1)
            elif self.pool == 'Max':
                self.max_3g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 3 + 1)
                self.max_4g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 4 + 1)
                self.max_5g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 5 + 1)
                self.max_6g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 6 + 1)
            else:
                raise BaseException("uknown pool")
        else:
            raise BaseException("Unknown seq model")

        #self.extra_ce_layer = torch.nn.Embedding(self.v_size, 1)
        print('WordRepresenter init complete.')

    def init_word2spelling(self,):
        #for v, s in self.word2spelling.items():
        #    if spellings is not None:
        #        spellings = torch.cat((spellings, torch.LongTensor(s).unsqueeze(0)), dim=0)
        #    else:
        #        spellings = torch.LongTensor(s).unsqueeze(0)
        lengths = self.spelling[:, -2]
        counts = self.spelling[:, -1].float()
        freqs = counts / counts.sum()
        spellings = self.spelling[:, :-2]
        sorted_lengths, sort_idx = torch.sort(lengths, 0, True)
        unsort_idx = get_unsort_idx(sort_idx)
        sorted_lengths = sorted_lengths.numpy().tolist()
        sorted_spellings = spellings[sort_idx, :]
        sorted_spellings = Variable(sorted_spellings, requires_grad=False)
        return sorted_spellings, sorted_lengths, unsort_idx, freqs

    def init_cuda(self,):
        self = self.cuda()
        self.sorted_spellings = self.sorted_spellings.cuda()
        self.unsort_idx = self.unsort_idx.cuda()
        self.vocab_idx = self.vocab_idx.cuda()

    def cnn_representer(self, emb):
        # (batch, seq_len, char_emb_size)
        emb = emb.transpose(1, 2)
        m_3g = self.max_3g(self.c1d_3g(emb)).squeeze()
        m_4g = self.max_4g(self.c1d_4g(emb)).squeeze()
        m_5g = self.max_5g(self.c1d_5g(emb)).squeeze()
        m_6g = self.max_6g(self.c1d_6g(emb)).squeeze()
        word_embeddings = torch.cat([m_3g, m_4g, m_5g, m_6g], dim=1)
        del emb, m_3g, m_4g, m_5g, m_6g
        return word_embeddings

    def rnn_representer(self, emb):
        packed_emb = pack(emb, self.sorted_lengths, batch_first=True)
        output, (ht, ct) = self.c_rnn(packed_emb, None)
        # output, l = unpack(output)
        del output, ct
        if ht.size(0) == 2:
            # concat the last ht from fwd RNN and first ht from bwd RNN
            ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)
        else:
            ht = ht.squeeze()
        if self.c_proj is not None:
            word_embeddings = self.c_proj(ht)
        else:
            word_embeddings = ht
        return word_embeddings

    def forward(self,):
        emb = self.ce_layer(self.sorted_spellings)
        if not hasattr(self, 'char_composition'):  # for back compatibility
            composed_word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'RNN':
            composed_word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'CNN':
            composed_word_embeddings = self.cnn_representer(emb)
        else:
            raise BaseException("unknown char_composition")

        unsorted_composed_word_embeddings = composed_word_embeddings[self.unsort_idx, :]
        if self.use_word_embeddings:
            word_embeddings = self.word_embeddings(self.vocab_idx)
            merge = self.merge_weights(self.vocab_idx).expand(self.v_size, word_embeddings.size(1))
            unsorted_word_embeddings = (merge * unsorted_composed_word_embeddings) + ((1.0 - merge) * word_embeddings)
        else:
            unsorted_word_embeddings = unsorted_composed_word_embeddings
        #print('word_emb called')
        return unsorted_word_embeddings


class VarLinear(nn.Module):
    def __init__(self, word_representer):
        super(VarLinear, self).__init__()
        self.word_representer = word_representer

    def matmul(self, data):
        #shape of data (bs,  seq_len, hidden_size)
        var = self.word_representer()
        #shape of var is (vocab_size, hidden_size)
        if data.dim() > 1:
            assert data.size(-1) == var.size(-1)
            return torch.matmul(data, var.transpose(0, 1))
        else:
            raise BaseException("data should be at least 2 dimensional")

    def forward(self, data):
        return self.matmul(data)


class VarEmbedding(nn.Module):
    def __init__(self, word_representer):
        super(VarEmbedding, self).__init__()
        self.word_representer = word_representer

    def forward(self, data):
        return self.lookup(data)

    def lookup(self, data):
        var = self.word_representer()
        embedding_size = var.size(1)
        if data.dim() == 2:
            batch_size = data.size(0)
            seq_len = data.size(1)
            data = data.contiguous()
            data = data.view(-1)  # , data.size(0), data.size(1))
            var_data = var[data]
            var_data = var_data.view(batch_size, seq_len, embedding_size)
        else:
            var_data = var[data]
        return var_data



