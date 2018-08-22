#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np
import pdb


class StreamWrapper():
    def __init__(self, idx, stream):
        self.idx = idx
        self.stream = stream


def unique(v, max_lim=0, fill=0):
    if v.is_cuda:
        u, u_i, u_inv = np.unique(v.data.cpu().numpy(), True, True)
    else:
        u, u_i, u_inv = np.unique(v.data.numpy(), True, True)
    if fill > 0 and max_lim > fill:
        u_fill = np.random.choice(np.arange(max_lim), size=fill, replace=False)
        u = np.union1d(u, u_fill)
    u_dict = dict((i, idx) for idx, i in enumerate(u))
    u = torch.Tensor(u).type_as(v.data)
    return u, u_dict


def t_sort(v):
    sorted_v, ind_v = torch.sort(v, 0, descending=True)
    return sorted_v, ind_v


def t_tolist(v):
    if v.is_cuda:
        v_list = v.cpu().numpy().tolist()
    else:
        v_list = v.numpy().tolist()
    return v_list


def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).type_as(sort_idx))
    return unsort_idx


class HighwayNetwork(nn.Module):

    def __init__(self, input_size):
        super(HighwayNetwork, self).__init__()
        # transform gate
        self.trans_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid())
        # highway
        self.activation = nn.ReLU()

        self.h_layer = nn.Sequential(
                   nn.Linear(input_size, input_size),
                   self.activation)
        self.trans_gate[0].weight.data.uniform_(-0.05, 0.05)
        self.h_layer[0].weight.data.uniform_(-0.05, 0.05)
        self.trans_gate[0].bias.data.fill_(0)
        self.h_layer[0].bias.data.fill_(0)

    def forward(self, x):
        t = self.trans_gate(x)
        h = self.h_layer(x)
        z = torch.mul(t, h)+torch.mul(1.0 - t, x)
        return z


class WordRepresenter(nn.Module):
    def __init__(self, spelling_matrix, cv_size, cp_idx, we_size, rnn_size,
                 bidirectional=True, dropout=0.3,
                 is_extra_feat_learnable=False,
                 ce_size=50,
                 cr_size=100,
                 c_rnn_layers=1,
                 char_composition='RNN',
                 pool='Max',
                 kernals='3456'):
        super(WordRepresenter, self).__init__()
        self.spelling_matrix = spelling_matrix
        # self.spellings, self.sorted_spellings, self.sorted_spell_lens, self.spell_lens, self.unsort_idx, self.freqs = self.init_word2spelling()
        self.spellings, self.spelling_lengths, self.freqs = self.init_word2spelling()
        self.v_size = len(self.spelling_lengths)
        self.ce_size = ce_size
        self.we_size = we_size
        self.rnn_size = rnn_size
        self.cv_size = cv_size
        self.cr_size = cr_size
        self.c_rnn_layers = c_rnn_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.ce_layer = torch.nn.Embedding(self.cv_size, self.ce_size, padding_idx=cp_idx)
        self.vocab_idx = Variable(torch.arange(self.v_size).long(), requires_grad=False)
        self.ce_layer.weight = nn.Parameter(
            torch.FloatTensor(self.cv_size, self.ce_size).uniform_(-0.5 / self.ce_size, 0.5 / self.ce_size))
        char_comp_items = char_composition.split('+')
        self.char_composition = char_comp_items[0]
        self.cached = None
        if self.rnn_size != self.we_size:
            self.proj = torch.nn.Linear(self.we_size, self.rnn_size)
        else:
            self.proj = None

        if len(char_comp_items) > 1:
            self.use_word_embeddings = char_comp_items[1].lower() == ('word')
            self.use_wordgate_embeddings = char_comp_items[1].lower() == ('wordgate')
            self.use_wordfreq_embeddings = char_comp_items[1].lower() == ('wordfreq')
        else:
            self.use_word_embeddings = False
            self.use_wordgate_embeddings = False
            self.use_wordfreq_embeddings = False
        self.pool = pool
        if self.use_wordfreq_embeddings:
            print('using wordfreq_embeddings')
            self.word_embeddings = nn.Embedding(self.v_size, self.we_size)
            self.freq_wt = nn.Linear(1, 1, bias=False)
            self.freq_bias = nn.Embedding(self.v_size, 1)
            self.freq_sig = torch.nn.Sigmoid()
        elif self.use_word_embeddings:
            print('using word_embeddings')
            self.word_embeddings = nn.Embedding(self.v_size, self.we_size)
            self.merge_weights = nn.Sequential(
                                               nn.Embedding(self.v_size, 1),
                                               nn.Dropout(0.2),
                                               torch.nn.Sigmoid()
                                              )
        elif self.use_wordgate_embeddings:
            print('using wordgate_embeddings')
            self.word_embeddings = nn.Embedding(self.v_size, self.we_size)
            self.merge_weights = nn.Sequential(
                                               nn.Embedding(self.v_size, self.we_size),
                                               nn.Dropout(0.2),
                                               torch.nn.Sigmoid()
                                              )
        else:
            pass

        if self.char_composition == 'RNN':
            self.c_rnn = torch.nn.LSTM(input_size=self.ce_size, hidden_size=self.cr_size, num_layers=c_rnn_layers,
                                       bidirectional=bidirectional, batch_first=True,
                                       dropout=dropout)
            if self.cr_size * (2 if bidirectional else 1) * self.c_rnn_layers != self.we_size:
                self.c_proj = torch.nn.Linear(self.cr_size * (2 if bidirectional else 1) * self.c_rnn_layers, self.we_size)
                print('using Linear c_proj layer')
            else:
                print('no Linear c_proj layer')
                self.c_proj = None
        elif self.char_composition == 'CNN':
            kernals = [int(i) for i in kernals]
            assert self.we_size % len(kernals) == 0
            cnns = []
            for k in kernals:
                seq = nn.Sequential(
                    nn.Conv1d(self.ce_size, self.we_size // len(kernals), k, padding=cp_idx),
                    nn.Tanh(),
                    nn.MaxPool1d(self.spellings.size(1) - k + 1)
                    )
                cnns.append(seq)
            self.cnns = nn.ModuleList(cnns)
            for cnn in self.cnns:
                cnn[0].weight.data.uniform_(-0.05, 0.05)
                cnn[0].bias.data.fill_(0.)
            self.highway1 = HighwayNetwork(self.we_size)
            self.highway2 = HighwayNetwork(self.we_size)
            #self.c1d_3g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 3)
            #self.c1d_4g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 4)
            #self.c1d_5g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 5)
            #self.c1d_6g = torch.nn.Conv1d(self.ce_size, self.we_size // 4, 6)
            #if self.pool == 'Avg':
            #    self.max_3g = torch.nn.AvgPool1d(self.spellings.size(1) - 3 + 1)
            #    self.max_4g = torch.nn.AvgPool1d(self.spellings.size(1) - 4 + 1)
            #    self.max_5g = torch.nn.AvgPool1d(self.spellings.size(1) - 5 + 1)
            #    self.max_6g = torch.nn.AvgPool1d(self.spellings.size(1) - 6 + 1)
            #elif self.pool == 'Max':
            #    self.max_3g = torch.nn.MaxPool1d(self.spellings.size(1) - 3 + 1)
            #    self.max_4g = torch.nn.MaxPool1d(self.spellings.size(1) - 4 + 1)
            #    self.max_5g = torch.nn.MaxPool1d(self.spellings.size(1) - 5 + 1)
            #    self.max_6g = torch.nn.MaxPool1d(self.spellings.size(1) - 6 + 1)
            #else:
            #    raise BaseException("uknown pool")
        else:
            raise BaseException("Unknown seq model")

        # self.extra_ce_layer = torch.nn.Embedding(self.v_size, 1)
        # self.register_forward_hook(self.forward_hook)
        # self.register_backward_hook(self.backward_hook)
        self.max_train_embed = 5000
        self.precomputed_word_embeddings = None
        self.prev_training = True
        print('WordRepresenter init complete.')

    def init_word2spelling(self,):
        #for v, s in self.word2spelling.items():
        #    if spellings is not None:
        #        spellings = torch.cat((spellings, torch.LongTensor(s).unsqueeze(0)), dim=0)
        #    else:
        #        spellings = torch.LongTensor(s).unsqueeze(0)
        lengths = self.spelling_matrix[:, -2]
        counts = self.spelling_matrix[:, -1].float()
        freqs = counts / counts.sum()
        freqs = Variable(freqs.unsqueeze(1), requires_grad=False)
        spellings = Variable(self.spelling_matrix[:, :-2], requires_grad=False)
        sorted_spell_lens, sort_idx = torch.sort(lengths, 0, True)
        unsort_idx = get_unsort_idx(sort_idx)
        sorted_spell_lens = sorted_spell_lens.numpy().tolist()
        sorted_spellings = spellings[sort_idx, :]
        return spellings, lengths, freqs

    def init_cuda(self,):
        self = self.cuda()
        # self.sorted_spellings = self.sorted_spellings.cuda()
        self.spellings = self.spellings.cuda()
        self.spelling_lengths = self.spelling_lengths.cuda()
        # self.unsort_idx = self.unsort_idx.cuda()
        self.vocab_idx = self.vocab_idx.cuda()
        self.freqs = self.freqs.cuda()
        if self.proj is not None:
            self.proj = self.proj.cuda()

    def cnn_representer(self, emb):
        # (batch, seq_len, char_emb_size)
        emb = emb.transpose(1, 2)
        # m_3g = self.max_3g(nn.functional.tanh(self.c1d_3g(emb))).squeeze()
        # m_4g = self.max_4g(nn.functional.tanh(self.c1d_4g(emb))).squeeze()
        # m_5g = self.max_5g(nn.functional.tanh(self.c1d_5g(emb))).squeeze()
        # m_6g = self.max_6g(nn.functional.tanh(self.c1d_6g(emb))).squeeze()
        #stream_tmp = []
        #streams = [(idx, torch.cuda.Stream()) for idx, cnn in enumerate(self.cnns)]
        #wrapped_stream = [StreamWrapper(idx, s) for idx, s in streams]
        #print(torch.cuda.current_device())
        #with torch.cuda.stream(wrapped_stream[0].stream):
        #    print('start stream 0')
        #    cnn = self.cnns[wrapped_stream[0].idx]
        #    stream_tmp.append((wrapped_stream[0].idx, cnn(emb).squeeze()))
        #    print('end stream 0')
        ##torch.cuda.synchronize()
        #with torch.cuda.stream(wrapped_stream[1].stream):
        #    print('start stream 1')
        #    cnn = self.cnns[wrapped_stream[1].idx]
        #    stream_tmp.append((wrapped_stream[1].idx, cnn(emb).squeeze()))
        #    print('end stream 1')
        ##torch.cuda.synchronize()
        #with torch.cuda.stream(wrapped_stream[2].stream):
        #    print('start stream 2')
        #    cnn = self.cnns[wrapped_stream[2].idx]
        #    stream_tmp.append((wrapped_stream[2].idx, cnn(emb).squeeze()))
        #    print('end stream 2')
        ##torch.cuda.synchronize()
        #with torch.cuda.stream(wrapped_stream[3].stream):
        #    print('start stream 3')
        #    cnn = self.cnns[wrapped_stream[3].idx]
        #    stream_tmp.append((wrapped_stream[3].idx, cnn(emb).squeeze()))
        #    print('end stream 3')
        #torch.cuda.synchronize()
        #assert len(stream_tmp) == 4
        #stream_tmp = [t for idx, t in sorted(stream_tmp)]
        #word_embeddings_stream = torch.cat(stream_tmp, dim=1)
        tmp = [_cnn(emb).squeeze() for _cnn in self.cnns]
        word_embeddings = torch.cat(tmp, dim=1)
        #diff = abs((word_embeddings_stream - word_embeddings).sum().data[0])
        #print(diff)
        #assert diff == 0
        word_embeddings = self.highway1(word_embeddings)
        word_embeddings = self.highway2(word_embeddings)
        del emb, tmp  # m_3g, m_4g, m_5g, m_6g
        return word_embeddings

    def rnn_representer(self, emb, lengths):
        if self.training:
            sorted_lengths, sorted_idx = t_sort(lengths)
            unsorter_idx = get_unsort_idx(sorted_idx)
            sorted_emb = self.dropout(emb[sorted_idx])
            packed_emb = pack(sorted_emb, t_tolist(sorted_lengths), batch_first=True)

            output, (ht, ct) = self.c_rnn(packed_emb, None)
            # output, l = unpack(output)
            del output, ct
            if ht.size(0) % 2 == 0:
                # concat the last ht from fwd RNN and first ht from bwd RNN
                ht = torch.cat([ht[i, :, :] for i in range(ht.size(0))], dim=1)
            else:
                ht = ht.squeeze()
            if self.c_proj is not None:
                word_embeddings = self.c_proj(self.dropout(ht))
            else:
                word_embeddings = ht
            word_embeddings = word_embeddings[unsorter_idx]
            self.precomputed_word_embeddings = None
        else:
            if self.precomputed_word_embeddings is None:
                if emb.size(0) > self.max_train_embed:
                    idx = np.arange(emb.size(0))
                    batch_idx = np.array_split(idx, int(emb.size(0) / self.max_train_embed))
                else:
                    idx = np.arange(emb.size(0))
                    batch_idx = [idx]
                sorted_lengths, sorted_idx = t_sort(lengths)
                unsorter_idx = get_unsort_idx(sorted_idx)
                sorted_emb = emb[sorted_idx]
                ht_stack = []
                for _i, b_idx in enumerate(batch_idx):
                    print(_i, b_idx)
                    b_idx = torch.Tensor(b_idx).type_as(sorted_lengths)
                    b_sorted_emb = sorted_emb[b_idx]
                    b_sorted_lens = sorted_lengths[b_idx]
                    b_packed_emb = pack(b_sorted_emb, t_tolist(b_sorted_lens), batch_first=True)
                    b_output, (b_ht, b_ct) = self.c_rnn(b_packed_emb, None)
                    # output, l = unpack(output)
                    del b_output, b_ct, b_sorted_emb, b_packed_emb, b_sorted_lens
                    if b_ht.size(0) % 2 == 0:
                        # concat the last ht from fwd RNN and first ht from bwd RNN
                        b_ht = torch.cat([b_ht[i, :, :] for i in range(b_ht.size(0))], dim=1)
                    else:
                        b_ht = b_ht.squeeze()
                    ht_stack.append(b_ht.data.clone())
                ht = Variable(torch.cat(ht_stack, dim=0))
                if self.c_proj is not None:
                    word_embeddings = self.c_proj(ht)
                else:
                    word_embeddings = ht
                word_embeddings = word_embeddings[unsorter_idx]
                self.precomputed_word_embeddings = word_embeddings
                assert self.precomputed_word_embeddings.size(0) == self.vocab_idx.size(0)
            else:
                word_embeddings = self.precomputed_word_embeddings
        return word_embeddings

    def forward_parts(self,):
        if self.cached is not None:
            return self.cached
        else:
            unsorted_word_embeddings = []
            for tgt_part in torch.split(self.vocab_idx.data, self.v_size // 10):
                emb_part = self(tgt_part)
                emb_part = emb_part.detach()
                unsorted_word_embeddings.append(emb_part)
            unsorted_word_embeddings = torch.cat(unsorted_word_embeddings, dim=0)
            print('done using parts')
            self.cached = unsorted_word_embeddings
            return unsorted_word_embeddings

    def forward(self, tgt_select=None):
        # print('word_representer forward call')
        if tgt_select is None:
            tgt_select = self.vocab_idx.data
        selected_spellings = self.spellings[tgt_select, :]
        selected_lengths = self.spelling_lengths[tgt_select]
        selected_vocab_idx = Variable(tgt_select, requires_grad=False)
        emb = self.ce_layer(selected_spellings)
        if not hasattr(self, 'char_composition'):  # for back compatibility
            composed_word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'RNN':
            composed_word_embeddings = self.rnn_representer(emb, selected_lengths)
        elif self.char_composition == 'CNN':
            composed_word_embeddings = self.cnn_representer(emb)
        else:
            raise BaseException("unknown char_composition")

        # unsorted_composed_word_embeddings = composed_word_embeddings[self.unsort_idx, :]
        unsorted_composed_word_embeddings = composed_word_embeddings

        if self.use_wordfreq_embeddings:
            raise NotImplementedError("removed this during reimplementation")
        elif self.use_word_embeddings:
            word_embeddings = self.word_embeddings(selected_vocab_idx)
            merge = self.merge_weights(selected_vocab_idx).expand(word_embeddings.size(0), word_embeddings.size(1))
            unsorted_word_embeddings = (merge * word_embeddings) + ((1.0 - merge) * unsorted_composed_word_embeddings)
        elif self.use_wordgate_embeddings:
            word_embeddings = self.word_embeddings(selected_vocab_idx)
            merge = self.merge_weights(selected_vocab_idx)
            unsorted_word_embeddings = (merge * word_embeddings) + ((1.0 - merge) * unsorted_composed_word_embeddings)
        else:
            unsorted_word_embeddings = unsorted_composed_word_embeddings
        return unsorted_word_embeddings


class VarLinear(nn.Module):
    def __init__(self, word_representer):
        super(VarLinear, self).__init__()
        self.word_representer = word_representer

    def matmul(self, data, tgt_select):
        # shape of data (bs,  seq_len, hidden_size)
        if self.training:
            self.word_representer.cached = None
            var = self.word_representer(tgt_select)
        else:
            var = self.word_representer.forward_parts()
        # shape of var is (vocab_size, hidden_size)
        if data.dim() > 1:
            var_proj = var if self.word_representer.proj is None else self.word_representer.proj(var)
            assert data.size(-1) == var_proj.size(-1)
            return torch.matmul(data, var_proj.transpose(0, 1))
        else:
            raise BaseException("data should be at least 2 dimensional")

    def forward(self, data, tgt_select=None):
        return self.matmul(data, tgt_select)


class VarEmbedding(nn.Module):
    def __init__(self, word_representer, tgt_word_vec_size):
        super(VarEmbedding, self).__init__()
        self.word_representer = word_representer
        self.embedding_size = tgt_word_vec_size

    def lookup(self, data):
        if self.training:
            self.word_representer.cached = None
            data_u, u_dict = unique(data)
            data_map = Variable(torch.Tensor([u_dict[i] for i in data.data.view(-1)]).type_as(data.data))
            data = data_map.view(data.shape)
            var = self.word_representer(data_u)
        else:
            # var = self.word_representer(None)
            var = self.word_representer.forward_parts()
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

    def forward(self, data):
        data = data.squeeze(dim=2)
        return self.lookup(data)


class VarGenerator(nn.Module):
    def __init__(self, var_linear):
        super(VarGenerator, self).__init__()
        self.var_linear = var_linear
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, tgt_select=None):
        var_lin = self.var_linear(input, tgt_select)
        return self.log_softmax(var_lin)
