#!/usr/bin/env python
__author__ = 'arenduchintala'

class WordRepresenter(nn.Module):
    def __init__(self, spelling, cv_size, cp_idx, we_size,
                 bidirectional=True, dropout=0.3,
                 is_extra_feat_learnable=False,
                 ce_size=20,
                 cr_size=50,
                 c_rnn_layers=1,
                 char_composition='RNN', pool='Max'):
        super(WordRepresenter, self).__init__()
        self.spelling = spelling
        self.sorted_spellings, self.sorted_spell_lens, self.spell_lens, self.unsort_idx, self.freqs = self.init_word2spelling()
        self.v_size = len(self.sorted_spell_lens)
        self.ce_size = ce_size
        self.we_size = we_size
        self.cv_size = cv_size
        self.cr_size = cr_size
        self.c_rnn_layers = c_rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.ce_layer = torch.nn.Embedding(self.cv_size, self.ce_size, padding_idx=cp_idx)
        self.vocab_idx = Variable(torch.arange(self.v_size).long(), requires_grad=False)
        self.ce_layer.weight = nn.Parameter(
            torch.FloatTensor(self.cv_size, self.ce_size).uniform_(-0.5 / self.ce_size, 0.5 / self.ce_size))
        char_comp_items = char_composition.split('+')
        self.char_composition = char_comp_items[0]
        if len(char_comp_items) > 1:
            self.use_word_embeddings = char_comp_items[1].lower() == ('word')
            self.use_wordfreq_embeddings = char_comp_items[1].lower() == ('wordfreq')
        else:
            self.use_word_embeddings = False
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
                                               torch.nn.Sigmoid()
                                              )
        else:
            pass

        if self.char_composition == 'RNN':
            self.c_rnn = torch.nn.LSTM(input_size=self.ce_size, hidden_size=self.cr_size, num_layers=c_rnn_layers,
                                       bidirectional=bidirectional, batch_first=True,
                                       dropout=self.dropout)
            if self.cr_size * (2 if bidirectional else 1) * self.c_rnn_layers != self.we_size:
                self.c_proj = torch.nn.Linear(self.cr_size * (2 if bidirectional else 1) * self.c_rnn_layers, self.we_size)
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

        # self.extra_ce_layer = torch.nn.Embedding(self.v_size, 1)
        # self.register_forward_hook(self.forward_hook)
        # self.register_backward_hook(self.backward_hook)
        self.prev_unsorted_word_embeddings = None
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
        freqs = Variable(freqs.unsqueeze(1), requires_grad=False)
        spellings = self.spelling[:, :-2]
        sorted_spell_lens, sort_idx = torch.sort(lengths, 0, True)
        unsort_idx = get_unsort_idx(sort_idx)
        sorted_spell_lens = sorted_spell_lens.numpy().tolist()
        lengths = lengths.numpy().tolist()
        sorted_spellings = spellings[sort_idx, :]
        sorted_spellings = Variable(sorted_spellings, requires_grad=False)
        return sorted_spellings, sorted_spell_lens, lengths, unsort_idx, freqs

    def init_cuda(self,):
        self = self.cuda()
        self.sorted_spellings = self.sorted_spellings.cuda()
        self.unsort_idx = self.unsort_idx.cuda()
        self.vocab_idx = self.vocab_idx.cuda()
        self.freqs = self.freqs.cuda()

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

    def sort_n_pack(self, emb, lens):
        return pack(emb, lens, batch_first=True)

    def rnn_representer(self, emb, lens):
        packed_emb = self.sort_n_pack(emb, lens)
        output, (ht, ct) = self.c_rnn(packed_emb, None)
        # output, l = unpack(output)
        del output, ct
        if ht.size(0) % 2 == 0:
            # concat the last ht from fwd RNN and first ht from bwd RNN
            ht = torch.cat([ht[i, :, :] for i in range(ht.size(0))], dim=1)
        else:
            ht = ht.squeeze()
        if self.c_proj is not None:
            word_embeddings = self.c_proj(ht)
        else:
            word_embeddings = ht
        return word_embeddings

    def forward(self, tgt_select=None):
        # print('word_representer forward call')
        if tgt_select is None:
            selected_spellings = self.sorted_spellings
            selected_vocab_idx = self.vocab_idx
            selected_freqs = self.freqs
            lens = self.sorted_spell_lens
        else:
            pdb.set_trace()
            selected_spellings = self.sorted_spellings[self.unsort_idx[tgt_select]]
            if tgt_select.is_cuda:
                lens = self.sorted_spell_lens[self.unsort_idx[tgt_select].cpu().numpy().tolist()]
            else:
                lens = self.sorted_spell_lens[self.unsort_idx[tgt_select].numpy().tolist()]
            selected_vocab_idx = tgt_select
            selected_freqs = self.freqs[tgt_select]
        emb = self.ce_layer(selected_spellings)
        if not hasattr(self, 'char_composition'):  # for back compatibility
            composed_word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'RNN':
            composed_word_embeddings = self.rnn_representer(emb, lens)
        elif self.char_composition == 'CNN':
            composed_word_embeddings = self.cnn_representer(emb)
        else:
            raise BaseException("unknown char_composition")

        unsorted_composed_word_embeddings = composed_word_embeddings[self.unsort_idx, :]

        pdb.set_trace()
        if self.use_wordfreq_embeddings:
            word_embeddings = self.word_embeddings(selected_vocab_idx)
            merge = self.freq_sig(self.freq_wt(selected_freqs) + self.freq_bias(selected_vocab_idx))
            merge = merge.expand(self.v_size, word_embeddings.size(1))
            unsorted_word_embeddings = (merge * word_embeddings) + ((1.0 - merge) * unsorted_composed_word_embeddings)
        elif self.use_word_embeddings:
            word_embeddings = self.word_embeddings(selected_vocab_idx)
            merge = self.merge_weights(selected_vocab_idx).expand(self.v_size, word_embeddings.size(1))
            unsorted_word_embeddings = (merge * word_embeddings) + ((1.0 - merge) * unsorted_composed_word_embeddings)
        else:
            unsorted_word_embeddings = unsorted_composed_word_embeddings
        self.prev_unsorted_word_embeddings = unsorted_word_embeddings
        return unsorted_word_embeddings



