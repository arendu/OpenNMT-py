#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import numpy as np

from torch.autograd import Variable

import pdb


def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


idx = np.arange(10)
np.random.shuffle(idx)
np_idx = idx
idx = torch.LongTensor(idx)

sorted_idx, sorter = torch.sort(idx)
unsorter = get_unsort_idx(sorter)
original_idx = sorted_idx[unsorter]

np_sorter = np.argsort(np_idx)
np_unsorter = np.argsort(np_sorter)
np_sorted_idx = np.sort(np_idx)
original_np_idx = np_sorted_idx[np_unsorter]


pdb.set_trace()
