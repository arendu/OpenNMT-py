#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys, os
import glob

acc_list = []
ppl_list = []
for file in glob.glob(sys.argv[1] + "/*.pt"):
    file = file.split('/')[-1]
    acc = float(file.split('/')[-1].split('_')[2])
    acc_list.append((acc, file))
    ppl = float(file.split('/')[-1].split('_')[4])
    ppl_list.append((ppl, file))

print(sys.argv[1])
acc_list.sort(reverse=True)
ppl_list.sort()
print(acc_list[0])
print(ppl_list[0])
os.symlink(acc_list[0][1], sys.argv[1] + '/best')
os.symlink(ppl_list[0][1], sys.argv[1] + '/best.ppl')
