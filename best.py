#!/usr/bin/env python
__author__ = 'arenduchintala'
import os
import glob
import sys

acc_list = []
ppl_list = []
log = []
for file in glob.glob(sys.argv[1] + "/*.pt"):
    file = file.split('/')[-1]
    epoch = int(file.split('_e')[-1].split('.')[0])
    acc = float(file.split('/')[-1].split('_')[2])
    acc_list.append((acc, file))
    ppl = float(file.split('/')[-1].split('_')[4])
    ppl_list.append((ppl, file))
    log.append((epoch, acc, ppl))


log_file = open(sys.argv[1] + "/training.acc.ppl.log", 'w', encoding='utf-8')
log.sort()
for l in log:
    log_file.write(' '.join([str(i) for i in l]) + '\n')
log_file.close()

print(sys.argv[1])
acc_list.sort(reverse=True)
ppl_list.sort()
print(acc_list[0])
print(ppl_list[0])
for i in range(len(acc_list)):
    if i == 0:
        os.symlink(acc_list[i][1], sys.argv[1] + '/best')
        os.symlink(ppl_list[i][1], sys.argv[1] + '/best.ppl')
    else:
        os.remove(sys.argv[1] + '/' + acc_list[i][1])
