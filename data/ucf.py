import random
import os
import numpy as np
import socket
import torch
from scipy import misc
import cv2
import os.path

class UCF(object):
    def __init__(self, data_root, seq_len, train):
        self.data_root = '%s/UCF/' % data_root
        self.seq_len = seq_len
        self.train = train
        self.actions = os.listdir(self.data_root)
        self.seed_set = False

    def get_sequence(self):
        t = self.seq_len
        while True: # skip seqeunces that are too short
            c_idx = np.random.randint(len(self.actions))
            c = self.actions[c_idx]
            if self.train:
                s_idx = np.random.randint(1, 19)
            else:
                s_idx = np.random.randint(19, 26)
            v = sorted(os.listdir(os.path.join(self.data_root, c)))[s_idx]
            dname = '%s/%s/%s' % (self.data_root, c, v)
            if len(os.listdir(dname)) >= t:
                break
        st = random.randint(1, len(os.listdir(dname))-t)
        seq = []
        shade = np.random.randint(25)
        flip = np.random.randint(2)
        for i in range(st, st+t):
            fname = '%s/%s' % (dname, "image_{:05d}.jpg".format(i))
            # if not os.path.isfile(fname):
            #     print(fname)
            im = cv2.imread(fname)/(255.+shade)
            if flip == 1:
                im = cv2.flip(im,1)
            seq.append(im)
        return np.array(seq)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return 2574258
