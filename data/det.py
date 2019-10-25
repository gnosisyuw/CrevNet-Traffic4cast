import os
import random
import numpy as np
import torch
import pickle as pkl
import cv2
from PIL import Image


class KITTI(object):
    def __init__(self, train):
        self.train = train
        if self.train:
            self.data = pkl.load(open("/home/stevelab2/Desktop/det/train.dat", "rb"))
            self.data2  =  pkl.load(open("/home/stevelab2/Desktop/det/test.dat", "rb"))
            self.data_len2 = len(self.data2)
        else:
            self.data = pkl.load(open("/home/stevelab2/Desktop/det/test.dat", "rb"))
        self.data_len = len(self.data)
        self.seed_set = False

    def get_sequence(self):

        if self.train:
            shade = np.random.randint(10)
            flip = np.random.randint(2)
        else:
            shade = 0
            flip = 0
        cc = np.random.randint(3)
        if cc == 0 and self.train:
            index = np.random.randint(self.data_len2)
            seq = np.asarray(self.data2[index]).astype(float)
            flip = 1
        else:
            index = np.random.randint(self.data_len)
            seq = np.asarray(self.data[index]).astype(float)
        if flip == 1:
            for j in range(4):
                seq[j] = cv2.flip(seq[j], 1)
        seq /= (255.0+shade)
        return seq

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return 7481 * 36 * 5  # arbitrary
