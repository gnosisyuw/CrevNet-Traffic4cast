import os
import random
import numpy as np
import torch
import pickle as pkl
import cv2

class CALTECH(object):
    def __init__(self, train, data_root, data_type='sequence', seq_len=20, image_size=64):
        self.train = train
        self.data_type = "train" if self.train else "test"
        self.data_root = '/home/stevelab2/Desktop/caltech'
        self.dirs = os.listdir("{}".format(self.data_root))
        self.seq_len = seq_len
        self.image_size = image_size
        self.processed_data= np.load("{}/caltech_{}.npy".format(self.data_root, self.data_type))
        self.seed_set = False

    def get_sequence(self):
        t = self.seq_len
        shade = np.random.randint(25)
        flip = np.random.randint(2)
        while True:
            id = np.random.randint(len(self.processed_data))
            id2 = np.random.randint(len(self.processed_data[id])-t)
            seq = self.processed_data[id][id2:id2+t]

            for j in range(t):
                if flip == 1:
                    seq[j] = cv2.flip(seq[j],1)/ (255.0+shade)
                else:
                    seq[j] = seq[j] / (255.0 + shade)

            return np.array(seq)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return len(self.dirs) * 36 * 5  # arbitrary


if __name__ == "__main__":
    a = KITTI(True, "/home/wei/PycharmProjects/kitti/data/KITTI/processed")
    for _ in range(1000):
        c = a.get_sequence()
        print(np.shape(c))
