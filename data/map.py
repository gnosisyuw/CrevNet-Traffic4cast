import os
import random
import numpy as np
import torch
import pickle as pkl
import h5py
import cv2



class CITY(object):
    def __init__(self, train, data_root, seq_len=13,input_len = 10, gap = 1):
        self.train = train
        self.data_type = "training" if self.train else "test"
        self.data_root = data_root
        self.dirs = os.listdir("{}".format(self.data_root))
        self.seq_len = seq_len
        self.city = ['Berlin', 'Istanbul', 'Moscow'] if self.train else ['Berlin', 'Istanbul', 'Moscow']#['Berlin', 'Istanbul', 'Moscow']
        self.seed_set = False
        self.flist = []
        for i in self.city:
            folder = self.data_root+i+'/'+i+'_'+self.data_type
            self.flist.append(os.listdir(folder))
        self.data = None
        self.count = 0
        self.pos = [57, 114, 174, 222, 258]
        self.pos2 = [30, 69, 126, 186, 234]
        self.input_len = input_len
        self.c = None
        self.tc = 0
        self.tt = 0
        self.tf = 0
        self.gap = gap
        self.map = []
        self.map.append(np.load('./data/filterb.npy'))
        self.map.append(np.load('./data/filteri.npy'))
        self.map.append(np.load('./data/filterm.npy'))
        # print(1)

    def get_sequence(self):
        # t = self.seq_len
        t = self.seq_len * self.gap
        seq = []
        while True:
            if self.train:
                c_idx = np.random.randint(len(self.city))
                self.c = self.city[c_idx]
                folder = self.data_root + self.c + '/' + self.c + '_' + self.data_type + '/'
                files = self.flist[c_idx]
                s_idx = np.random.randint(len(files))
                f = files[s_idx]
                self.data = np.load(folder + f)
                start_time = np.random.randint(0, len(self.data) - t)
                cc = c_idx
            else:
                self.c = self.city[self.tc]
                cc = self.tc
                folder = self.data_root + self.c + '/' + self.c + '_' + self.data_type + '/'
                files = self.flist[self.tc]
                f = files[self.tf]
                self.data = np.load(folder + f)
                if self.c =='Berlin':
                    start_time = self.pos2[self.tt] - self.input_len - 3
                else:
                    start_time = self.pos[self.tt] - self.input_len - 3
                # print(self.tc,self.tf,self.tt)
                self.tt += 1
                if self.tt == len(self.pos):
                    self.tt = 0
                    self.tf += 1
                if self.tf >=72:
                    self.tf = 0
                    self.tc +=1
                if self.tc ==len(self.city):
                    self.tc = 0

            for i in range(self.seq_len):
                im = self.data[start_time + i * self.gap]


                patch = np.zeros((1, 436, 3), dtype=int)
                patch2 = np.zeros((496, 12, 3), dtype=int)
                im = np.concatenate((im, patch), axis=0)
                im = np.concatenate((im, patch2), axis=1)

                patch3 = np.zeros((496, 448, 1), dtype=int)
                im = np.concatenate((im, patch3), axis=2)

                im[:,:,3] = self.map[cc]*255.

                seq.append(im/255.0)



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
    a = CITY(True, "/home/wei/Desktop/city/")
    for _ in range(1000):
        c = a.get_sequence()
        print(np.shape(c))
