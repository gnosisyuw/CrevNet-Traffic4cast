import os
import random
import numpy as np
import torch
import pickle as pkl
import h5py
import cv2



class TRAFFIC(object):
    def __init__(self, train, data_root, seq_len=13,input_len = 10):
        self.train = train
        self.data_type = "training" if self.train else "test"
        self.data_root = data_root
        self.dirs = os.listdir("{}".format(self.data_root))
        self.seq_len = seq_len
        self.city = ['Moscow']  # ['Berlin', 'Istanbul', 'Moscow']
        self.seed_set = False
        self.file_len = []
        self.flist = {}
        for i in self.city:
            folder = self.data_root+i+'/'+i+'_'+self.data_type
            fn =  os.listdir(folder)
            fn.sort()
            self.flist[i] = fn
            self.file_len.append(len(fn))
        self.data = None
        self.c = 0
        self.f = 0
        self.p = 0
        self.pos = [57, 114, 174, 222, 258]
        self.pos2 = [30, 69, 126, 186, 234]
        self.input_len=input_len

    def get_sequence(self):
        t = self.seq_len
        seq = []
        while True:
            k = self.p
            cc = self.city[self.c]
            files = self.flist[cc]
            cf = files[self.f]
            if self.p == 0:
                folder = self.data_root + cc + '/' + cc + '_' + self.data_type + '/'
                self.data = np.load(folder +cf)
                if cc == 'Berlin':
                    start_time = self.pos2[self.p] - self.input_len
                else:
                    start_time = self.pos[self.p] - self.input_len
                self.p +=1
            elif self.p < 4:
                if cc == 'Berlin':
                    start_time = self.pos2[self.p] - self.input_len
                else:
                    start_time = self.pos[self.p] - self.input_len
                self.p +=1
            elif self.p ==4:
                if cc == 'Berlin':
                    start_time = self.pos2[self.p] - self.input_len
                else:
                    start_time = self.pos[self.p] - self.input_len

                self.p = 0

                if self.f == self.file_len[self.c]-1:
                    self.f = 0
                    self.c +=1
                else:
                    self.f += 1
            start_time = start_time -3
            for i in range(t):
                im = self.data[start_time + i]
                patch = np.zeros((1, 436, 3), dtype=int)
                patch2 = np.zeros((496, 12, 3), dtype=int)
                im = np.concatenate((im, patch), axis=0)
                im = np.concatenate((im, patch2), axis=1)
                seq.append(im/255.0)
            print(k,cc,cf)

            # return np.array(seq),k
            return np.array(seq)

    def __getitem__(self, index):
        # if not self.seed_set:
        #     self.seed_set = True
        #     random.seed(index)
        #     np.random.seed(index)
            # torch.manual_seed(index)
        # return torch.from_numpy(self.get_sequence()[0]),self.get_sequence()[1]
        return torch.from_numpy(self.get_sequence())



    def __len__(self):
        return len(self.dirs) * 36 * 5  # arbitrary


if __name__ == "__main__":
    a = TRAFFIC(False, "/home/wei/Desktop/city/")
    for i in range(2000):
        c = a.get_sequence()
        print(i,c[1],c[2])
















# import os
# import random
# import numpy as np
# import torch
# import pickle as pkl
# import h5py
# import cv2
#
#
#
# class TRAFFIC(object):
#     def __init__(self, train, data_root, seq_len=13):
#         self.train = train
#         self.data_type = "training" if self.train else "test"
#         self.data_root = data_root
#         self.dirs = os.listdir("{}".format(self.data_root))
#         self.seq_len = seq_len
#         self.city = ['Berlin', 'Istanbul', 'Moscow']
#         self.seed_set = False
#         self.flist = []
#         for i in self.city:
#             folder = self.data_root+i+'/'+i+'_'+self.data_type
#             self.flist.append(os.listdir(folder))
#         self.data = None
#         self.count = 0
#
#     def get_sequence(self):
#         t = self.seq_len
#         while True:
#             if self.count == 0:
#                 c_idx = np.random.randint(len(self.city))
#                 c = self.city[c_idx]
#                 folder = self.data_root + c + '/' + c + '_' + self.data_type + '/'
#                 files = self.flist[c_idx]
#                 s_idx = np.random.randint(len(files))
#                 f = files[s_idx]
#                 fr = h5py.File(folder + f, 'r')
#                 a_group_key = list(fr.keys())[0]
#                 self.data = list(fr[a_group_key])
#                 self.count += 1
#             elif self.count >= 100:
#                 self.count = 0
#             else:
#                 self.count += 1
#                 print(self.count)
#
#             seq = []
#             start_time = np.random.randint(0,len(self.data)-t)
#             flip = np.random.randint(2)
#
#             for i in range(t):
#                 im = self.data[start_time + i]
#                 if flip == 1:
#                     im = cv2.flip(im, 1)
#                 patch = np.zeros((1, 436, 3))
#                 patch2 = np.zeros((496, 4, 3))
#                 im = np.concatenate((im, patch), axis=0)
#                 im = np.concatenate((im, patch2), axis=1)
#                 seq.append(im/255.0)
#             return np.array(seq)
#
#     def __getitem__(self, index):
#         if not self.seed_set:
#             self.seed_set = True
#             random.seed(index)
#             np.random.seed(index)
#             # torch.manual_seed(index)
#         return torch.from_numpy(self.get_sequence())
#
#
#
#     def __len__(self):
#         return len(self.dirs) * 36 * 5  # arbitrary
#
#
# if __name__ == "__main__":
#     a = TRAFFIC(True, "/home/wei/Desktop/traffic/")
#     for _ in range(1000):
#         c = a.get_sequence()
#         print(np.shape(c))