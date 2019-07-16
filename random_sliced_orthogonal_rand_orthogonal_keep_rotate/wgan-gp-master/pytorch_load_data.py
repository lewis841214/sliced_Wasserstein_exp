from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

# Ignore warnings
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

# temp=np.load('./data/down_sample1.npy')
# temp=np.reshape(temp,(1074*6),)
# test=np.load('/home/lewis/Desktop/1project/styleMuseGAN/down_sample6.npy')
# print(test.shape)
level = []
for i in range(9):
    if i == 0:
        level.append(1)
    else:
        level.append(3 ** 1 * 2 ** (i - 1))
level.reverse()
print(level)

"""
temp=[]
for k in range(9):
    if k==0:
        number_in_section=1
    else:
        number_in_section=3**1*2**(k-1)
    temp.append(np.load('./data/down_sample{}.npy'.format(number_in_section)))
    print(temp[k].shape)

"""


# inherite from the class Dataset
# level from low to high 0,1,2,...,9
# the lower the level, the higher the nuber in each section
class Get_up_sampling_data(Dataset):

    def __init__(self, level_number, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rolls = np.load('./data/down_sample{}.npy'.format(level[level_number]))
        self.transform = transform

    def __len__(self):
        return self.rolls.shape[0]

    def __getitem__(self, idx):
        rollss = self.rolls
        sample = rollss[idx]
        # print(sample.shape)
        # sample=pickle.load(file_pathss[idx])
        #
        """
        here we need to select which downsampling we want to use

        """
        if self.transform:
            sample = self.transform(sample)

        return sample


data_level = Get_up_sampling_data(1)
# plt.imshow(data_level[10005][1].transpose())
# plt.show()
dataload = DataLoader(data_level, batch_size=4, shuffle=True, num_workers=4)
# print(dataload[0])
for i_batch, sample in enumerate(dataload):
    print(sample.size())