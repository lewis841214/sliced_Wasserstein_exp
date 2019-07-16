from __future__ import print_function, division
import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from models import Generator, Discriminator
from training import Trainer

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
        self.rolls=self.rolls.astype(np.float32)
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
        #print(sample.shape)
        return sample


data_level = Get_up_sampling_data(0)
# plt.imshow(data_level[10005][1].transpose())
# plt.show()
batch_size=10000
dataload = DataLoader(data_level, batch_size=batch_size, shuffle=True, num_workers=5,drop_last=True)
# print(dataload[0])



img_size = [5,1,84]

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16,batch_size=batch_size)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 2000
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(dataload, epochs, save_training_gif=0)

# Save models
name = 'rolls'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
