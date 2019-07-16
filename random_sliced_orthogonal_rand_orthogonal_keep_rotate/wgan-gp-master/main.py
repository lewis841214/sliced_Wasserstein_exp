from __future__ import print_function, division
import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from models import *
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
import os
#import getpass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
batch_size=12444
epochs =25000
sliced_projection_mode='rand'#orthogonal,rand,rand_orthogonal
#sliced_loss=0 choose in string below
img_size = [5,1,84]
gen='cnn' #linear, logistic, cnn
dis='sliced' # linear,logistic,cnn,sliced
lr = 1e-6

betas = (.9, .99)
slicedloss=0
if gen=='cnn':
    #print('#############################')
    generator=Generator_cnn(img_size=img_size, latent_dim=1000, dim=16)
elif gen=='logistic':
    #print('#############################')
    generator=Generator_logistic(img_size=img_size, latent_dim=1000, dim=16)
elif gen=='linear':
    #print('#############################')
    generator=Generator_linear(img_size=img_size, latent_dim=1000, dim=16)
else:
    print('no gen')

if dis=='cnn':
    #print('#############################')
    discriminator = Discriminator_cnn(img_size=img_size, dim=16,batch_size=batch_size)
elif dis=='logistic':
    #print('#############################')
    discriminator = Discriminator_logistic(img_size=img_size, dim=16,batch_size=batch_size)
elif dis=='linear':
    #print('#############################')
    discriminator = Discriminator_linear(img_size=img_size, dim=16,batch_size=batch_size)
elif dis=='sliced':
    discriminator=Discriminator_linear(img_size=img_size, dim=16,batch_size=batch_size)
    slicedloss=1

print(len(data_level))
dataload = DataLoader(data_level, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True)
# print(dataload[0])
name = gen+dis
#########################################
#decide whether to load state dict
##################
load_dict=1
if load_dict==1:
    #generator = Generator_cnn(img_size=img_size, latent_dim=1000, dim=16)
    generator.load_state_dict(torch.load('./state_dict/gen_' + name + '.pt'))
    generator.eval()
    #discriminator=Discriminator_cnn(img_size=img_size, dim=16,batch_size=batch_size)
    #discriminator.load_state_dict(torch.load('./state_dict/dis_' + name + '.pt'))
    #discriminator.eval()
    #generator=torch.load('./gen_rolls.pt')
    #generator.eval()



print(generator)
print(discriminator)

# Initialize optimizers

G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
#trainer add batch size cause
#self.sliced_distance=_sliced_wasserstein_distance(data.view(100,-1),generated_data.view(100,-1),num_projections=400,p=2,device='cuda')
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available(),batchsize=batch_size,sliced_loss=slicedloss,sliced_loss_projection_mode=sliced_projection_mode)
trainer.train(dataload, epochs, save_training_gif=0)

# Save models
name = gen+dis
torch.save(trainer.G.state_dict(), './state_dict/gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './state_dict/dis_' + name + '.pt')
