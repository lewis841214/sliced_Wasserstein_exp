import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim),int(img_size[0]*img_size[1]*img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid()
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2]))

            #nn.View(img_size[0],img_size[1],img_size[2])
            #nn.ReLU()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        #print('G_input_size',input_data.size())
        #print(x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2]).size())
        return x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2])

    def sample_latent(self, num_samples):
        #print('num_samples',num_samples)
        #print('sample_latent_size',torch.randn((num_samples, self.latent_dim)).size())
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size,dim, batch_size):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()
        self.batch_size=batch_size
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(img_size[0] * img_size[1] * img_size[2],1 )
        )


    def forward(self, input_data):
        #print('input_size',input_data.size())
        x = input_data.view(self.batch_size, -1)
        #print('D_x_size', x.size())
        x = self.image_to_features(x)
        #print('D_x_size', x.size())
        #x = x.view(batch_size, -1)
        return x



class Generator_second_layter(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim),int(img_size[0]*img_size[1]*img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid()
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2]))

            #nn.View(img_size[0],img_size[1],img_size[2])
            #nn.ReLU()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        #print('G_input_size',input_data.size())
        #print(x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2]).size())
        return x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2])

    def sample_latent(self, num_samples):
        #print('num_samples',num_samples)
        #print('sample_latent_size',torch.randn((num_samples, self.latent_dim)).size())
        return torch.randn((num_samples, self.latent_dim))


class Discriminator_second_layter(nn.Module):
    def __init__(self, img_size,dim, batch_size):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()
        self.batch_size=batch_size
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(img_size[0] * img_size[1] * img_size[2],1 )
        )


    def forward(self, input_data):
        #print('input_size',input_data.size())
        x = input_data.view(self.batch_size, -1)
        #print('D_x_size', x.size())
        x = self.image_to_features(x)
        #print('D_x_size', x.size())
        #x = x.view(batch_size, -1)
        return x