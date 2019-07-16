import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim),int(img_size[0]*img_size[1]*img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
           # nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),

            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
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

"""

"""
class Discriminator(nn.Module):
    def __init__(self, img_size,dim, batch_size):

        
        super(Discriminator, self).__init__()
        self.batch_size=batch_size
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Linear(int(latent_dim), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),

            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
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

"""


class Discriminator_logistic(nn.Module):
    def __init__(self, img_size, dim, batch_size):
        super(Discriminator_logistic, self).__init__()
        self.batch_size = batch_size
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            # nn.Linear(int(latent_dim), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),

            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            # nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(img_size[0] * img_size[1] * img_size[2], 1)
        )

    def forward(self, input_data):
        # print('input_size',input_data.size())
        x = input_data.view(self.batch_size, -1)
        # print('D_x_size', x.size())
        x = self.image_to_features(x)
        # print('D_x_size', x.size())
        # x = x.view(batch_size, -1)
        return x

class Discriminator_linear(nn.Module):
    def __init__(self, img_size, dim, batch_size):
        super(Discriminator_linear, self).__init__()
        self.batch_size = batch_size
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            # nn.Linear(int(latent_dim), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),

            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            # nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(img_size[0] * img_size[1] * img_size[2], 1)
        )

    def forward(self, input_data):
        # print('input_size',input_data.size())
        x = input_data.view(self.batch_size, -1)
        # print('D_x_size', x.size())
        x = self.image_to_features(x)
        # print('D_x_size', x.size())
        # x = x.view(batch_size, -1)
        return x

class Discriminator_cnn(nn.Module):
    def __init__(self, img_size,dim, batch_size):

        #img_size : (int, int, int)
        super(Discriminator_cnn, self).__init__()
        self.batch_size=batch_size
        self.img_size = img_size
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
         #               padding_mode='zeros')



        """
        self.image_to_features = nn.Sequential(
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(img_size[0] * img_size[1] * img_size[2],1 )
        )
        """



        self.layer1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=(3, 11), stride=(1, 1)),
            # nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU()
        )
        # (4*8)->(6*16)
        self.layer2 = nn.Sequential(
            nn.Conv2d( dim,2* dim, kernel_size=(3, 2), stride=(1, 2)),
            nn.BatchNorm2d(2*dim),
            nn.LeakyReLU()
        )

        # (6*16)-> (8*32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(2* dim, 4*dim, kernel_size=(3, 2), stride=(1, 2)),
            nn.BatchNorm2d(4*dim),
            nn.LeakyReLU()

        )
        # (8*32)-> (10*42)
        self.layer4 = nn.Sequential(

            nn.Conv2d(4*dim, 8 * dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU()
        )
        self.dense1=nn.Linear(128*2*4, 128)
        self.dense2 = nn.Linear(128 , 32)
        self.dense3 = nn.Linear(32, 1)


    def forward(self, input_data):
        #print('input_size',input_data.size())
        #print(input_data.size())
        x = input_data.permute(0,2,1,3)
        x=x.view(self.batch_size,1,10,42)
        x = self.layer1(x)
        #print('intermidiate',x.size())
        x = self.layer2(x)
        #print('intermidiate', x.size())
        x = self.layer3(x)
        #print('intermidiate', x.size())
        x = self.layer4(x)
        #print('D_x_size', x.size())
        x=x.view(self.batch_size,-1)
        x=self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        #print(x.size())
        #x = self.image_to_features(x)
        #print('D_x_size', x.size())
        #x = x.view(batch_size, -1)
        return x



class Generator_cnn(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator_cnn, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        #linear vers


        """
         self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim),int(img_size[0]*img_size[1]*img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid()
            #nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2]))

            #nn.View(img_size[0],img_size[1],img_size[2])
            #nn.ReLU()
        )

        """


        #convolution version
        #
        #1*8 -> 2*4 -> 4*8-> 6*16 -> 8*32 ->10*42 -> 5*84
        self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim), 8*dim*2*4),
            nn.ReLU()
        )
        #(2*4)->(4*8)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=2,stride= 2),
            #nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU()
        )
        #(4*8)->(6*16)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2* dim, kernel_size=(3,2),stride= (1,2)),
            #nn.BatchNorm2d(2* dim),
            nn.LeakyReLU()
        )

        # (6*16)-> (8*32)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2*dim,  dim, kernel_size=(3, 2), stride=(1, 2)),
            #nn.BatchNorm2d( dim),
            nn.LeakyReLU()
        )
        # (8*32)-> (10*42)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d( dim,  1, kernel_size=(3, 11), stride=(1, 1)),
            #nn.BatchNorm2d(2 * dim),
            nn.Sigmoid()
        )




    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        x=x.view(-1,self.dim*8,2,4)
        #print('feature', x.size())
        x=self.layer1(x)
        #print('intermidiate',x.size())
        x = self.layer2(x)
        #print('intermidiate', x.size())
        x = self.layer3(x)
        #print('intermidiate', x.size())
        x = self.layer4(x)
        #print('intermidiate', x.size())
        x=x.view(-1,1,5,84)
        #print('intermidiate', x.size())
        x=x.permute(0,2,1,3)
        #print('G_input_size',input_data.size())
        #print(x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2]).size())
        #print(x.size())
        return x

    def sample_latent(self, num_samples):
        #print('num_samples',num_samples)
        #print('sample_latent_size',torch.randn((num_samples, self.latent_dim)).size())
        return torch.randn((num_samples, self.latent_dim))


class Generator_linear(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator_linear, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        #linear vers


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
        #x=x.view(-1,self.dim*8,2,4)
        #print('feature', x.size())
        #x=self.layer1(x)
        #print('intermidiate',x.size())
        #x = self.layer2(x)
        #print('intermidiate', x.size())
        #x = self.layer3(x)
        #print('intermidiate', x.size())
        #x = self.layer4(x)
        #print('intermidiate', x.size())
        #x=x.view(-1,1,5,84)
        #print('intermidiate', x.size())
        #x=x.permute(0,2,1,3)
        #print('G_input_size',input_data.size())
        #print(x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2]).size())
        return x.view(-1,5,1,84)

    def sample_latent(self, num_samples):
        #print('num_samples',num_samples)
        #print('sample_latent_size',torch.randn((num_samples, self.latent_dim)).size())
        return torch.randn((num_samples, self.latent_dim))


class Generator_logistic(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator_logistic, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        # linear vers

        self.latent_to_features = nn.Sequential(
            nn.Linear(int(latent_dim), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2])),
            nn.Sigmoid(),
            # nn.Linear(int(img_size[0] * img_size[1] * img_size[2]), int(img_size[0] * img_size[1] * img_size[2]))

            # nn.View(img_size[0],img_size[1],img_size[2])
            # nn.ReLU()
        )


    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # x=x.view(-1,self.dim*8,2,4)
        # print('feature', x.size())
        # x=self.layer1(x)
        # print('intermidiate',x.size())
        # x = self.layer2(x)
        # print('intermidiate', x.size())
        # x = self.layer3(x)
        # print('intermidiate', x.size())
        # x = self.layer4(x)
        # print('intermidiate', x.size())
        # x=x.view(-1,1,5,84)
        # print('intermidiate', x.size())
        # x=x.permute(0,2,1,3)
        # print('G_input_size',input_data.size())
        # print(x.view(-1,self.img_size [0],self.img_size [1],self.img_size [2]).size())
        return x.view(-1,5,1,84)


    def sample_latent(self, num_samples):
        # print('num_samples',num_samples)
        # print('sample_latent_size',torch.randn((num_samples, self.latent_dim)).size())
        return torch.randn((num_samples, self.latent_dim))
