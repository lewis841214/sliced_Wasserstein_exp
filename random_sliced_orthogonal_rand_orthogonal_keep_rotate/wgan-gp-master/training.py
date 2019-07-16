import imageio
import numpy as np
#import minpy.numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from distributions import rand_cirlce2d
import matplotlib.pyplot as plt
import time
import random

def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    #print(type(projections))
    #print(projections.shape)
    return torch.from_numpy(projections).type(torch.FloatTensor).cuda()

def orthogonal_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    #projections = [w / np.sqrt((w**2).sum())  # L2 normalization
    #               for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.identity(embedding_dim)
    projections = [w / np.sqrt((w ** 2).sum())  # L2 normalization
                   for w in projections]
    projections = np.asarray(projections)[:400]
    #print(type(projections))
    return torch.from_numpy(projections).type(torch.FloatTensor).cuda()

ini=np.identity(420)
def rotate(input_array, i,j, angle):
    n=input_array.shape[0]
    rotation=np.identity(n)
    rotation[i][i]=np.cos(angle/180*np.pi)
    rotation[i][j]=-np.sin(angle/180*np.pi)
    rotation[j][i]=np.sin(angle/180*np.pi)
    rotation[j][j]=np.cos(angle/180*np.pi)
    return np.matmul(input_array,rotation)
def permute_n_rotate(input_):

    dimension=ini.shape[0]
    arr=[x for x in range(dimension)]
    random.shuffle(arr)
    #print(int(dimension/2))
    angle=np.random.uniform(0,1,size=(int(dimension/2)))
    for i in range(int(dimension/2)):
        #print(arr)
        input_=rotate(input_,arr[i*2],arr[i*2+1],45+angle[i]/100)
    return input_

def random_orthogonal_projections(embedding_dim, num_samples=50,project_matrix=ini):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    #projections = [w / np.sqrt((w**2).sum())  # L2 normalization
    #               for w in np.random.normal(size=(num_samples, embedding_dim))]
    #ini = permute_n_rotate(project_matrix)
    projections=[w / np.sqrt((w**2).sum())  # L2 normalization
                for w in project_matrix]
    projections=np.asarray(projections)
    #print(type(projections))
    return torch.from_numpy(projections).type(torch.FloatTensor).cuda()


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cuda',projection_mode='orthogonal',project_matrix=ini):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    distribution_samples=distribution_samples.to('cuda')
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    #projections = rand_projections(embedding_dim, num_projections).to(device)
    if projection_mode=='orthogonal':
        projections = orthogonal_projections(embedding_dim, num_projections).to(device)
    #random_orthogonal_projections
#   #random orthogonal projection
    if projection_mode=='rand_orthogonal':
        projections = random_orthogonal_projections(embedding_dim, num_projections,project_matrix=project_matrix).to(device)
    if projection_mode=='rand':
        #('asdfasdfasdfasdf')
        projections = rand_projections(embedding_dim, num_projections).to(device)

    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples,
                                distribution_fn=rand_cirlce2d,
                                num_projections=50,
                                p=2,
                                device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw random samples from latent space prior distribution
    z = distribution_fn(batch_size).to(device)
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, z,
                                       num_projections, p, device)
    return swd





################################################
#DESIGN sliced loss function
#
#sliced wd doesn't need to train discriminator
###############################################
#######################
######################
####################
######################
####################
#####################
#######################
#sliced_loss=1









class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=2, print_every=1000,
                 use_cuda=False,batchsize=10000,sliced_loss=0,sliced_loss_projection_mode='rand'):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.batchsize=batchsize
        self.sliced_distance=0
        self.sliced_loss=sliced_loss
        self.projection_matrix=np.identity(420)
        self.lowest_rand_ortho_sliced_distance=10000
        self.lowest_rand_sliced_distance=10000
        self.sliced_loss_projection_mode=sliced_loss_projection_mode
        print('sliced_projection_mode',self.sliced_loss_projection_mode)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        #print(223)
        self.show_generated_data=generated_data
        # Calculate probabilities on real and generated data
        data = Variable(data)
        #print('data_size',data.size())
        #print('generated_data', generated_data.size())
        if self.use_cuda:
            data = data.cuda()
        #print('data_size',data.size())

        d_real = self.D(data)
        d_generated = self.D(generated_data)
        #################################
        #define sliced distance here
        #################################
        #print(data.view(20,-1)size())
        self.ortho_sliced_distance=_sliced_wasserstein_distance(data.view(self.batchsize,-1),generated_data.view(self.batchsize,-1),num_projections=400,p=2,device='cuda',projection_mode='rand_orthogonal')
        self.rand_sliced_distance=_sliced_wasserstein_distance(data.view(self.batchsize,-1),generated_data.view(self.batchsize,-1),num_projections=400,p=2,device='cuda',projection_mode='rand')

        #print(self.sliced_distance)
        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        if self.sliced_loss==1:
            #print('d liced confirmed')
            d_loss = self.sliced_distance
        else:
            self.D_opt.zero_grad()
            d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
            #print('hi')
            d_loss.backward()
            self.D_opt.step()

        # Record loss
#        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)

        # g_loss = - d_generated.mean()
        ############################################################################################
        #
        ###################################################################################################
        #print(type(data))
        #print(type(generated_data))
        if self.sliced_loss==1:
            self.projection_matrix=permute_n_rotate(self.projection_matrix)
            #print'sliced confirmed!')
            #self.sliced_distance=_sliced_wasserstein_distance(data.view(self.batchsize,-1).to('cuda'),generated_data.view(self.batchsize,-1).to('cuda'),num_projections=400,p=2,device='cuda')
            #print()
            g_loss=_sliced_wasserstein_distance(data.view(self.batchsize,-1).to('cuda'),generated_data.view(self.batchsize,-1).to('cuda'),num_projections=400,p=2,device='cuda',projection_mode=self.sliced_loss_projection_mode,project_matrix=self.projection_matrix)
        else:
            g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            #print(data[0])
            #self._critic_train_iteration(data[0])
            #print(234)
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

            #if i % self.print_every == 0:
                #print("Iteration {}".format(i + 1))
                #print("D: {}".format(self.losses['D'][-1]))
                #print("GP: {}".format(self.losses['GP'][-1]))
                #print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                #print('sliced_distance=',self.sliced_distance)

                #print(show.shape)
                #import matplotlib.pyplot as plt
                #plot out the result

                """
                printing out the result here
                """

                #show = self.show_generated_data.cpu().clone().detach().numpy()
                #plt.imshow(show[250][1].transpose())
                #plt.show()

                #if self.num_steps > self.critic_iterations:
                 #   print("G: {}".format(self.losses['G'][-1]))
    ##################################################################################
    #input dataloader to here
    ################################################################################

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(32))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        start=time.time()
        sl=0
        sum_rand_ortho_sliced_distance100=0
        sum_rand_sliced_distance100=0
        for epoch in range(epochs):
            self._train_epoch(data_loader)
            #self.lowest_rand_ortho_sliced_distance = 10000
            #self.lowest_rand_sliced_distance = 10000
            sum_rand_ortho_sliced_distance100=sum_rand_ortho_sliced_distance100+self.ortho_sliced_distance.data.item()
            sum_rand_sliced_distance100=sum_rand_sliced_distance100+self.rand_sliced_distance.data.item()
            if sl==0:
                self.lowest_rand_ortho_sliced_distance=self.ortho_sliced_distance
                self.lowest_rand_sliced_distance=self.rand_sliced_distance
            sl=sl+1
            if sl == 200:
                self.lowest_rand_ortho_sliced_distance = self.ortho_sliced_distance
                self.lowest_rand_sliced_distance = self.rand_sliced_distance
            if self.lowest_rand_ortho_sliced_distance.data.item()>self.ortho_sliced_distance.data.item():
                #print('hi')
                self.lowest_rand_ortho_sliced_distance=self.ortho_sliced_distance
            if self.lowest_rand_sliced_distance.data.item()>self.rand_sliced_distance.data.item():
                self.lowest_rand_sliced_distance = self.rand_sliced_distance
            #print(self.lowest_rand_ortho_sliced_distance.data.item())
            if epoch%100==0:
                file = open('record' + self.sliced_loss_projection_mode + '.txt', 'a')
                file.write(str(epoch) + '\n')
                file.write('average_rand_ortho_sliced_distance=' + str(sum_rand_ortho_sliced_distance100 / 100) + '\n')
                file.write('average_rand_sliced_distance' + str(sum_rand_sliced_distance100 / 100) + '\n')
                file.close()

                print('average_rand_ortho_sliced_distance=',sum_rand_ortho_sliced_distance100/100)
                print('average_rand_sliced_distance',sum_rand_sliced_distance100/100)
                sum_rand_ortho_sliced_distance100 = 0
                sum_rand_sliced_distance100 = 0
            #reinitilized the original matrix to prevend error propogation
            if epoch%2000==0:
                self.projection_matrix=np.identity(420)
            if epoch % 100 == 0:
                tem = time.time()
                print("\nEpoch {}".format(epoch + 1))
                if (tem-start)/60<60:
                    torch.set_printoptions(precision=10)
                    print('rand_ortho_sliced_distance=', self.ortho_sliced_distance,'used time=',(tem-start)/60,'min','one epoch spend=',(tem-start)/60/(epoch+1),'mins')
                    print('rand_sliced_distance=', self.rand_sliced_distance,'used time=',(tem-start)/60,'min','one epoch spend=',(tem-start)/60/(epoch+1),'mins')
                    print('lowest_rand_ortho_sliced_distance=',self.lowest_rand_ortho_sliced_distance.data.item())
                    print('lowest_rand_sliced_distance=',self.lowest_rand_sliced_distance.data.item())

                else:
                    print('ortho_sliced_distance=', self.ortho_sliced_distance, 'used time=', (tem - start) / 60, 'min',
                          'one epoch spend=', (tem - start) / 60 / (epoch + 1), 'mins')
                    print('rand_sliced_distance=', self.rand_sliced_distance, 'used time=', (tem - start) / 60, 'min',
                          'one epoch spend=', (tem - start) / 60 / (epoch + 1), 'mins')
                    print('lowest_rand_ortho_sliced_distance=', self.lowest_rand_ortho_sliced_distance.data.item())
                    print('lowest_rand_sliced_distance=', self.lowest_rand_sliced_distance.data.item())
            # now i control the epoch to be lower than 3000
            #if (epoch<1000 and epoch % 50 == 1) or (epoch % 300 == 2): #or (epoch>1000 and epoch % 1000 == 2) :
            if (epoch %3000 == 7) :
                show = np.around(self.show_generated_data.cpu().clone().detach().numpy())
                #print(self.show_generated_data.cpu().clone().detach().numpy())
                #print(show)
                #print(show.shape)
                #plt.imshow(show[250].transpose(1, 0, 2)[0])
                #plt.show()
                #plt.imshow(show[499].transpose(1, 0, 2)[0])
                #plt.show()

            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                #img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                img_grid = np.transpose(img_grid.numpy())
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        print('lowest_rand_sliced_distance',self.lowest_rand_sliced_distance.data.item())
        print('lowest_rand_ortho_sliced_distance',self.lowest_rand_ortho_sliced_distance.data.item())
        if save_training_gif:
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
                            training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
