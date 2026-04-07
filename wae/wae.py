#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import init

device="cuda:0"


def weights_init(layer):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(layer.weight)
        #nn.init.normal_(layer.weight)
        #nn.init.zeros_(layer.weight)
        

class WAE(nn.Module):
    def __init__(self,input_dim, middle_dim=512,latent_dim=20, device=device):
        super(WAE, self).__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.device = device
        self.encoder = nn.Sequential(nn.Linear(input_dim,middle_dim),
                                    nn.LeakyReLU(0.2,inplace = False),
                                    nn.Linear(middle_dim, middle_dim),
                                    nn.LeakyReLU(0.2,inplace = False),
                                    # nn.Linear(middle_dim, middle_dim),
                                    # nn.LeakyReLU(0.2,inplace = False),
                                    
                                    nn.Linear(middle_dim, latent_dim)
                                    )
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim,middle_dim),
                                    nn.LeakyReLU(0.2,inplace = False),
                                    
                                    nn.Linear(middle_dim,middle_dim),
                                    nn.LeakyReLU(0.2,inplace = False),
                                    # nn.Linear(middle_dim,middle_dim),
                                    # nn.LeakyReLU(0.2,inplace = False),
                           
                                    nn.Linear(middle_dim,input_dim),
                                    nn.Sigmoid()
                                    )
       
        self.decoder.apply(weights_init)
        self.encoder.apply(weights_init)
        
    def encode(self, x):
        x=x.reshape(-1,self.input_dim)
        x=self.encoder(x)
        return x
 
   
    def decode(self, z):
        
        x = self.decoder(z)
        x=x.reshape(-1,self.input_dim)
        return x
 
    def forward(self, x):
        z = self.encode(x)
        x_fake = self.decode(z)
        #loss, reconstruction_loss, MMD_loss = self.loss_func(x_fake, z, x)
        return x_fake, z

    def loss_func(self, x_out, mu, x):
        device = self.device
        ratio = 100
        reconstruction_loss = F.mse_loss(x_out, x.reshape(-1,self.input_dim))
        reconstruction_loss *= ratio
        
        sigma=8
        z = sigma*torch.randn(mu.size()).to(device) 
        MMD_loss = imq_kernel(mu, z, h_dim=self.latent_dim, device = device).to(device)
        MMD_loss = MMD_loss / mu.size(0)
        #KL_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2,dim = 1),dim = 0)
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # KLD_ele = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_ele).mul_(-0.5)
        #print(reconstruction_loss.item())
        #print(reconstruction_loss1.item())
        #print(MMD_loss.item())
        return reconstruction_loss+MMD_loss, reconstruction_loss, MMD_loss



def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int, device): # common kerntl to choose
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t()).to(device)  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x # mm matrix multiplicaiton

    norms_y = Y.pow(2).sum(1, keepdim=True).to(device)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t()).to(device)  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]: # need more study on this
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).to(device)) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
def initialize_weights1(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            init.constant_(m.bias, 1)


class TClassfier(nn.Module):
    def __init__(self,input_dim):
        super(TClassfier, self).__init__()
        self.input_dim=input_dim
        self.network = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(0.2, inplace = False),
                                    nn.Linear(16, 2)
                                    
                                    )
        initialize_weights(self)
                                    
    def forward(self, jh):
        #x = torch.cat((zn,zc), dim = 1)
        x = self.network(jh)
        return x

class interface1(nn.Module):
    def __init__(self,input_dim):
        super(interface1, self).__init__()
        self.input_dim=input_dim
        self.network = nn.Sequential(nn.Linear(input_dim, 1,bias=True)           
                                    )
        initialize_weights1(self)
                                    
    def forward(self, j):
        x = self.network(j)
        return x
    
class interface2(nn.Module):
    def __init__(self,input_dim):
        super(interface2, self).__init__()
        self.input_dim=input_dim
        self.network = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, 1),
                                    nn.Tanh()
                                    )
        initialize_weights(self)                      
    def forward(self, j):
        x = self.network(j)*30
        return x






