#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from optparse import OptionParser
import torch.optim as optim
import time
from datetime import datetime

def weights_init(layer):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(layer.weight)

class FCNN(nn.Module):
    '''
    A fully connected neural network, which maps the latent space to parameter space.
    '''
    
    def __init__(self):
        latent_dim = 3
        param_dim = 2
        midlayer = 16
    
        super(FCNN, self).__init__()

        self.model = nn.Sequential(nn.Linear(latent_dim, midlayer),
                                   nn.LeakyReLU(0.2, inplace = False),
                                   nn.Linear(midlayer, midlayer),
                                   nn.LeakyReLU(0.2, inplace = False),
                                   nn.Linear(midlayer, midlayer),
                                   nn.LeakyReLU(0.2, inplace = False),
                                   nn.Linear(midlayer, midlayer),
                                   nn.LeakyReLU(0.2, inplace = False),
                                   nn.Linear(midlayer, midlayer),
                                   nn.LeakyReLU(0.2, inplace = False),
                                   nn.Linear(midlayer, param_dim)
                                   )
        self.model.apply(weights_init)

    def forward(self, x):
        out = self.model(x)
        return out

    def loss_func(self, x_real, x_pred):
        '''
        An MSE loss between a real x and a predicted x.
        '''

        loss = F.mse_loss(x_real, x_pred)
        
        return loss

class mydata(Dataset):
    def __init__(self, in_features):
        self.in_features = in_features

    
    def __getitem__(self, index):
        in_feature = torch.tensor(self.in_features[index],dtype=torch.float)
        return in_feature

    def __len__(self):
        return len(self.in_features)

def load_data(fin, batch_size):
    '''
    Load datas from fin and fout.
    
    Parameters:
    fin: 
        A file contains inputs to a simulator.
    fout: 
        A file contains ouputs from a simulator when given inputs in fin.

    Return:
    a tuple contains data_loader, in_features, out_features
    '''

    datas = np.loadtxt(fin)

    params = datas[:, :2]       # The first two columns are parameters
    latents = datas[:, 2:5]     # The next three columns are hidden features

    train_data = datas[:, :5]   # The first five columns will be our train data
    train_data = mydata(train_data)
    # use drop_last to drop the last incomplete batch
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader, params, latents

if __name__ == '__main__':
    
    
    parser = OptionParser()
    parser.add_option("--filename", dest="fname", default="None",
                      help="A filename contains parameters and latent features.")
    (options, args) = parser.parse_args()
    fname=options.fname

    if fname == "None":
        print("Please input a file name.")
        exit()

    prefix = fname.split('.')[0]

    epoches = 2000
    BATCH_SIZE = 256
    device_id = 0
    device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")
    
    train_datas, params, latents = load_data(prefix + '.out', batch_size = BATCH_SIZE)
    fc_model = FCNN().to(device)
    beta1, beta2 = 0.5, 0.9
    lr = 5e-4
    optimizer= optim.Adam(fc_model.parameters(), lr = lr, betas = (beta1, beta2))

    time_tot = 0
    for epoch in range(epoches):
        episodic_loss_tot = []
        
        start_time = time.time()
        for idx, batch in enumerate(train_datas):
            # trigue train mode
            fc_model.train()
            fc_model.zero_grad()
            optimizer.zero_grad()

            y_real, x_real = batch[:, :2].to(device), batch[:, 2:5].to(device)
            y_pred = fc_model(x_real)
            loss = fc_model.loss_func(y_real, y_pred)

            loss.backward()
            optimizer.step()

            episodic_loss_tot.append(loss.item())

        time_tot += time.time() - start_time
        strs ="Episode: {}, loss: {}, time: {} seconds".format(epoch, np.array(episodic_loss_tot).mean(), time.time() - start_time) 
        print(strs)

    # Evaluation
    fc_model.eval()
    latents_proto = torch.Tensor(np.loadtxt(prefix + '.ptype')).to(device)
    params_proto = fc_model(latents_proto).detach().cpu().numpy()

    np.savetxt(prefix + '.pparam', params_proto)
    
