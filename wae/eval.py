#!/usr/bin/env python

from CNAG1 import *
import yaml
import os
from optparse import OptionParser
from wae import *
from data_process import *
from utils import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE
from datetime import datetime

# Load model parameters from a designated file or the latest file
#fsave = find_latest(name)
def sort_label(labels, labels_LS):
    '''
    Re-index labels. Make the most number of structures in one type to be index 0. 
    Less structures, larger label index.
    '''

    # count number of structures in each group
    nbin = int(np.max(labels))
    eps = 0.00001
    hist, bin_edges = np.histogram(labels, bins=nbin+1, range=(-eps, nbin-eps+1))

    # sort labels according to number of structures
    indices = np.argsort(-hist)

    # build a look up table according to the mapping rule
    l = len(indices)
    ihash = np.zeros(l)
    for i, x in enumerate(indices):
        ihash[x] = i

    # Reindex labels and labels_LS
    labels = np.array([ihash[label] for label in labels])
    labels_LS = np.array([ihash[label] for label in labels_LS])

    return labels, labels_LS


# ############################################################
# Read yaml input settings
parser = OptionParser()
parser.add_option("--model", dest="model", default=None,
                  help="Specify a saved model.")

(options, args) = parser.parse_args()
if options.model == None:
    fsave = find_latest(name)
else:
    fsave = options.model

prefix  = '.'.join(fsave.split('.')[:-1])
configure = prefix + '.yaml'
with open(configure, encoding="utf-8") as f:
    temp = yaml.safe_load(f)

name = temp['name']
fin = temp['fin']
fout = temp['fout']
latent_space_zn = temp['latent']
neurons = temp['neurons']
BATCH_SIZE = temp['batch']
epoches = temp['epoches']
lr = temp['lr']
step_size = temp['step_size']
device_id = temp['device']
is_restart = temp['is_restart']
is_train = temp['is_train']
K_max = temp['Kmax']
alpha = temp['alpha']
input_f_dim=temp['input_f_dim']


# ############################################################
# Encode the input
print(f"Loading data ...")
data_loader, lj_parameters, order_parameters=load_data(fin,fout,BATCH_SIZE)

lj_parameters=lj_parameters.reshape(-1, input_f_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")

shape_input=order_parameters.shape[1]*order_parameters.shape[2]
#print(order_parameters.shape)

# Define a WAE model
#latent_space_zn = 256
WAE_test=WAE(shape_input, middle_dim = neurons, latent_dim = latent_space_zn, device = device).to(device)

date = datetime.now()
datestr = date.strftime("%y-%m-%d-%H-%M")
try:
    os.mkdir('saves')
except FileExistsError:
    pass

print(f"Load model from {fsave}")
checkpoint = torch.load(fsave)
WAE_test.load_state_dict(checkpoint['model_state_dict'])

print("WAE archtecture:")
print(WAE_test)
total_params = sum(p.numel() for p in WAE_test.parameters())
print(f"Number of parameters: {total_params}")

print("Start evaluating ...")

order_parameters = torch.tensor(order_parameters, dtype=torch.float)

latent_parameters = WAE_test.encode(order_parameters.to(device)).cpu().detach().numpy()


# ############################################################
# Determine the number of clusters and classify clusters
k = CNAK(np.array(latent_parameters),name,alpha=0.0005,gamma=0.2,K_min=2,K_max=K_max)

labels,labels1=process_result(latent_parameters, lj_parameters, k)
labels, labels1 = sort_label(labels, labels1)

#print(lj_parameters.shape, latent_parameters.shape, labels.shape, labels1.shape)
out = np.concatenate([lj_parameters, latent_parameters, labels[:, np.newaxis], labels1[:, np.newaxis]], axis=1)
l = lj_parameters.shape[-1] + latent_parameters.shape[-1]
fmt = '%10.5f ' * l + '%d ' * 2
print(out.shape, fmt)
np.savetxt(prefix+'.out', out, fmt=fmt)


# ############################################################
# Identify each group using prototypes

fproto = 'proto_qs.npy'         # prototypes' order parameters
proto_qs = np.load(fproto)
proto_qs = torch.tensor(proto_qs, dtype=torch.float)
proto_latents = WAE_test.encode(proto_qs.to(device))

latent_parameters = torch.Tensor(latent_parameters).to(device)

distances = latent_parameters[:, None, :] - proto_latents[None, :, :] # [nsample, nprototype, 5]
distances = torch.sqrt(torch.sum(distances ** 2, axis=-1)) # [nsample, nprototype]
distances = distances.detach().cpu().numpy()
np.savetxt(prefix+'.score', distances, fmt='%10.5f')
np.savetxt(prefix+'.ptype', proto_latents.detach().cpu().numpy())

# decide structure type
match_prototype_min_min(prefix)
