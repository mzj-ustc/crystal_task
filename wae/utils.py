#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as normal
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from itertools import chain as ichain
import time
from IPython.display import clear_output
import sys
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
def train_for_classifier(T_class,data_loader):
    decay = 2.5*1e-4
    loss_class=nn.CrossEntropyLoss()
    optimizer_C = optim.Adam(T_class.parameters(), lr = 1e-3,weight_decay = decay)
    epochs =  200
    for epoch in range(epochs):
        episodic_loss_c = []

        for idx, batch in enumerate(data_loader):
            T_class.train()
            T_class.zero_grad()
            optimizer_C.zero_grad()
            x, y = batch
            y_pred=T_class(x.to(device))
            loss_c=loss_class(y_pred,y.to(device))
            loss_c.backward()
            optimizer_C.step()
            episodic_loss_c.append(loss_c.detach().item())
        #print(loss_c.detach().item())


    
def train_for_interface(interface,data_loader,JH_class):
    optimizer_I = optim.Adam(interface.parameters(), lr = 1e-4)
    y2=torch.tensor([0.5,0.5]*64).reshape(-1,2).to(device)
    loss_class=nn.CrossEntropyLoss()
    epochs =  200
    for epoch in range(epochs):
        episodic_loss_i = []

        for idx, batch in enumerate(data_loader):
            interface.train()
            interface.zero_grad()
            optimizer_I.zero_grad()
            x, y = batch
            x_temp=x[:,1].reshape(-1,1).to(device)
            y_pred=interface(x_temp)
            x_temp1=torch.cat((y_pred,x_temp),axis=-1)
            prob= JH_class(x_temp1)
            loss_c=loss_class(prob,y2)
            loss_c.backward()
            optimizer_I.step()
            episodic_loss_i.append(loss_c.detach().item())
        #print(loss_c.detach().item())  

def find_latest(name):
    '''
    Find the most recent saved model.
    '''

    dname = './saves/'
    fsaves = os.listdir(dname)
    items = [fsave.split('_')[-1].split('.') for fsave in fsaves]
    dates = []
    for item in items:
        if item[1] == 'pt':
            dates.append(item[0])
    dates_sorted = sorted(dates)
    date_last = dates_sorted[-1]

    fname = dname + name + '_' + date_last + '.pt'
    return fname

def match_prototype_ratio(prefix):
    '''
    Match a group of structures to a set of ideal structures.
    
    Algorithm: 
    For a specific structure, calculate its distances with all prototypes in latent space.
    The prototype has the minimum distance will be its crystal type.
    Determine the crystal structure for all samples in a group.

    Input: prefix [date] of model.
    output: save percentage of prototypes in a group to prefix.report
    
    '''

    with open('indices_table.txt', 'r') as fid:
        lines = fid.readlines()
        
    lookup = []    
    for lino, line in enumerate(lines):
        lookup.append(line.split()[1])

    datas = np.loadtxt(prefix + '.out')
    epsilons, rs = datas[:, 0], datas[:, 1]
    latents = datas[:, 2:-2]
    z1, z2, z3 = latents[:, 0], latents[:, 1], latents[:, 2]
    labels = datas[:, -2].astype('int')
    labels_LS = datas[:, -1].astype('int')
    
    dat = np.loadtxt(prefix + '.score')
    
    #print(dat.shape)
    #print(datas.shape)
    
    ncluster = np.max(labels) + 1
    w, l = dat.shape
    counts = np.zeros(ncluster)

    proto_ids = np.argmin(dat, axis=-1)
    scores = np.zeros((ncluster, l))
    for i in range(w):
        scores[labels[i], proto_ids[i]] += 1
        counts[labels[i]] += 1

    fid = open(prefix + '.report', 'w')
    
    #print(counts)
    for i in range(ncluster):
        scores[i] /= counts[i]
        score = scores[i]
        
        sorted_ids = np.argsort(-score)
        fid.writelines(f"Cluster {i}:\n")
        for j in range(l):
            prob = score[sorted_ids[j]] * 100
            if prob > 0:
                fid.writelines(f"{prob} %, {lookup[sorted_ids[j]]}\n")

def match_prototype_min_min(prefix):
    '''
    Match a group of structures to a set of ideal structures.
    
    Algorithm: 
    For a specific structure, calculate its distances with all prototypes in latent space.
    Find the minimum distance in a group with respect to a prototype.

    Input: prefix [date] of model.
    output: save percentage of prototypes in a group to prefix.report
    
    '''

    with open('indices_table.txt', 'r') as fid:
        lines = fid.readlines()
        
    lookup = []    
    for lino, line in enumerate(lines):
        lookup.append(line.split()[1])

    datas = np.loadtxt(prefix + '.out')
    epsilons, rs = datas[:, 0], datas[:, 1]
    latents = datas[:, 2:-2]
    z1, z2, z3 = latents[:, 0], latents[:, 1], latents[:, 2]
    labels = datas[:, -2].astype('int')
    labels_LS = datas[:, -1].astype('int')
    
    dat = np.loadtxt(prefix + '.score')
    
    #print(dat.shape)
    #print(datas.shape)
    
    ncluster = np.max(labels) + 1
    w, l = dat.shape

    scores = np.ones((ncluster, l)) * 1000
    counts = np.zeros(ncluster)
    for i in range(w):
        counts[labels[i]] += 1
        for j in range(l):
            if scores[labels[i], j] > dat[i, j]:
                scores[labels[i], j] = dat[i, j]

    fid = open(prefix + '.report', 'w')
    for i in range(ncluster):
        score = scores[i]
        fid.writelines(f"Cluster {i}: {counts[i]}\n")
        sorted_ids = np.argsort(score)
        for j in range(l):
            fid.writelines(f"{score[sorted_ids[j]]}, {lookup[sorted_ids[j]]}\n")
