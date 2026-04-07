#!/usr/bin/env python

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.mixture import GaussianMixture
import scipy as sp
import math

from multiprocessing import Pool

class trainset(Dataset):
    def __init__(self, in_features):
        self.in_features = in_features

    
    def __getitem__(self, index):
        in_feature = torch.tensor(self.in_features[index],dtype=torch.float)
        return in_feature

    def __len__(self):
        return len(self.in_features)

def load_data(fin, fout, batch_size):
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

    in_features = np.load(fout)

    t2 = np.load(fin)
    
    train_data = trainset(in_features)
    # use drop_last to drop the last incomplete batch
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader,t2,in_features

def process_result(enc,ljpq,K):
    kmeans = GaussianMixture(n_components=K, max_iter=300,tol=0.0001, n_init=10, init_params='kmeans').fit(enc)
    #kmeans = AgglomerativeClustering(n_clusters=K)
    labels=kmeans.fit_predict(enc)
    label_prop_model = LabelSpreading('knn',n_neighbors=30,alpha=0.1)
    label_prop_model.fit(ljpq, labels)
    labels1=label_prop_model.predict(ljpq)
    #plt.scatter(ljpq,enc[:,1],c=labels1,s = 15,cmap='Paired')

    return labels,labels1





