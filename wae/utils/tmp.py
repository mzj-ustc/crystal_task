#!/usr/bin/env python3

import numpy as np

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

    for i in range(ncluster):
        score = scores[i]
        print(f"Cluster {i}: {counts[i]}")
        sorted_ids = np.argsort(score)
        for j in range(l):
            print(f"{score[sorted_ids[j]]} %, {lookup[sorted_ids[j]]}")
        print('\n')
    exit()
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

match_prototype_min_min('saves/LJ_24-07-23-20-06')
