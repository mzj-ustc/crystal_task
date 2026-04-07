#!/usr/bin/env python3

import numpy as np
import os
from add_noise import add_site_noise
from make_structs import ParseCHTAB, ParseXYZ, saveXYZ

if __name__ == '__main__':

    sigma = 0.1

    nsample = 61 * 101 # 1 prototype + 100 noisy samples per prototype (61 prototypes)
    natom = 2744
    ndim = 3
    samples = np.ones((nsample, natom, ndim))

    indices_table="indices_table.txt"
    with open(indices_table, 'r') as fid:
        lines = fid.readlines()
        
    for lino, line in enumerate(lines):
        
        _, dname = line.split()
        fname = dname + '/xyz.cluster'
        chtab = ParseCHTAB()
        xyz = ParseXYZ(fname, chtab)

        a = float(xyz['lattices'][0, 0])
        center = np.mean(xyz['sites'], axis=0)
        xyz['sites'] = xyz['sites'] - center

        samples[lino] = xyz['sites'] * a
        for i in range(100):
            noisy_sample = add_site_noise(xyz, sigma=sigma)
            samples[lino + (i+1)*61] = noisy_sample['sites'] * a


    #print(samples.shape)
    np.save(f'noisy_prototypes_sigma_{sigma}.npy', samples)
