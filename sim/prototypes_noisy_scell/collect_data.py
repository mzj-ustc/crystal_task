#!/usr/bin/env python3

import numpy as np
import os

nsample = 61
natom = 2744
ndim = 3
samples = np.ones((nsample, natom, ndim))

indices_table="indices_table.txt"
with open(indices_table, 'r') as fid:
    lines = fid.readlines()
    
for lino, line in enumerate(lines):
    
    _, dname = line.split()
    fname = dname + '/xyz.cluster'
    
    print(line)
    samples[lino] = np.loadtxt(fname, skiprows=4, usecols=(0, 1, 2))

    # convert to Cartesian coordinate
    with open(fname, 'r') as fid:
        a = float(fid.readline().split()[0])

    samples[lino] *= a

    center = np.mean(samples[lino], axis=0)

    for atom_lino in range(natom):
        samples[lino, atom_lino] = samples[lino, atom_lino] - center
    
#print(samples.shape)
np.save('prototypes.npy', samples)
