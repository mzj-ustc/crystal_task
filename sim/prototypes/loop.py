#!/usr/bin/env python3

import numpy as np

dat = np.load('prototypes.npy')

indices_table="indices_table.txt"
with open(indices_table, 'r') as fid:
    lines = fid.readlines()
    
for lino, line in enumerate(lines):
    
    _, dname = line.split()
    fname = dname + '/crystal.txt'

    np.savetxt(fname, dat[lino])
