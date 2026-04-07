#!/usr/bin/env python3

import numpy as np
from optparse import OptionParser
import scipy as sp
from scipy.interpolate import interp1d

def find_first_peak(gofr):
    '''
    Find the location of the first peak in gofr.
    Algorithm: Find the first maximum. Calculate the first derivative and find the first solution equals to zero.
    '''

    dg = np.diff(gofr[:, 1])
    l = len(dg)
    for i in np.arange(10, l):
        if dg[i] * dg[i+1] < 0:
            break
    return i
    
def interp_gofr(gofr, x):
    '''
    Rescale gofr according to the location of the first peak.
    Interp to predesigned x.
    '''

    loc_i = find_first_peak(gofr)
    #print(gofr[loc_i, 0])
    gofr[:, 0] /= gofr[loc_i, 0]
    f = interp1d(gofr[:, 0], gofr[:, 1])

    return f(x)
    
    
parser = OptionParser()
parser.add_option("--output", dest="output", default=None,
                  help="Specify an output from evaluation.")

(options, args) = parser.parse_args()
fout = options.output
prefix = fout.split('.')[0]
fout = prefix + '.out'

drelax = '/data5/store1/hy/MD/lammps/LJ_Gaussian_Potential/sim/relax'

datas = np.loadtxt(fout)
nsample, ncol = datas.shape
ngroup = int(np.max(datas[:, 5]) + 1)

nx = 1000
x = np.linspace(0.05, 8, nx)
gofrs = np.zeros((nx, ngroup + 1))
gofrs[:, 0] = x
ncount = np.zeros(ngroup)

for s_id in range(nsample):
    s = '{0:04d}'.format(s_id)
    fname = drelax + '/RUN_' + s + '/gofr.dat-003'

    g_id = int(datas[s_id, 5])
    gofr = np.loadtxt(fname, skiprows=2)
    
    ncount[g_id] += 1
    gofrs[:, g_id+1] += interp_gofr(gofr, x)

for g_id in range(ngroup):
    gofrs[:, g_id+1] /= ncount[g_id]

print(gofrs.shape)
np.savetxt(prefix + '.gofr', gofrs)
          

