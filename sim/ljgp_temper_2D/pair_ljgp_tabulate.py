#!/usr/bin/env python

##############################################################################################################################
# A script to generate the Lennard-Jones Gauss potential (LJGP) according to Eq.(2) in                                       #
# Ref. "Moving beyond the constraints of chemistry via crystal structure discovery with isotropic multiwell pair potentials" #
#
# This code uses scripts from simulation package LAMMPS to generate tabulated potential files
##############################################################################################################################

from lammps_src.tabulate import PairTabulate
import numpy as np

################################################################################
import math
epsilon = np.loadtxt('EPSILON')
r0 = np.loadtxt('R0')
sigma_g2 = 0.02

def ljgp_energy(r):
    e = math.pow(1.0/r, 12.0) - 2 *math.pow(1.0/r, 6.0) -epsilon *math.exp(-(r-r0)**2 /2/sigma_g2)
    return e

def ljgp_force(r):
    f = 12.0 *math.pow(1.0/r, 13.0)  -12*math.pow(1.0/r, 7.0) -(r-r0)/sigma_g2*epsilon *math.exp(-(r-r0)**2 /2/sigma_g2)
    return f

################################################################################

if __name__ == "__main__":
    ptable = PairTabulate(ljgp_energy, ljgp_force)
    idx = 0
    fname = 'LJGP'
    print(f'file: {fname}, epsilon: {epsilon}, r0: {r0}')
    ptable.run(fname)
