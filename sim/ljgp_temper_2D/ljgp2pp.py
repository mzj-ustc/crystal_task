#!/usr/bin/env python

# Create pp type pair potential

import numpy as np

epsilon = np.loadtxt('EPSILON')
r0 = np.loadtxt('R0')
sigma_g2 = 0.02

def ljgp_energy(r):
    e = 1.0/r**12.0 - 2.0/r**6.0 -epsilon *np.exp(-(r-r0)**2 /2/sigma_g2)
    return e

def ljgp_force(r):
    f = 12.0 * 1.0/r**13.0  -12.0/r**7.0 -(r-r0)/sigma_g2*epsilon *np.exp(-(r-r0)**2 /2/sigma_g2)
    return f

################################################################################

if __name__ == "__main__":

    rmin = 0.01
    rmax = 3.01
    n = 301
    
    rs = np.linspace(rmin, rmax, n, endpoint=True)
    energies = ljgp_energy(rs)
    
    print(f"{n} 1 1 {rmin} {rmax} # created by ljpg2pp.py")
    print(f"1 1")
    for i in range(n):
        print(f"{rs[i]:.2f} {energies[i]}")
