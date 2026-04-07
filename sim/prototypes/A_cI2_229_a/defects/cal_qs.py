#!/usr/bin/env python3

################################################################
# Calculate Steinhardt Parameters for all atoms inside an interior sphere.
# See "Collective Variables for Crystallization Simulations from Early Developments to Recent Advances"
################################################################

import numpy as np
import scipy as sp

def cal_distance(sites):
    '''
    Calculate distances of each atom from the origin.
    '''

    return np.sqrt(np.sum(sites**2, axis=-1))

def to_polar(site):
    '''
    Convert a atomic site in cartesian coordinate into polar coordinate system.
    '''
    
    r = np.sqrt(np.sum(site**2))
    theta = np.arccos(site[2]/r)
    
    rxy = np.sqrt(site[0]**2+site[1]**2)

    if rxy == 0:
        phi = 0
    else: 
        phi = np.arccos(site[0]/rxy) + np.sign(site[1]) * np.pi

    return r, theta, phi

def cal_sph_harm(m, n, site):
    '''
    Calculate sphereical harmonics
    '''

    r, theta, phi = to_polar(site)

    #return sp.special.sph_harm(m, n, theta, phi)
    return sp.special.sph_harm(m, n, phi, theta)

def cal_q(m, n, sites_f, site0):
    '''
    Calculate order parameters q given spherical harmonics order (m, n). Refer to Eq.(1)
    sites store all atomic sites
    site0 will be the center atom
    '''

    # cutoff radius of shells

    # shift the central atom to the origin
    sites = sites_f - site0
    dists = cal_distance(sites)

    #print(dists)
    # set cutoff radius to be 1.2 * shortest_bond
    r_cut = np.sort(dists)[1] * 1.2
    #r_cut = 1.2
    #print(r_cut)

    # find all atoms within r_cut
    is_rc = (dists < r_cut) & (dists > 0)
    rc_sites = sites[is_rc]
    natom, _ = rc_sites.shape

    # define a spherical harmonics with m, n
    q = 0
    for site in rc_sites:
        q += cal_sph_harm(m, n, site)
    
    q /= natom
    return q

def cal_steinhardt_parameters_one_site(l, sites, site0):
    '''
    Calculate Steinhardt parameters for a given site. Refer to Eq.(2)
    l is quantum number for spherical harmonics.
    sites contains all atom sites
    site0 is the central atom
    '''

    ql = 0
    for m in range(2*l+1):
        ql += np.abs(cal_q(m-l, l, sites, site0))**2

    # Steinhardt parameters Eq.(2)
    sp = np.sqrt(4*np.pi/(2*l+1) * ql)
    return sp

def cal_steinhardt_parameters(l, sites):

    r_inner = 10.0

    dists = cal_distance(sites)
    # find all atoms within r_cut
    is_inner = dists < r_inner
    rc_sites = sites[is_inner]

    qls = []
    for site in rc_sites:
        qls.append(cal_steinhardt_parameters_one_site(l, sites, site))
    
    return qls

def cal_steinhardt_parameters_center_atom(l, sites):

    dists = cal_distance(sites)
    # find all atoms within r_cut
    atom_id = np.argmin(dists)

    return cal_steinhardt_parameters_one_site(l, sites, sites[atom_id])

# l = 10
# qls = cal_steinhardt_parameters(l, structure)
# print(qls)

if __name__ == '__main__':
    # a file contains atomic positions in Cartesian coordinate
    # mass center are moved to the origin
    fname = 'crystal-100.txt'
    structure = np.loadtxt(fname)

    for l in [2, 4, 6, 8, 10]:
        print(l, cal_steinhardt_parameters_center_atom(l, structure))

