#!/usr/bin/env python3

import os
import shutil
import numpy as np
from optparse import OptionParser

base = os.environ['base']
fpps = base+'/Sources/QuantumESPRESSO/pseudo/default/pbe/' # os.environ['mgtools']+'/qe/SSSP/' # pseudopotential library
fchtab = os.environ['mgtools']+'/INFO/CHTAB' 
PS='pseudo'                     # directory of pseudopotential

def ParseCHTAB():
    fid = open(fchtab)
    lines = fid.readlines()

    chtab = {}
    masses = []
    diameters = []
    shortnames = []
    longnames = []
    pps = []
    
    for lino, line in enumerate(lines):
        if lino == 0:
            continue;

        words = line.split()
        idx = int(words[0])
        masses.append(float(words[1]))
        diameters.append(float(words[2]))
        shortnames.append(words[3])
        longnames.append(words[4])
        try:
            fupf = os.listdir(fpps+words[3])[0]
        except FileNotFoundError:
            fupf = 'NULL'
        pps.append(fupf)

    # for ii in range(idx):
    #     print(shortnames[ii])

    chtab['masses'] = masses
    chtab['diameters'] = diameters
    chtab['shortnames'] = shortnames
    chtab['longnames'] = longnames
    chtab['pps'] = pps
    
    return chtab

def ParseXYZ(fxyz, chtab):
    lattices = np.loadtxt(fxyz, max_rows=3, dtype='float')
    sites = np.loadtxt(fxyz, skiprows=4, usecols=(0,1,2,3,4,5));

    try:
        na, _ = np.shape(sites)
    except ValueError:
        sites = np.expand_dims(sites, axis=0)
        na, _ = np.shape(sites)
        
    xyz = {}
    xyz['lattices'] = lattices
    xyz['sites'] = sites[:, 0:3] # sites
    xyz['an'] = sites[:, 3].astype('int')      # atomic number
    xyz['species'] = len(np.unique(xyz['an']))
    xyz['species_list'] = np.unique(xyz['an'])
    xyz['na'] = na

    prefix=""
    for idx, an in enumerate(xyz['species_list']):
        prefix=prefix+chtab['shortnames'][an]
    xyz['prefix'] = prefix
    
    return xyz

def frac2cart(xyz, frac_vecs):
    '''
    Convert fractional coordinates to Cartesian coordinates.
    '''

    #print(xyz['lattices'].shape, frac_vecs.shape)
    return frac_vecs @ xyz['lattices'] 
    
def cal_distance(cart_vecs):
    '''
    Calculate distance from the Origin.
    '''

    return np.sqrt(np.sum(cart_vecs**2,axis=-1))
    
parser = OptionParser()
# parser.add_option("-f", "--file", dest="filename",
#                   help="write report to FILE", metavar="FILE")
# parser.add_option("-q", "--quiet",
#                   action="store_false", dest="verbose", default=True,
#                   help="don't print status messages to stdout")

parser.add_option("-a", "--box-size", dest="box_size",
                  help="Specify edge length a of a cubic box.", type='float')
parser.add_option("-n", "--number-of-atom", dest="na",
                  help="Specify number of atoms inside a cluster.", type='int')
(options, args) = parser.parse_args()
box_size = options.box_size
na = options.na

fxyz = args[0]
chtab = ParseCHTAB()
xyz = ParseXYZ(fxyz, chtab)

candidates = [[0, 0, 0]]
frac_vecs = np.ones((xyz['na'], 3))
cluster = np.array([])
head = 0
tail = 0
while True:
    try:
        shift_vec = candidates[head]
    except IndexError:
        break

    if head > tail:
        break
    
    head += 1
    frac_vecs[:, 0] = xyz['sites'][:, 0] + shift_vec[0]
    frac_vecs[:, 1] = xyz['sites'][:, 1] + shift_vec[1]
    frac_vecs[:, 2] = xyz['sites'][:, 2] + shift_vec[2]
    cart_vecs = frac2cart(xyz, frac_vecs)
    #dists = cal_distance(cart_vecs)
    edge_max = np.max(np.abs(cart_vecs), axis=-1)
    
    is_inside = edge_max < box_size/2

    if True not in is_inside:
        #del candidates[0]
        continue
    
    cluster = np.append(cluster, cart_vecs[is_inside])

    new_candidate = [shift_vec[0]+1, shift_vec[1], shift_vec[2]]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)
    new_candidate = [shift_vec[0]-1, shift_vec[1], shift_vec[2]]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)
    new_candidate = [shift_vec[0], shift_vec[1]+1, shift_vec[2]]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)
    new_candidate = [shift_vec[0], shift_vec[1]-1, shift_vec[2]]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)
    new_candidate = [shift_vec[0], shift_vec[1], shift_vec[2]+1]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)
    new_candidate = [shift_vec[0], shift_vec[1], shift_vec[2]-1]
    if new_candidate not in candidates:
        tail += 1
        candidates.append(new_candidate)

cluster = np.reshape(cluster, (-1, 3))
w, l = cluster.shape

if w < na:
    print(f"Cannot find enough atoms ({w} < {na}) inside a cubic box with edge length {box_size}")
    exit()
print(f"Finding {w} atoms inside a cubic box with edge length {box_size}")

dists = cal_distance(cluster)
indices = np.argsort(dists)
with open("xyz.cluster", "w") as fid:
    fid.write(f"{box_size} 0 0\n")
    fid.write(f"0 {box_size} 0\n")
    fid.write(f"0 0 {box_size}\n")
    fid.write(str(na)+"\n")
    for ii in range(na):
        site = cluster[indices[ii]]
        fid.write(f"{site[0]/box_size} {site[1]/box_size} {site[2]/box_size} 1 {ii} {ii}\n")

print("Writing to xyz.cluster!")
