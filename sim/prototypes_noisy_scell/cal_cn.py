#!/usr/bin/env python3

################################################################
# Find number of nearest neighbors and nest nearest neighbors. #
# Use a threshold to distinguish different shells.             #
################################################################

import numpy as np
import jax
import jax.numpy as jnp

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        key = '_'.join(str(n))
        if key in myMap: myMap[key] += 1
        else: myMap[key] = 1

        # Keep track of maximum on the go
        if myMap[key] > maximum[1]: maximum = (n,myMap[key])

    return maximum

@jax.jit
def cal_distance(sites):
    '''
    Calculate distances of each atom from the origin.
    '''

    return jnp.sqrt(jnp.sum(sites**2, axis=-1))

# If pairs' lengthes differ greater than threshold, they belong to different shells
threshold = 0.1

# a file contains atomic positions in Cartesian coordinate
# mass center are moved to the origin
fname = 'prototypes.npy'
structures = np.load(fname)

# only check atoms in the interior sphere to avoid surface problem
r_inner = 20

# cutoff radius of shells
r_cut = 10.0

for struct_id, sites in enumerate(structures):
    #print(struct_id, sites.shape)

    dists = cal_distance(sites)

    cns_avg = []
    for site in sites[dists < r_inner]:
        sites_shifted = sites - site
        dists_shifted = cal_distance(sites_shifted)
        dists_shifted = np.sort(dists_shifted)
        
        shell_start = dists_shifted[1]
        cns = []
        count = 0
        for dist in dists_shifted[1:]:
            if dist - shell_start > threshold: # it is a new shell
                shell_start = dist
                cns.append(count)
                count = 1
            else:
                count += 1

            if dist > r_cut:
                break

        cns_avg.append(cns)

    cns = find_majority(cns_avg)[0]
    print(struct_id, ' '.join([str(i) for i in cns]))

# for struct_id, sites in enumerate(structures):
#     #print(struct_id, sites.shape)

#     dists = cal_distance(sites)
#     atom_id = np.argmin(dists)  # find the atom cloest to the origin
    
#     cns = []
#     sites_shifted = sites - sites[atom_id]
#     dists_shifted = cal_distance(sites_shifted)
#     dists_shifted = np.sort(dists_shifted)
    
#     shell_start = dists_shifted[1]
#     count = 0
#     for dist in dists_shifted[1:]:
#         if dist - shell_start > threshold: # it is a new shell
#             shell_start = dist
#             cns.append(count)
#             count = 1
#         else:
#             count += 1
            
#         if dist > r_cut:
#             break
            
#     print(struct_id, cns)
    
