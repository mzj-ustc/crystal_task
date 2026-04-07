#!/usr/bin/env python3

import numpy as np
from optparse import OptionParser
import scipy as sp
import math

from multiprocessing import Pool

def cartesian_to_spherical(xyz):
    # 计算 r
    r = np.linalg.norm(xyz, axis=1)
    
    # 计算 theta
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    
    # 计算 phi
    phi = np.arccos(xyz[:, 2] / r)
    
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    # 计算直角坐标 x
    x = r * np.sin(phi) * np.cos(theta)
    
    # 计算直角坐标 y
    y = r * np.sin(phi) * np.sin(theta)
    
    # 计算直角坐标 z
    z = r * np.cos(phi)
    return x, y, z

def cal_distance(sites):
    '''
    Calculate distances of each atom from the origin.
    '''

    return np.sqrt(np.sum(sites**2, axis=-1))

def closest_to_origin(points):
    dist=cal_distance(points)
    index1=np.argmin(dist)
    # 找到距离原点最近的点
    
    return points[index1].reshape(-1,3)

def clean_up(in_features, ncore):
    """
    Move the closest atom to the origin (make an atom centered at the origin).
    Only keep a spherical cluster containing a certain number of atoms (natom_sel).
    """
    nsample, natom_tot, dim = in_features.shape

    q2 = sphere(in_features, 2, ncore).reshape(nsample, -1, 1)
    q4 = sphere(in_features, 4, ncore).reshape(nsample, -1, 1)
    q6 = sphere(in_features, 6, ncore).reshape(nsample, -1, 1)
    q8 = sphere(in_features, 8, ncore).reshape(nsample, -1, 1)
    q10 = sphere(in_features, 10, ncore).reshape(nsample, -1, 1)
    
    nsample, natom_tot, dim = in_features.shape
    natom_sel = 1024            # number of selected atoms

    # nsample and natom_sel radii of selected atoms, with x,y,z dimensions
    indices_sel = np.zeros((nsample, natom_sel), dtype='int')
    q2_sel = np.zeros((nsample, natom_sel), dtype='float')
    q4_sel = np.zeros((nsample, natom_sel), dtype='float')
    q6_sel = np.zeros((nsample, natom_sel), dtype='float')
    q8_sel = np.zeros((nsample, natom_sel), dtype='float')
    q10_sel = np.zeros((nsample, natom_sel), dtype='float')
    coords_sel = np.zeros((nsample, natom_sel, 3))
    for i in range(nsample):
        
        # find the closest point to the origin
        coords = in_features[i]
        clost_point = closest_to_origin(coords)

        # move the selected atom to the origin
        coords=coords-clost_point
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # distance to the origin 
        r = np.sqrt(x**2 + y**2 + z**2)

        # distances of neighbor points up to 3
        indices = np.argsort(r)[:natom_sel]

        indices_sel[i, :] = indices

        q2_sel[i, :] = q2[i, indices, 0]
        q4_sel[i, :] = q4[i, indices, 0]
        q6_sel[i, :] = q6[i, indices, 0]
        q8_sel[i, :] = q8[i, indices, 0]
        q10_sel[i, :] = q10[i, indices, 0]
        coords_sel[i, :, :] = in_features[i, indices, :]

    q2_sel = q2_sel[:, :, np.newaxis]
    q4_sel = q4_sel[:, :, np.newaxis]
    q6_sel = q6_sel[:, :, np.newaxis]
    q8_sel = q8_sel[:, :, np.newaxis]
    q10_sel = q10_sel[:, :, np.newaxis]
    #print(in_features.shape)
    #np.save('qs_core.npy', coords_sel)
    features = np.concatenate([q2_sel, q4_sel, q6_sel, q8_sel, q10_sel], axis=-1)
    #print(features.shape)
    return features

def qlm(theta,phi,l,m):
    N=theta.shape[0]
    return np.sum(sp.special.sph_harm(m,l,theta, phi))/N

def ql(theta,phi,l):
    temp=0
    for i in range(-l,l+1):
        temp=temp+np.abs(qlm(theta,phi,l,i))**2
    return math.sqrt(math.pi*4/(2*l+1)*temp)

#def sphere_value(in_features, q_num):
def sphere_value(args):
    in_features, q_num = args 
    natom_tot, dim = in_features.shape
    q_temp = np.zeros(natom_tot)
    for sid, sites in enumerate(in_features):
        points_shift = in_features-sites
        dist = cal_distance(points_shift)
        dists_shifted = np.argpartition(dist, 2) # find the shortest bond
        min_dist = dist[dists_shifted[1]]*1.2
        inner_point = points_shift[np.logical_and(dist<=min_dist,dist>0)]
        r, theta, phi = cartesian_to_spherical(inner_point)
        q_temp[sid] = ql(theta,phi,q_num)
    return q_temp

def sphere(in_features, q_num, ncore):
    '''
    Use multiprocessing to accelerate computing.
    INPUT:
    in_features: input structures
    q_num: quantum number
    ncore: number of cores
    '''
    
    nsample, natom_tot, dim = in_features.shape
    qs = np.zeros((nsample, natom_tot))

    pool = Pool(ncore)
    qs_parts = pool.map(sphere_value, [(feature, q_num) for feature in in_features])
    qs = np.concatenate(qs_parts, axis=0)
    return qs


if __name__ == '__main__':
    '''
    Load datas from fin and fout.
    
    Parameters:
    fin: 
        A file contains inputs to a simulator.
    fout: 
        A file contains ouputs from a simulator when given inputs in fin.

    Return:
    a tuple contains data_loader, in_features, out_features
    '''

    parser = OptionParser()
    parser.add_option("--cartesian", dest="fcart", default=None,
                      help="A file contains structure sites in artesian coordinates.")
    parser.add_option("--output", dest="fout", default=None,
                      help="Output q parameters to a file.")
    
    (options, args) = parser.parse_args()
    fcart = options.fcart
    fout = options.fout
    
    ncore = 16
    
    in_features = np.load(fcart)
    print(in_features.shape)
    in_features = clean_up(in_features, ncore)
    print(in_features.shape)
    np.save(fout, in_features)

