#!/usr/bin/env python3

import numpy as np

a = 15
def rotatexyz(sites):
    center = np.mean(sites, axis=0)
    sites -= center
    
    dist = np.sqrt(np.sum(sites**2, axis=-1))
    #print(dist)
    idx = np.argmin(dist)
    sites -= sites[idx]
    #print(sites)
    idxs = np.argsort(dist)

    idx = idxs[9]
    alpha, gamma = cal_angle(sites[idx])
    sites = rotate_z(sites, alpha)
    sites = rotate_x(sites, gamma)

    idx = idxs[13]
    alpha, gamma = cal_angle(sites[idx])
    sites = rotate_z(sites, alpha)

    return sites
    
def cal_angle(site):
    '''
    Calculate angle
    '''
    r = np.sqrt(np.sum(site**2))
    alpha = np.arctan(site[0]/site[1])
    gamma = np.arccos(site[2]/r)

    return alpha, gamma
    
def rotate_x(sites, gamma):
    '''
    Rotate along x axis by gamma
    '''
    
    rot_x = np.array([[1,0,0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])

    return (np.matmul(sites, rot_x.T))

def rotate_y(sites, beta):
    '''
    Rotate along y axis by beta
    '''

    rot_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    return (np.matmul(sites, rot_y.T))

def rotate_z(sites, alpha):
    '''
    Rotate along z axis by alpha
    '''

    rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

    return (np.matmul(sites, rot_z.T))

def update_sites(sites, rot_angles):
    '''
    Rotate sites by rot_angles.
    '''
    
    gamma = rot_angles[0]
    beta = rot_angles[1]
    alpha = rot_angles[2]

    sites = rotate_x(sites, gamma)
    sites = rotate_y(sites, beta)
    sites = rotate_z(sites, alpha)

    return sites
    
def printxyz(sites):
    print(f"{a} 0 0")
    print(f"0 {a} 0")
    print(f"0 0 {a}")
    print(1024)

    for site in sites:
        print(f"{site[0]} {site[1]} {site[2]} 1 1 1")
        
    
dat = np.load('qs_core.npy')
dat = dat / a

site = dat[0]
site = rotatexyz(site)

printxyz(site)
