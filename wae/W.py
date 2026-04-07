import scipy as sp
import numpy as np
import math
from sympy.physics.wigner import wigner_3j
from optparse import OptionParser

def returnWcount(n):
    result=[]
    count = 0
    for a in range(-n, n+1):
        for b in range(-n, n+1):
            for c in range(-n, n+1):
                if a + b + c == 0:
                    result.append([a,b,c])
    return result
def qlm(theta,phi,l,m):
    N=theta.shape[0]
    return np.sum(sp.special.sph_harm(m,l,theta, phi))/N
def ql(theta,phi,l):
    temp=0
    for i in range(-l,l+1):
        temp=temp+np.abs(qlm(theta,phi,l,i))**2
    return math.sqrt(temp**3)
def returnW(n,result1,theta,phi):
    temp=0
    for i in result1:
        temp=temp+(float(wigner_3j(n,n,n,i[0],i[1],i[2]))*qlm(theta,phi,n,i[0])*qlm(theta,phi,n,i[1])*qlm(theta,phi,n,i[2]))
    return (temp/ql(theta,phi,n)).real
def cartesian_to_spherical(xyz):
    # 计算 r
    r = np.linalg.norm(xyz, axis=1)
    # 计算 theta
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    # 计算 phi
    phi = np.arccos(xyz[:, 2] / r)
    
    return r, theta, phi
def cal_distance(sites):
    '''
    Calculate distances of each atom from the origin.
    '''
    return np.sqrt(np.sum(sites**2, axis=-1))
def W_value(in_features1):
    in_features=np.copy(in_features1)
    result1=returnWcount(6)
    q_temp=[]
    for sites in in_features:
        points_shift=in_features-sites
        dist=cal_distance(points_shift)
        dists_shifted = np.sort(dist)
        min_dist = dists_shifted[1]*1.2
        inner_point = points_shift[np.logical_and(dist<=min_dist,dist>0)]
        r, theta, phi=cartesian_to_spherical(inner_point)
        q_temp.append(returnW(6,result1,theta,phi))
    return q_temp 
def W(in_features1):
    in_features=np.copy(in_features1)
    q_temp=[]
    for i in in_features:
        q_temp.append(W_value(i))
    return np.array(q_temp)
in_features2 =np.load('./crystals.npy')

parser = OptionParser()
(options, args) = parser.parse_args()
n = int(args[0])

step = 100
start_id = n*step
end_id = n*step+step

W_final=W(in_features2[start_id:end_id])
np.save('test'+str(n)+'.npy', W_final)


