#!/usr/bin/env python3

from make_structs import frac2cart, cart2frac, ParseCHTAB, ParseXYZ, saveXYZ
from optparse import OptionParser
import numpy as np

def add_site_noise(xyz, sigma=0.05):
    xyz_noisy = xyz.copy()
    positions = frac2cart(xyz, xyz['sites'])
    noise = np.random.normal(0, sigma, positions.shape)
    positions = positions + noise
    xyz_noisy['sites'] = (cart2frac(xyz, positions) ) % 1
    return xyz_noisy

if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()

    fxyz = args[0]
    chtab = ParseCHTAB()
    xyz = ParseXYZ(fxyz, chtab)

    xyz = add_site_noise(xyz, sigma=0.05)

    saveXYZ(fxyz+'_noisy', xyz)