#!/usr/bin/env python3

from optparse import OptionParser
from make_structs import ParseCHTAB, ParseXYZ
import spglib

def xyz2spgcell(xyz):
    '''
    Classify the structure by spglib.
    '''

    lattice = xyz['lattices']
    positions = xyz['sites']
    numbers = xyz['an']

    cell = (lattice, positions, numbers)

    return cell

if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    fxyz = args[0]

    chtab = ParseCHTAB()
    xyz = ParseXYZ(fxyz, chtab)

    cell = xyz2spgcell(xyz)

    # Get space group info
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-03, angle_tolerance=-1.0)
    print("Space group number:", dataset['number'])
    print("International symbol:", dataset['international'])
    print("Hall symbol:", dataset['hall'])
