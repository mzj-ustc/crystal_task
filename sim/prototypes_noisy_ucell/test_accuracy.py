#!/usr/bin/env python3

import argparse
import spglib
from multiprocessing import Pool
from make_structs import ParseCHTAB, ParseXYZ
from add_noise import add_site_noise

def xyz2spgcell(xyz):
    '''
    Convert xyz structure to spglib cell format.
    '''
    lattice = xyz['lattices']
    positions = xyz['sites']
    numbers = xyz['an']

    cell = (lattice, positions, numbers)

    return cell

def process_xyz_file(args):
    '''
    Worker function for parallel processing of a single XYZ file.
    
    Parameters
    ----------
    args : tuple
        Tuple of (fxyz, sigma, symprec, angle_tolerance, ntrial).
    
    Returns
    -------
    result : tuple
        Tuple of (filename, accuracies) where accuracies is a list of
        match ratios for each of the 5 folds.
    '''
    fxyz, sigma, symprec, angle_tolerance, ntrial = args
    
    chtab = ParseCHTAB()
    xyz = ParseXYZ(fxyz, chtab)
    
    accuracies = []
    for fold in range(5):
        accuracy = test_fold(xyz, sigma, symprec, angle_tolerance, ntrial)
        accuracies.append(accuracy)
    
    return (fxyz, accuracies)

def test_fold(xyz, sigma, symprec, angle_tolerance, ntrial):
    '''
    Test symmetry detection accuracy for a single fold.
    
    Adds site noise to the structure and checks if the symmetry group number
    matches the original structure.
    
    Parameters
    ----------
    xyz : dict
        The original structure parsed from XYZ file.
    sigma : float
        Standard deviation of site noise.
    symprec : float
        Symmetry finding precision.
    angle_tolerance : float
        Angle tolerance for symmetry finding.
    ntrial : int
        Number of trials.
    
    Returns
    -------
    accuracy : float
        Ratio of matches (noisy structure has same space group as original).
    '''
    n_match = 0
    res1 = spglib.get_symmetry_dataset(xyz2spgcell(xyz), symprec=symprec, angle_tolerance=angle_tolerance)
    
    for trial in range(ntrial):
        xyz_noisy = add_site_noise(xyz, sigma=sigma)
        res2 = spglib.get_symmetry_dataset(xyz2spgcell(xyz_noisy), symprec=symprec, angle_tolerance=angle_tolerance)
        if res1['number'] == res2['number']:
            n_match += 1
    
    return n_match / ntrial

def test_accuracy(xyz_files, sigma=0.001, symprec=1e-02, angle_tolerance=-1.0, ntrial=100, nproc=None):
    '''
    Test accuracy of symmetry detection with noisy structures.
    
    For each input XYZ file, adds site noise multiple times and checks if the
    symmetry group number matches the original structure.
    
    Parameters
    ----------
    xyz_files : list of str
        List of XYZ file paths to process.
    sigma : float
        Standard deviation of site noise.
    symprec : float
        Symmetry finding precision.
    angle_tolerance : float
        Angle tolerance for symmetry finding.
    ntrial : int
        Number of trials per fold.
    nproc : int, optional
        Number of parallel processes. If None, uses all available CPU cores.
    
    Returns
    -------
    results : list of tuple
        List of (filename, accuracies) tuples, where accuracies is a list of
        match ratios for each of the 5 folds.
    '''
    # Prepare arguments for parallel processing
    pool_args = [(fxyz, sigma, symprec, angle_tolerance, ntrial) for fxyz in xyz_files]
    
    # Process files in parallel
    with Pool(processes=nproc) as pool:
        results = pool.map(process_xyz_file, pool_args)
    
    # Print results in the order of input files
    for fxyz, accuracies in results:
        out = f"{fxyz}\t"
        for acc in accuracies:
            out += f"{acc:.2f}\t"
        print(out)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test accuracy of symmetry detection with noisy structures.")
    parser.add_argument("xyz_files", nargs="+", help="One or more XYZ files to process")
    parser.add_argument("--sigma", type=float, default=0.001,
                        help="Standard deviation of site noise (default: 0.001)")
    parser.add_argument("--symprec", type=float, default=1e-02,
                        help="Symmetry finding precision (default: 1e-02)")
    parser.add_argument("--angle-tolerance", type=float, dest="angle_tolerance", default=-1.0,
                        help="Angle tolerance for symmetry finding (default: -1.0)")
    parser.add_argument("--ntrial", type=int, default=100,
                        help="Number of trials per fold (default: 100)")
    parser.add_argument("--nproc", type=int, default=None,
                        help="Number of parallel processes (default: number of CPU cores)")

    args = parser.parse_args()

    test_accuracy(
        xyz_files=args.xyz_files,
        sigma=args.sigma,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        ntrial=args.ntrial,
        nproc=args.nproc
    )
