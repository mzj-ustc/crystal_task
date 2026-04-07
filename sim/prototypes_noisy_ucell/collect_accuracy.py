#!/usr/bin/env python3

import os
import re
import glob
import numpy as np

def parse_filename(filename):
    """Extract sigma and symprec values from filename"""
    match = re.search(r'sigma_([\d.]+)_symprec_([\d.]+).txt', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def read_accuracy_data(filename):
    """Read accuracy data from file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            accuracies = []
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    # Extract the 5 accuracy values (columns 2-6)
                    try:
                        values = [float(parts[i]) for i in range(1, 6)]
                        accuracies.append(np.mean(values))  # Average accuracy for this structure
                    except ValueError:
                        continue
            if accuracies:
                return np.mean(accuracies)  # Average accuracy across all structures
            return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def collect_accuracy_data():
    """Collect accuracy data for different sigma and symprec values"""
    results = {}
    error_files = {}

    # Find all result files
    result_files = glob.glob('experiments/results_*.txt')

    for file in result_files:
        sigma, symprec = parse_filename(file)
        if sigma is None or symprec is None:
            continue

        accuracy = read_accuracy_data(file)
        if accuracy is None:
            error_files[file] = "Could not parse accuracy"
            continue

        if sigma not in results:
            results[sigma] = {}
        results[sigma][symprec] = accuracy

    return results, error_files

def compute_statistics(results):
    """Compute mean average and find highest accuracy for each sigma"""
    statistics = {}

    for sigma, symprec_data in results.items():
        symprec_values = list(symprec_data.keys())
        accuracies = list(symprec_data.values())

        mean_accuracy = np.mean(accuracies)
        max_accuracy = np.max(accuracies)
        best_symprec = symprec_values[np.argmax(accuracies)]

        statistics[sigma] = {
            'mean_accuracy': mean_accuracy,
            'true_accuracy': max_accuracy,  # The true accuracy is the max accuracy
            'best_symprec': best_symprec,
            'symprec_values': symprec_values,
            'accuracies': accuracies
        }

    return statistics

def print_results(statistics, error_files):
    """Print the collected results"""
    print("=" * 80)
    print("TRUE ACCURACY STATISTICS")
    print("=" * 80)

    # Print statistics for each sigma
    sigmas = sorted(statistics.keys())
    for sigma in sigmas:
        stats = statistics[sigma]
        print(f"\nSigma: {sigma}")
        print(f"  True Accuracy: {stats['true_accuracy']:.4f}")
        print(f"  Best Symprec: {stats['best_symprec']}")


def save_results(statistics, filename='accuracy_summary.txt'):
    """Save results to a file"""
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ACCURACY STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        sigmas = sorted(statistics.keys())
        for sigma in sigmas:
            stats = statistics[sigma]
            f.write(f"Sigma: {sigma}\n")
            f.write(f"  True Accuracy: {stats['true_accuracy']:.4f}\n")
            f.write(f"  Best Symprec: {stats['best_symprec']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("EXECUTION COMPLETE\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    # Collect accuracy data
    results, error_files = collect_accuracy_data()

    # Compute statistics
    statistics = compute_statistics(results)

    # Print results
    print_results(statistics, error_files)

    # Save results to file
    save_results(statistics)

    print("\n" + "=" * 80)
    print("Results saved to accuracy_summary.txt")
    print("=" * 80)