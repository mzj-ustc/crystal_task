#!/usr/bin/env python3

import argparse
import numpy as np

def cal_np_accuracy(score_file):
    """
    Calculate NP accuracy using NumPy for improved performance.
    Reads score file with 61 columns (prototype match distances) using loadtxt
    and calculates accuracy by finding minimum values and comparing with
    target indices (modulo 61).
    """
    # Step 1: Read the score file using numpy.loadtxt
    data = np.loadtxt(score_file, delimiter=None)


    # Step 2: Read the indices table (not used in calculation but kept for consistency)
    indices_file = 'indices_table.txt'
    try:
        with open(indices_file, 'r') as f:
            indices_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Indices file '{indices_file}' not found.")
        return
    spgace_lookup = np.array([int(line.split()[1].split('_')[2]) for line in indices_lines])

    # Define equivalent space groups that belong to the same group
    equivalent_groups = {13, 14, 32, 34, 56, 59}
    # Define new group that belongs to a separate group
    new_group = {15, 20}

    # Create normalized space group mapping
    def normalize_spg(spg):
        if spg in new_group:
            return 15  # Normalize new group to 15
        elif spg in equivalent_groups:
            return 13  # Normalize equivalent groups to 13
        return spg
    
    normalize_vectorized = np.vectorize(normalize_spg)
    spgace_lookup_norm = normalize_vectorized(spgace_lookup)

    # Step 3: Calculate predictions and targets using vectorized operations
    if data.size == 0:
        print("No valid data found in score file.")
        return

    predictions = np.argmin(data, axis=1)  # +1 because indices start from 1
    total_count = len(predictions)
    targets = (np.arange(total_count) % 61)

    # Step 5: Calculate top-1 accuracy (with equivalent group matching)
    spg_predictions_norm = normalize_vectorized(predictions)
    spg_targets_norm = normalize_vectorized(targets)
    correct_count_grouped = np.sum(spg_predictions_norm == spg_targets_norm)
    accuracy_grouped = correct_count_grouped / total_count

    # Step 6: Calculate top-3 accuracy (exact match)
    # Get indices of top 3 smallest values for each row
    top3_predictions = np.argsort(data, axis=1)[:, :3]
    # Map to space groups
    spg_top3 = normalize_vectorized(top3_predictions)
    # Check if target is in top 3 predictions for each sample

    # Step 7: Calculate top-3 accuracy (with equivalent group matching)
    spg_top3_norm = normalize_vectorized(spg_top3)
    top3_correct_grouped = np.any(spg_top3_norm == spg_targets_norm[:, np.newaxis], axis=1)
    top3_correct_count_grouped = np.sum(top3_correct_grouped)
    top3_accuracy_grouped = top3_correct_count_grouped / total_count

    # Step 8: Print results
    print(f"Top-1 NP Accuracy (grouped): {accuracy_grouped:.4f} ({correct_count_grouped}/{total_count})")
    print(f"Top-3 NP Accuracy (grouped): {top3_accuracy_grouped:.4f} ({top3_correct_count_grouped}/{total_count})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate NP accuracy from score file')
    parser.add_argument('score_file', help='Path to the score file (.score)')
    
    args = parser.parse_args()
    cal_np_accuracy(args.score_file)
