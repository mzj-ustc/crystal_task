#!/bin/bash

# Example script to run test_accuracy.py
# This script tests the accuracy of symmetry detection with noisy structures

# Record start time
start_time=$(date +%s)
start_datetime=$(date)
echo "=========================================="
echo "Job started at: $start_datetime"
echo "=========================================="

# Basic usage - test a single file
# python test_accuracy.py sigma_0.01/A_cF4_225_a/xyz.ini

# Test multiple files with custom parameters

count=0
total=84  # 12 sigma values * 7 symprec values

for sigma in 0.001 0.005 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.1;
do
    for symprec in 0.01 0.05 0.1 0.2 0.3 0.4 0.5;
    do
        count=$((count + 1))
        iter_start=$(date +%s)
        
        echo "[$count/$total] Running sigma=$sigma, symprec=$symprec..."
        
        python ../test_accuracy.py */xyz.ini \
            --sigma $sigma \
            --symprec $symprec \
            --ntrial 100 \
            --nproc 16 1> results_sigma_${sigma}_symprec_${symprec}.txt 2> errors_sigma_${sigma}_symprec_${symprec}.txt
        
        iter_end=$(date +%s)
        iter_elapsed=$((iter_end - iter_start))
        echo "  -> Completed in ${iter_elapsed}s"
    done
done

# Record end time and calculate total elapsed time
end_time=$(date +%s)
end_datetime=$(date)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo "=========================================="
echo "Job finished at: $end_datetime"
echo "Total elapsed time: ${hours}h ${minutes}m ${seconds}s (${elapsed}s)"
echo "=========================================="

