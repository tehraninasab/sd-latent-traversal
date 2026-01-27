#!/bin/bash

# Run script for multiple GPUs
# Save this as run_multi_gpu.sh and make it executable with: chmod +x run_multi_gpu.sh

# Disease and style combinations to process
DISEASES=("melanoma" "nevus")
STYLES=("hairs" "gel_bubbles" "ruler" "ink")

# Number of GPUs to use
NUM_GPUS=4

# Process each combination
for DISEASE in "${DISEASES[@]}"; do
  for STYLE in "${STYLES[@]}"; do
    echo "Processing $DISEASE/$STYLE across $NUM_GPUS GPUs"
    
    # Launch a separate process for each GPU
    for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
      echo "Starting GPU $GPU_ID for $DISEASE/$STYLE"
      python lpips_multigpu_isic.py --gpu $GPU_ID --total-gpus $NUM_GPUS --disease $DISEASE --style $STYLE &
    done
    
    # Wait for all GPU processes to finish
    wait
    
    # Combine results from all GPUs
    echo "Combining results for $DISEASE/$STYLE"
    python -c "
import pandas as pd
import glob
import os

disease = '$DISEASE'
style = '$STYLE'
num_gpus = $NUM_GPUS

# Get all partial result files
files = glob.glob(f'../metrics/lpips_{disease}_{style}_gpu*.csv')
if not files:
    print(f'No result files found for {disease}/{style}')
    exit(0)

# Combine all results
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

# Save combined results
output_path = f'../metrics/lpips_{disease}_{style}_interpolation.csv'
combined.to_csv(output_path, index=False)
print(f'Saved combined results for {disease}/{style} with {len(combined)} patients')

# Clean up individual GPU files
for f in files:
    os.remove(f)
"
    
    echo "Completed $DISEASE/$STYLE"
  done
done

echo "All LPIPS computations completed!"