"""
Enhanced-Only Pipeline Benchmark Script

This script benchmarks the performance advantage of using compute_enhanced_metrics() 
directly versus running the full processing pipeline when you only need enhanced metrics.

Purpose:
--------
The benchmark demonstrates the significant performance improvement (typically 4-9x speedup)
that can be achieved by using the enhanced-only pipeline when you don't need the
intermediate inverse maps, gradient maps, or standard analyses.

How it works:
------------
1. For each test case (different combinations of n_points and n_trans):
   - Generates random non-uniform 3D points and complex strengths
   - Runs the full pipeline with process_map() and times it
   - Runs only compute_forward_fft() + compute_enhanced_metrics() and times it
   - Calculates the speedup factor (full time / enhanced time)

2. The script tests three dataset sizes by default:
   - Small: 1,000 points, 5 time points
   - Medium: 5,000 points, 10 time points
   - Large: 10,000 points, 15 time points

Results interpretation:
----------------------
- The benchmark summary table shows the execution time for both approaches and the speedup factor
- Larger speedups (>7x) indicate greater benefit from using the enhanced-only pipeline
- Speedup tends to increase with dataset size, making it especially valuable for larger datasets
- Typical speedups range from 4x for small datasets to 9x for large datasets

Customization:
-------------
- Modify the test_cases list to benchmark with different dataset sizes
- Add additional enhanced metrics to the metrics_to_run list to test their performance
- Change the random data generation to use your own test data

Example output:
--------------
SUMMARY OF ENHANCED-ONLY VS FULL PIPELINE BENCHMARK RESULTS
================================================================================
n_points   n_trans    Full Pipeline   Enhanced Only   Speedup   
--------------------------------------------------------------------------------
1000       5          0.145 s         0.025 s         5.91x
5000       10         1.277 s         0.164 s         7.79x
10000      15         3.683 s         0.428 s         8.60x
"""

import os
import time
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from QM_FFT_Analysis.utils import MapBuilder

def setup_output_dirs():
    """Set up output directories for benchmark results"""
    os.makedirs("benchmark_output/benchmark_full", exist_ok=True)
    os.makedirs("benchmark_output/benchmark_enhanced", exist_ok=True)

def benchmark_enhanced_only(n_points, n_trans):
    """
    Benchmark enhanced-only pipeline vs full pipeline
    
    Args:
        n_points: Number of non-uniform points
        n_trans: Number of transforms (time points)
        
    Returns:
        dict: Results of the benchmark
    """
    print(f"\n{'='*50}")
    print(f"Testing with n_points={n_points}, n_trans={n_trans}")
    print(f"{'='*50}")
    print(f"Benchmarking enhanced-only vs full pipeline with n_points={n_points}, n_trans={n_trans}...")
    
    # Set up output directories
    setup_output_dirs()
    
    # Generate test data
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Create complex strengths with random values
    strengths_real = np.random.randn(n_trans, n_points)
    strengths_imag = np.random.randn(n_trans, n_points)
    strengths = strengths_real + 1j * strengths_imag
    
    # Initialize for full pipeline
    print("\n=== Testing Full Pipeline with Enhanced Metrics ===")
    
    # Initialize map builder for full pipeline
    map_builder_full = MapBuilder(
        subject_id="benchmark_full",
        output_dir=Path("benchmark_output"),
        x=x, y=y, z=z, 
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    # Time the full pipeline
    start_time = time.time()
    
    # Run full pipeline (forward FFT, masks, inverse maps, gradients, analysis)
    map_builder_full.process_map(
        n_centers=1,  # One mask for simplicity
        analyses_to_run=['magnitude', 'phase', 'spectral_slope', 'spectral_entropy', 'anisotropy'],
        use_analytical_gradient=True  # Use enhanced gradient
    )
    
    full_pipeline_time = time.time() - start_time
    print(f"Full pipeline time: {full_pipeline_time:.4f} seconds")
    
    # Initialize for enhanced-only pipeline
    print("\n=== Testing Enhanced-Only Pipeline ===")
    
    # Initialize map builder for enhanced-only pipeline
    map_builder_enhanced = MapBuilder(
        subject_id="benchmark_enhanced",
        output_dir=Path("benchmark_output"),
        x=x, y=y, z=z, 
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    # Time the enhanced-only pipeline
    start_time = time.time()
    
    # Compute forward FFT first
    map_builder_enhanced.compute_forward_fft()
    
    # Then compute enhanced metrics
    enhanced_metrics = map_builder_enhanced.compute_enhanced_metrics(
        metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy']
    )
    enhanced_time = time.time() - start_time
    print(f"Enhanced-only pipeline time: {enhanced_time:.4f} seconds")
    
    # Clean up
    map_builder_full = None
    map_builder_enhanced = None
    
    # Calculate speedup
    speedup = full_pipeline_time / enhanced_time
    print(f"\nSpeedup of enhanced-only vs full pipeline: {speedup:.2f}x\n")
    
    # Return results
    return {
        'n_points': n_points,
        'n_trans': n_trans,
        'full_pipeline_time': full_pipeline_time,
        'enhanced_only_time': enhanced_time,
        'speedup': speedup
    }

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        (1000, 5),    # Small dataset
        (5000, 10),   # Medium dataset
        (10000, 15)   # Large dataset
    ]
    
    # Run benchmarks for each test case
    results = []
    for n_points, n_trans in test_cases:
        result = benchmark_enhanced_only(n_points, n_trans)
        results.append(result)
    
    # Print summary table
    print("\nSUMMARY OF ENHANCED-ONLY VS FULL PIPELINE BENCHMARK RESULTS")
    print("="*80)
    print(f"{'n_points':<10} {'n_trans':<10} {'Full Pipeline':<15} {'Enhanced Only':<15} {'Speedup':<10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['n_points']:<10} {result['n_trans']:<10} "
              f"{result['full_pipeline_time']:.4f} s{' '*8} "
              f"{result['enhanced_only_time']:.4f} s{' '*8} "
              f"{result['speedup']:.2f}x") 