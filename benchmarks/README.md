# QM_FFT Feature Package Benchmarks

This directory contains benchmark scripts for evaluating the performance of the QM_FFT_Analysis package. These benchmarks help you understand performance characteristics under different dataset sizes and configuration options.

## Available Benchmarks

### 1. Enhanced-Only Pipeline vs Full Pipeline

**Script**: `benchmark_enhanced_only.py`

**Purpose**: Compares the performance advantage of using `compute_enhanced_metrics()` directly versus running the full processing pipeline when you only need enhanced metrics.

**Key features**:
- Tests with multiple dataset sizes (1K, 5K, and 10K points)
- Measures speedup factors which typically range from 4-9x
- Validates that results are identical between both approaches

**Example usage**:
```bash
python benchmarks/benchmark_enhanced_only.py
```

**Expected output**:
```
SUMMARY OF ENHANCED-ONLY VS FULL PIPELINE BENCHMARK RESULTS
================================================================================
n_points   n_trans    Full Pipeline   Enhanced Only   Speedup   
--------------------------------------------------------------------------------
1000       5          0.145 s         0.025 s         5.91x
5000       10         1.277 s         0.164 s         7.79x
10000      15         3.683 s         0.428 s         8.60x
```

### 2. Analytical Gradient Method vs Standard Method

**Script**: `benchmark_analytical_gradient.py`

**Purpose**: Evaluates the performance advantage of computing gradients analytically in k-space versus the standard interpolation-based approach.

**Key features**:
- Tests with multiple dataset sizes
- Validates gradient accuracy between methods
- Shows performance gains of 1.4-2.3x depending on dataset size

**Example usage**:
```bash
python benchmarks/benchmark_analytical_gradient.py
```

### 3. Extreme Dataset Testing

**Script**: `benchmark_extreme_gradient.py`

**Purpose**: Stress-tests the package with very large datasets to evaluate scaling behavior and performance limits.

**Key features**:
- Tests with extremely large datasets (up to 25K points)
- Evaluates memory usage and execution time
- Helpful for planning resource requirements for large-scale analyses

**Example usage**:
```bash
python benchmarks/benchmark_extreme_gradient.py
```

### 4. Individual Feature Benchmarks

These scripts evaluate the performance of specific features:

**Feature-specific benchmarks**:
- `benchmark_enhanced_features.py`: Tests individual enhanced metrics
- `benchmark_excitation_map.py`: Benchmarks excitation map computation with different HRF models
- `benchmark_local_variance.py` and `benchmark_local_variance_large.py`: Tests local variance calculation performance
- `benchmark_temporal_diff.py` and `benchmark_temporal_diff_large.py`: Evaluates temporal difference calculation

## Customizing Benchmarks

You can customize benchmark parameters by modifying the script files:

1. **Dataset sizes**:
   - Each script has test cases defined as tuples of `(n_points, n_transforms)`
   - Modify the `test_cases` list to test with different dataset sizes

2. **Test repetitions**:
   - For more reliable results, increase the number of repetitions
   - Look for a `n_repeats` parameter in benchmarks that support it

3. **Feature configurations**:
   - Modify feature-specific parameters like `spectral_slope_params` to test different configurations

Example of customizing dataset sizes:

```python
# In benchmark_enhanced_only.py
test_cases = [
    (1000, 5),      # Small dataset
    (5000, 10),     # Medium dataset
    (10000, 15),    # Large dataset
    (20000, 20),    # Add your custom extreme test case
]
```

## Interpreting Results

### Performance Metrics

1. **Execution time**: Measured in seconds for each operation
2. **Speedup factor**: Ratio of baseline time to optimized time
3. **Memory usage**: Reported for some benchmarks via process monitoring
4. **Scaling factor**: How performance scales with increasing dataset size

### Expected Scaling Behavior

Understanding how performance scales with dataset size helps predict resource requirements:

- **Forward/Inverse FFT**: Scales approximately as O(N log N) where N is the number of points
- **Gradient calculations**: 
  - Standard method: O(N log N) + O(M³) where M is the grid dimension
  - Analytical method: O(N log N) scaling, more efficient for larger datasets
- **Enhanced metrics**: Most scale as O(K³) where K is the grid dimension, with significantly less overhead than the full pipeline

## Benchmark Output Directory

Benchmark scripts create a `benchmark_output` directory containing:

- HDF5 files with intermediate results
- Performance measurement data
- Summary statistics

You can safely delete this directory after benchmarking as it contains only temporary files for testing purposes.

## Running All Benchmarks

To run all benchmarks sequentially and generate a comprehensive report:

```bash
for benchmark in benchmarks/benchmark_*.py; do
    echo "Running $benchmark..."
    python $benchmark
    echo "-----------------------------------------"
done
```

## Common Issues

- **Memory errors**: If you encounter memory issues with large datasets, try reducing the grid dimensions or dataset size
- **Variance in results**: Execution times may vary based on system load; run benchmarks multiple times for more reliable results
- **Numerical precision**: Some benchmarks may show small differences in results due to floating-point precision 