import numpy as np
import time
from QM_FFT_Analysis.utils.map_analysis import (
    calculate_local_variance, 
    calculate_local_variance_vectorized,
    calculate_local_variance_fully_vectorized
)

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function by running it multiple times and measuring the average execution time."""
    n_runs = 3  # Number of runs for averaging
    total_time = 0
    
    # Warm-up run
    func(*args, **kwargs)
    
    # Timed runs
    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / n_runs
    return avg_time, result

# Create test data
n_trans = 5
n_points = 1000

# Create complex data for testing
complex_data = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

# Create 3D point coordinates for local variance calculation
points = np.random.randn(n_points, 3)

print("Benchmarking local variance calculation...")

# Original local variance
print("Running original implementation...")
orig_var_time, orig_var_result = benchmark_function(
    calculate_local_variance, complex_data, points, 5
)
print(f"  Original local variance: {orig_var_time:.6f} seconds")

# Vectorized local variance
print("Running vectorized implementation...")
vec_var_time, vec_var_result = benchmark_function(
    calculate_local_variance_vectorized, complex_data, points, 5
)
print(f"  Vectorized local variance: {vec_var_time:.6f} seconds")
print(f"  Speedup: {orig_var_time / vec_var_time:.2f}x")

# Fully vectorized local variance
print("Running fully vectorized implementation...")
full_vec_var_time, full_vec_var_result = benchmark_function(
    calculate_local_variance_fully_vectorized, complex_data, points, 5
)
print(f"  Fully vectorized local variance: {full_vec_var_time:.6f} seconds")
print(f"  Speedup over original: {orig_var_time / full_vec_var_time:.2f}x")
print(f"  Speedup over vectorized: {vec_var_time / full_vec_var_time:.2f}x")

# Verify results are consistent
print(f"\nResults consistency check:")
print(f"  Original vs Vectorized: {np.allclose(orig_var_result, vec_var_result, atol=1e-5)}")
print(f"  Original vs Fully Vectorized: {np.allclose(orig_var_result, full_vec_var_result, atol=1e-5)}")
print(f"  Vectorized vs Fully Vectorized: {np.allclose(vec_var_result, full_vec_var_result, atol=1e-5)}") 