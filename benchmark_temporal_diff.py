import numpy as np
import time
from QM_FFT_Analysis.utils.map_analysis import (
    calculate_temporal_difference,
    calculate_temporal_difference_vectorized
)

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function by running it multiple times and measuring the average execution time."""
    n_runs = 5  # Number of runs for averaging
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
n_trans = 100  # More time points to better show the difference
n_points = 10000

# Create complex data for testing
complex_data = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

print("Benchmarking temporal difference calculation...")

# Original temporal difference
print("Running original implementation...")
orig_td_time, orig_td_result = benchmark_function(
    calculate_temporal_difference, complex_data
)
print(f"  Original temporal difference: {orig_td_time:.6f} seconds")

# Vectorized temporal difference
print("Running vectorized implementation...")
vec_td_time, vec_td_result = benchmark_function(
    calculate_temporal_difference_vectorized, complex_data
)
print(f"  Vectorized temporal difference: {vec_td_time:.6f} seconds")
print(f"  Speedup: {orig_td_time / vec_td_time:.2f}x")

# Verify results are consistent
print(f"\nResults consistency check:")
print(f"  Original vs Vectorized: {np.allclose(orig_td_result, vec_td_result, atol=1e-5)}") 