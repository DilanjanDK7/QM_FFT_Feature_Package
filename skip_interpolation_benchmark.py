import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from QM_FFT_Analysis.utils import calculate_analytical_gradient

def run_benchmark(n_points_list, skip_interpolation):
    """Run benchmark for different numbers of points."""
    times = []
    
    for n_points in n_points_list:
        # Generate sample data
        n_trans = 5  # Multiple time points
        
        # Create coordinates in range [-1, 1]
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.random.uniform(-1, 1, n_points)
        
        # Create complex strengths
        strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)
        
        # Time the calculation
        start_time = time.time()
        
        calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            output_dir=None,
            average=True,
            skip_interpolation=skip_interpolation
        )
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        print(f"Points: {n_points}, Skip Interpolation: {skip_interpolation}, Time: {elapsed_time:.4f} seconds")
    
    return times

def main():
    # Test with different numbers of points
    n_points_list = [100, 500, 1000, 2000, 5000]
    
    # Run with interpolation
    print("Running with interpolation...")
    times_with_interp = run_benchmark(n_points_list, skip_interpolation=False)
    
    # Run without interpolation
    print("\nRunning without interpolation...")
    times_without_interp = run_benchmark(n_points_list, skip_interpolation=True)
    
    # Calculate speedup
    speedup = [t1/t2 for t1, t2 in zip(times_with_interp, times_without_interp)]
    
    # Print summary
    print("\nPerformance Summary:")
    print("Points\tWith Interp(s)\tWithout Interp(s)\tSpeedup")
    for i, n_points in enumerate(n_points_list):
        print(f"{n_points}\t{times_with_interp[i]:.4f}\t{times_without_interp[i]:.4f}\t{speedup[i]:.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_points_list, times_with_interp, 'o-', label='With Interpolation')
    plt.plot(n_points_list, times_without_interp, 'o-', label='Without Interpolation')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(n_points_list, speedup, 'o-', color='green')
    plt.xlabel('Number of Points')
    plt.ylabel('Speedup (x times)')
    plt.title('Performance Speedup')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('skip_interpolation_benchmark.png')
    print("Results saved to skip_interpolation_benchmark.png")

if __name__ == "__main__":
    main() 