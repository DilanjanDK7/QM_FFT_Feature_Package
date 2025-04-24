import numpy as np
import time
from pathlib import Path
import shutil
from QM_FFT_Analysis.utils.map_builder import MapBuilder

def benchmark_analytical_gradient(n_points=1000, n_trans=10, n_runs=5):
    """Benchmark the performance of analytical gradient calculation vs standard method."""
    print(f"Benchmarking analytical gradient with n_points={n_points}, n_trans={n_trans}...")
    
    # Setup test data
    output_dir = Path("benchmark_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate random points in 3D space
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Generate strength data with 3D Gaussian patterns
    strengths_list = []
    for i in range(n_trans):
        # Shift center slightly for each transform to simulate temporal changes
        center_shift = i * 0.2
        str_i = np.exp(-((x - center_shift)**2 + y**2 + z**2) / 0.5) + np.random.randn(n_points) * 0.01
        strengths_list.append(str_i)
    strengths = np.stack(strengths_list).astype(np.complex128)
    
    results = {}
    
    # Test with standard method (interpolation-based gradients)
    print("\nTesting standard gradient method...")
    map_builder_std = MapBuilder(
        subject_id=f"benchmark_gradient_std_{n_points}_{n_trans}",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    print("Computing forward FFT...")
    map_builder_std.compute_forward_fft()
    
    # Generate masks
    print("Generating k-space masks...")
    map_builder_std.generate_kspace_masks(n_centers=2, radius=0.5)
    
    # Compute inverse maps (needed for both methods)
    print("Computing inverse maps...")
    map_builder_std.compute_inverse_maps()
    
    # Warm-up run for standard method
    print("Computing standard gradient (warm-up run)...")
    map_builder_std.compute_gradient_maps(use_analytical_method=False)
    
    # Benchmark standard method
    print(f"Benchmarking standard gradient computation ({n_runs} runs)...")
    std_times = []
    for i in range(n_runs):
        start_time = time.time()
        map_builder_std.compute_gradient_maps(use_analytical_method=False)
        std_time = time.time() - start_time
        std_times.append(std_time)
        print(f"  Run {i+1}: {std_time:.4f} seconds")
    
    avg_std_time = sum(std_times) / len(std_times)
    std_std_time = np.std(std_times)
    print(f"Average standard gradient computation time: {avg_std_time:.4f} ± {std_std_time:.4f} seconds")
    
    # Clean up
    map_builder_std.data_file.close()
    map_builder_std.analysis_file.close()
    map_builder_std.enhanced_file.close()
    
    results['standard_times'] = std_times
    results['avg_standard_time'] = avg_std_time
    results['std_standard_time'] = std_std_time
    
    # Test with analytical method
    print("\nTesting analytical gradient method...")
    map_builder_ana = MapBuilder(
        subject_id=f"benchmark_gradient_ana_{n_points}_{n_trans}",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    print("Computing forward FFT...")
    map_builder_ana.compute_forward_fft()
    
    # Generate masks
    print("Generating k-space masks...")
    map_builder_ana.generate_kspace_masks(n_centers=2, radius=0.5)
    
    # Compute inverse maps (needed for both methods)
    print("Computing inverse maps...")
    map_builder_ana.compute_inverse_maps()
    
    # Warm-up run for analytical method
    print("Computing analytical gradient (warm-up run)...")
    map_builder_ana.compute_gradient_maps(use_analytical_method=True)
    
    # Benchmark analytical method
    print(f"Benchmarking analytical gradient computation ({n_runs} runs)...")
    ana_times = []
    for i in range(n_runs):
        start_time = time.time()
        map_builder_ana.compute_gradient_maps(use_analytical_method=True)
        ana_time = time.time() - start_time
        ana_times.append(ana_time)
        print(f"  Run {i+1}: {ana_time:.4f} seconds")
    
    avg_ana_time = sum(ana_times) / len(ana_times)
    std_ana_time = np.std(ana_times)
    print(f"Average analytical gradient computation time: {avg_ana_time:.4f} ± {std_ana_time:.4f} seconds")
    
    # Calculate speedup
    speedup = avg_std_time / avg_ana_time
    print(f"\nSpeedup of analytical gradient vs standard method: {speedup:.2f}x")
    
    # Clean up
    map_builder_ana.data_file.close()
    map_builder_ana.analysis_file.close()
    map_builder_ana.enhanced_file.close()
    
    results['analytical_times'] = ana_times
    results['avg_analytical_time'] = avg_ana_time
    results['std_analytical_time'] = std_ana_time
    results['speedup'] = speedup
    
    # Final cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    return results

if __name__ == "__main__":
    # Test with different dataset sizes
    test_sizes = [
        (1000, 5),    # Small dataset
        (5000, 10),   # Medium dataset
        (10000, 15)   # Large dataset
    ]
    
    results = []
    for n_points, n_trans in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with n_points={n_points}, n_trans={n_trans}")
        print(f"{'='*50}")
        result = benchmark_analytical_gradient(n_points, n_trans)
        result['n_points'] = n_points
        result['n_trans'] = n_trans
        results.append(result)
        print()
    
    # Print summary table
    print("\nSUMMARY OF ANALYTICAL GRADIENT BENCHMARK RESULTS")
    print("="*80)
    print(f"{'n_points':<10} {'n_trans':<10} {'Standard Time':<20} {'Analytical Time':<20} {'Speedup':<10}")
    print("-"*80)
    for result in results:
        n_points = result['n_points']
        n_trans = result['n_trans']
        std_time = result['avg_standard_time']
        std_err = result['std_standard_time']
        ana_time = result['avg_analytical_time']
        ana_err = result['std_analytical_time']
        speedup = result['speedup']
        print(f"{n_points:<10} {n_trans:<10} {std_time:.4f} ± {std_err:.4f} s {'  '} {ana_time:.4f} ± {ana_err:.4f} s {'  '} {speedup:.2f}x") 