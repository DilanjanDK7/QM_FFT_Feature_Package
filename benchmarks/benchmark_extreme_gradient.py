import numpy as np
import time
from pathlib import Path
import shutil
from QM_FFT_Analysis.utils.map_builder import MapBuilder
import gc

def benchmark_extreme_gradient():
    """Benchmark the performance of analytical gradient with an extremely large dataset."""
    n_points = 22000
    n_trans = 15
    n_runs = 3  # Reduced number of runs due to extreme dataset size
    
    print(f"Benchmarking analytical gradient with EXTREME dataset:")
    print(f"n_points={n_points}, n_trans={n_trans}")
    
    # Setup test data
    output_dir = Path("benchmark_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate random points in 3D space
    print("Generating random test data...")
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Generate strength data with 3D Gaussian patterns
    print("Generating strength patterns...")
    strengths_list = []
    for i in range(n_trans):
        # Shift center slightly for each transform to simulate temporal changes
        center_shift = i * 0.2
        str_i = np.exp(-((x - center_shift)**2 + y**2 + z**2) / 0.5) + np.random.randn(n_points) * 0.01
        strengths_list.append(str_i)
    strengths = np.stack(strengths_list).astype(np.complex128)
    
    results = {}
    
    # First, test with standard method (interpolation-based gradients)
    print("\n" + "="*50)
    print("TESTING STANDARD GRADIENT METHOD...")
    print("="*50)
    
    map_builder_std = MapBuilder(
        subject_id=f"benchmark_gradient_std_extreme",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    print("Computing forward FFT...")
    start_time = time.time()
    map_builder_std.compute_forward_fft()
    fft_time = time.time() - start_time
    print(f"Forward FFT computation time: {fft_time:.4f} seconds")
    
    # Generate masks
    print("Generating k-space masks...")
    map_builder_std.generate_kspace_masks(n_centers=1, radius=0.5)  # Just one mask to save time
    
    # Compute inverse maps (needed for both methods)
    print("Computing inverse maps...")
    start_time = time.time()
    map_builder_std.compute_inverse_maps()
    inverse_time = time.time() - start_time
    print(f"Inverse maps computation time: {inverse_time:.4f} seconds")
    
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
    del map_builder_std
    gc.collect()  # Force garbage collection to free memory
    
    results['standard_times'] = std_times
    results['avg_standard_time'] = avg_std_time
    results['std_standard_time'] = std_std_time
    results['std_fft_time'] = fft_time
    results['std_inverse_time'] = inverse_time
    
    # Now test with analytical method
    print("\n" + "="*50)
    print("TESTING ANALYTICAL GRADIENT METHOD...")
    print("="*50)
    
    map_builder_ana = MapBuilder(
        subject_id=f"benchmark_gradient_ana_extreme",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    print("Computing forward FFT...")
    start_time = time.time()
    map_builder_ana.compute_forward_fft()
    fft_time = time.time() - start_time
    print(f"Forward FFT computation time: {fft_time:.4f} seconds")
    
    # Generate masks
    print("Generating k-space masks...")
    map_builder_ana.generate_kspace_masks(n_centers=1, radius=0.5)  # Just one mask to save time
    
    # Compute inverse maps (needed for both methods)
    print("Computing inverse maps...")
    start_time = time.time()
    map_builder_ana.compute_inverse_maps()
    inverse_time = time.time() - start_time
    print(f"Inverse maps computation time: {inverse_time:.4f} seconds")
    
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
    results['ana_fft_time'] = fft_time
    results['ana_inverse_time'] = inverse_time
    results['speedup'] = speedup
    
    # Final cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    return results

if __name__ == "__main__":
    print("\n" + "="*50)
    print("EXTREME GRADIENT BENCHMARK")
    print("="*50)
    
    results = benchmark_extreme_gradient()
    
    # Print summary
    print("\nSUMMARY OF EXTREME GRADIENT BENCHMARK RESULTS")
    print("="*80)
    print("Dataset: 22,000 points, 15 time points\n")
    
    print(f"Standard method:")
    print(f"  FFT time: {results['std_fft_time']:.4f} seconds")
    print(f"  Inverse maps time: {results['std_inverse_time']:.4f} seconds")
    print(f"  Gradient time: {results['avg_standard_time']:.4f} ± {results['std_standard_time']:.4f} seconds")
    
    print(f"\nAnalytical method:")
    print(f"  FFT time: {results['ana_fft_time']:.4f} seconds")
    print(f"  Inverse maps time: {results['ana_inverse_time']:.4f} seconds")
    print(f"  Gradient time: {results['avg_analytical_time']:.4f} ± {results['std_analytical_time']:.4f} seconds")
    
    print(f"\nSpeedup of analytical gradient vs standard method: {results['speedup']:.2f}x") 