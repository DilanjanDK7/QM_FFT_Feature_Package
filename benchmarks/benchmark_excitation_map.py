import numpy as np
import time
from pathlib import Path
import shutil
from QM_FFT_Analysis.utils.map_builder import MapBuilder

def benchmark_excitation_map(n_points=1000, n_trans=10):
    """Benchmark the performance of excitation map calculation."""
    print(f"Benchmarking excitation map with n_points={n_points}, n_trans={n_trans}...")
    
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
    
    # Create MapBuilder instance
    map_builder = MapBuilder(
        subject_id=f"benchmark_excitation_{n_points}_{n_trans}",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    # Compute forward FFT first (required for enhanced metrics)
    print("Computing forward FFT...")
    start_time = time.time()
    map_builder.compute_forward_fft()
    fft_time = time.time() - start_time
    print(f"Forward FFT computation: {fft_time:.4f} seconds")
    
    # Warm-up run for excitation map
    print("Computing excitation map (warm-up run)...")
    map_builder.compute_enhanced_metrics(metrics_to_run=['excitation'])
    
    # Benchmark excitation map computation (5 runs for average)
    print("Benchmarking excitation map computation (5 runs)...")
    excitation_times = []
    for i in range(5):
        start_time = time.time()
        map_builder.compute_enhanced_metrics(metrics_to_run=['excitation'])
        excitation_time = time.time() - start_time
        excitation_times.append(excitation_time)
        print(f"  Run {i+1}: {excitation_time:.4f} seconds")
    
    # Calculate average and standard deviation
    avg_time = sum(excitation_times) / len(excitation_times)
    std_time = np.std(excitation_times)
    print(f"Average excitation map computation time: {avg_time:.4f} ± {std_time:.4f} seconds")
    
    # Clean up
    map_builder.data_file.close()
    map_builder.analysis_file.close()
    map_builder.enhanced_file.close()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    return {
        'n_points': n_points,
        'n_trans': n_trans,
        'fft_time': fft_time,
        'excitation_times': excitation_times,
        'avg_excitation_time': avg_time,
        'std_excitation_time': std_time
    }

if __name__ == "__main__":
    # Test with varying number of time points (n_trans)
    # Since excitation map calculation depends heavily on n_trans
    test_sizes = [
        (5000, 5),    # Few time points
        (5000, 10),   # Medium time points
        (5000, 20),   # Many time points
        (5000, 50),   # Very many time points
    ]
    
    results = []
    for n_points, n_trans in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with n_points={n_points}, n_trans={n_trans}")
        print(f"{'='*50}")
        result = benchmark_excitation_map(n_points, n_trans)
        results.append(result)
        print()
    
    # Print summary table
    print("\nSUMMARY OF EXCITATION MAP BENCHMARK RESULTS")
    print("="*60)
    print(f"{'n_points':<10} {'n_trans':<10} {'FFT Time':<15} {'Excitation Time':<20} {'Ratio':<10}")
    print("-"*60)
    for result in results:
        n_points = result['n_points']
        n_trans = result['n_trans']
        fft_time = result['fft_time']
        avg_time = result['avg_excitation_time']
        std_time = result['std_excitation_time']
        ratio = avg_time / n_trans
        print(f"{n_points:<10} {n_trans:<10} {fft_time:.4f} s {'      '} {avg_time:.4f} ± {std_time:.4f} s {'  '} {ratio:.4f} s/trans") 