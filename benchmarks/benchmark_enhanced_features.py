import numpy as np
import time
from pathlib import Path
import shutil
from QM_FFT_Analysis.utils.map_builder import MapBuilder

def benchmark_enhanced_features(n_points=1000, n_trans=10):
    """Benchmark the performance of enhanced features calculation."""
    print(f"Benchmarking enhanced features with n_points={n_points}, n_trans={n_trans}...")
    
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
        # Shift center slightly for each transform
        center_shift = i * 0.2
        str_i = np.exp(-((x - center_shift)**2 + y**2 + z**2) / 0.5) + np.random.randn(n_points) * 0.01
        strengths_list.append(str_i)
    strengths = np.stack(strengths_list).astype(np.complex128)
    
    # Create MapBuilder instance
    map_builder = MapBuilder(
        subject_id="benchmark_enhanced",
        output_dir=output_dir,
        x=x, y=y, z=z,
        strengths=strengths,
        enable_enhanced_features=True
    )
    
    # Compute forward FFT first (required for enhanced metrics)
    start_time = time.time()
    map_builder.compute_forward_fft()
    fft_time = time.time() - start_time
    print(f"Forward FFT computation: {fft_time:.4f} seconds")
    
    # Benchmark each enhanced metric individually
    metrics = [
        'spectral_slope',
        'spectral_entropy',
        'anisotropy',
        'higher_moments'
    ]
    
    metrics_times = {}
    for metric in metrics:
        # Reset timer
        start_time = time.time()
        # Compute specific enhanced metric
        map_builder.compute_enhanced_metrics(metrics_to_run=[metric])
        metric_time = time.time() - start_time
        metrics_times[metric] = metric_time
        print(f"{metric} computation: {metric_time:.4f} seconds")
    
    # Benchmark all metrics together
    start_time = time.time()
    enhanced_results = map_builder.compute_enhanced_metrics()
    all_metrics_time = time.time() - start_time
    print(f"All enhanced metrics computation: {all_metrics_time:.4f} seconds")
    
    # Clean up
    map_builder.data_file.close()
    map_builder.analysis_file.close()
    map_builder.enhanced_file.close()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    return {
        'fft_time': fft_time,
        'metrics_times': metrics_times,
        'all_metrics_time': all_metrics_time
    }

if __name__ == "__main__":
    # Test with different dataset sizes
    test_sizes = [
        (500, 5),    # Small dataset
        (2000, 10),  # Medium dataset
        (5000, 20)   # Large dataset
    ]
    
    for n_points, n_trans in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with n_points={n_points}, n_trans={n_trans}")
        print(f"{'='*50}")
        benchmark_enhanced_features(n_points, n_trans)
        print() 