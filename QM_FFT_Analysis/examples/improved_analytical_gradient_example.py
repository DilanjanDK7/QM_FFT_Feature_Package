#!/usr/bin/env python
"""
Improved Analytical Gradient Example

This example demonstrates the enhanced analytical gradient function with:
1. FINUFFT parameter tuning (upsampfac, spreadwidth)
2. Adaptive grid estimation
3. Performance optimization options
4. Comprehensive parameter exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import h5py

from QM_FFT_Analysis.utils import calculate_analytical_gradient

def generate_complex_sample_data(n_points=2000, n_trans=8):
    """Generate more complex sample 3D data for demonstration."""
    # Generate non-uniform coordinates with clustering
    # Create some clustered regions to test adaptive grid estimation
    cluster_centers = [
        (-2, -2, -2), (2, 2, 2), (0, 0, 0), (-1, 1, -1)
    ]
    
    x_coords = []
    y_coords = []
    z_coords = []
    
    points_per_cluster = n_points // len(cluster_centers)
    
    for center in cluster_centers:
        # Add clustered points around each center
        cluster_x = np.random.normal(center[0], 0.8, points_per_cluster)
        cluster_y = np.random.normal(center[1], 0.8, points_per_cluster)
        cluster_z = np.random.normal(center[2], 0.8, points_per_cluster)
        
        x_coords.extend(cluster_x)
        y_coords.extend(cluster_y)
        z_coords.extend(cluster_z)
    
    # Add remaining points randomly
    remaining = n_points - len(x_coords)
    if remaining > 0:
        x_coords.extend(np.random.uniform(-3, 3, remaining))
        y_coords.extend(np.random.uniform(-3, 3, remaining))
        z_coords.extend(np.random.uniform(-3, 3, remaining))
    
    x = np.array(x_coords)
    y = np.array(y_coords)
    z = np.array(z_coords)
    
    # Calculate radius and create complex patterns
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Generate multiple time points with different frequency content
    strengths = np.zeros((n_trans, n_points), dtype=np.complex128)
    
    for t in range(n_trans):
        # Create multi-scale patterns
        freq1 = 0.5 + 0.1 * t  # Low frequency component
        freq2 = 2.0 + 0.3 * t  # High frequency component
        
        # Real part: Multi-frequency oscillations with Gaussian envelope
        real_part = (np.exp(-r**2 / 4) * np.cos(freq1 * r) + 
                    0.3 * np.exp(-r**2 / 2) * np.cos(freq2 * r))
        
        # Imaginary part: Phase modulation
        imag_part = 0.5 * np.sin(freq1 * r + t * np.pi / 4)
        
        # Add some noise
        noise_real = np.random.normal(0, 0.05, n_points)
        noise_imag = np.random.normal(0, 0.05, n_points)
        
        strengths[t, :] = (real_part + noise_real) + 1j * (imag_part + noise_imag)
    
    return x, y, z, strengths

def benchmark_finufft_parameters():
    """Benchmark different FINUFFT parameter combinations."""
    print("=" * 60)
    print("FINUFFT PARAMETER BENCHMARKING")
    print("=" * 60)
    
    # Generate test data
    x, y, z, strengths = generate_complex_sample_data(n_points=1500, n_trans=5)
    output_dir = Path("./benchmark_output")
    output_dir.mkdir(exist_ok=True)
    
    # Parameter combinations to test
    test_configs = [
        {
            'name': 'Standard (upsampfac=2.0)',
            'upsampfac': 2.0,
            'eps': 1e-6,
            'spreadwidth': None
        },
        {
            'name': 'Fast (upsampfac=1.25)',
            'upsampfac': 1.25,
            'eps': 1e-5,
            'spreadwidth': None
        },
        {
            'name': 'High Precision (upsampfac=2.5)',
            'upsampfac': 2.5,
            'eps': 1e-8,
            'spreadwidth': None
        },
        {
            'name': 'Balanced (upsampfac=1.5)',
            'upsampfac': 1.5,
            'eps': 1e-6,
            'spreadwidth': None
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"Parameters: upsampfac={config['upsampfac']}, eps={config['eps']}")
        
        start_time = time.time()
        
        try:
            gradient_results = calculate_analytical_gradient(
                x=x, y=y, z=z, strengths=strengths,
                subject_id=f"benchmark_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}",
                output_dir=output_dir,
                upsampfac=config['upsampfac'],
                eps=config['eps'],
                spreadwidth=config['spreadwidth'],
                skip_interpolation=True,  # For speed
                average=True
            )
            
            execution_time = time.time() - start_time
            
            # Calculate some quality metrics
            gradient_mean = np.mean(gradient_results['gradient_map_nu'])
            gradient_std = np.std(gradient_results['gradient_map_nu'])
            
            results[config['name']] = {
                'execution_time': execution_time,
                'gradient_mean': gradient_mean,
                'gradient_std': gradient_std,
                'k_space_info': gradient_results['k_space_info'],
                'success': True
            }
            
            print(f"  ✓ Completed in {execution_time:.3f}s")
            print(f"  ✓ Gradient mean: {gradient_mean:.6f}")
            print(f"  ✓ K-space sampling sufficient: {gradient_results['k_space_info']['sampling_sufficient']}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[config['name']] = {
                'execution_time': float('inf'),
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        print(f"Fastest configuration: {fastest[0]} ({fastest[1]['execution_time']:.3f}s)")
        
        # Show relative performance
        for name, result in successful_results.items():
            relative_speed = result['execution_time'] / fastest[1]['execution_time']
            print(f"{name:25s}: {result['execution_time']:.3f}s ({relative_speed:.2f}x)")
    
    return results

def demonstrate_adaptive_grid():
    """Demonstrate adaptive grid estimation feature."""
    print("\n" + "=" * 60)
    print("ADAPTIVE GRID ESTIMATION DEMO")
    print("=" * 60)
    
    # Generate data with varying complexity
    x, y, z, strengths = generate_complex_sample_data(n_points=1000, n_trans=3)
    output_dir = Path("./adaptive_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test different target accuracy levels
    accuracy_levels = [0.8, 0.9, 0.95, 0.99]
    
    for accuracy in accuracy_levels:
        print(f"\nTesting adaptive grid with target accuracy: {accuracy}")
        
        start_time = time.time()
        
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            subject_id=f"adaptive_accuracy_{accuracy}",
            output_dir=output_dir,
            adaptive_grid=True,
            target_accuracy=accuracy,
            skip_interpolation=True,
            average=True
        )
        
        execution_time = time.time() - start_time
        grid_size = results['k_space_info']['total_grid_points']
        
        print(f"  Grid size: {grid_size:,} points")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Grid dimensions: {results['coordinates']['nx']} x {results['coordinates']['ny']} x {results['coordinates']['nz']}")

def demonstrate_performance_modes():
    """Demonstrate different performance optimization modes."""
    print("\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION MODES")
    print("=" * 60)
    
    # Generate larger dataset for performance testing
    x, y, z, strengths = generate_complex_sample_data(n_points=3000, n_trans=6)
    output_dir = Path("./performance_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    modes = [
        {
            'name': 'Maximum Speed',
            'params': {
                'upsampfac': 1.25,
                'eps': 1e-4,
                'skip_interpolation': True,
                'adaptive_grid': False,
                'upsampling_factor': 2.0  # Reduced for speed
            }
        },
        {
            'name': 'Balanced',
            'params': {
                'upsampfac': 1.5,
                'eps': 1e-6,
                'skip_interpolation': True,
                'adaptive_grid': True,
                'target_accuracy': 0.9
            }
        },
        {
            'name': 'Maximum Accuracy',
            'params': {
                'upsampfac': 2.5,
                'eps': 1e-8,
                'skip_interpolation': False,
                'adaptive_grid': True,
                'target_accuracy': 0.99
            }
        }
    ]
    
    for mode in modes:
        print(f"\nTesting: {mode['name']} mode")
        
        start_time = time.time()
        
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            subject_id=f"performance_{mode['name'].replace(' ', '_').lower()}",
            output_dir=output_dir,
            **mode['params']
        )
        
        execution_time = time.time() - start_time
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Grid size: {results['k_space_info']['total_grid_points']:,} points")
        print(f"  K-space sampling sufficient: {results['k_space_info']['sampling_sufficient']}")
        if 'gradient_map_grid' in results:
            print(f"  Includes interpolated grid: Yes")
        else:
            print(f"  Includes interpolated grid: No (skip_interpolation=True)")

def main():
    """Main demonstration function."""
    print("IMPROVED ANALYTICAL GRADIENT DEMONSTRATION")
    print("This script demonstrates the enhanced analytical gradient function")
    print("with FINUFFT parameter tuning and performance optimizations.\n")
    
    # Run all demonstrations
    benchmark_results = benchmark_finufft_parameters()
    demonstrate_adaptive_grid()
    demonstrate_performance_modes()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Check the output directories for saved results:")
    print("- ./benchmark_output/")
    print("- ./adaptive_demo_output/")
    print("- ./performance_demo_output/")
    
    print("\nKey takeaways:")
    print("1. upsampfac=1.25 provides significant speedup for many applications")
    print("2. Adaptive grid estimation automatically optimizes grid size")
    print("3. skip_interpolation=True provides major performance benefits")
    print("4. Parameter tuning can achieve 2-5x performance improvements")
    print("5. Default upsampling_factor increased to 3.0 for better accuracy")

if __name__ == "__main__":
    main() 