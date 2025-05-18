#!/usr/bin/env python
"""
Example script demonstrating the use of the standalone analytical gradient function.

This example shows how to:
1. Generate sample data
2. Calculate the analytical gradient
3. Compute and access the averaged result
4. Save the results to disk with the proper directory structure
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from QM_FFT_Analysis.utils import calculate_analytical_gradient

def generate_sample_data(n_points=1000, n_trans=5):
    """Generate sample 3D data for demonstration."""
    # Generate random coordinates
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Calculate radius (distance from origin)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Generate multiple time points with different patterns
    strengths = np.zeros((n_trans, n_points), dtype=np.complex128)
    
    for t in range(n_trans):
        # Create a pattern that varies with time
        # Each time point has a different frequency content
        scale = 1.0 + 0.2 * t  # Increasing scale parameter
        
        # Real part: Gaussian modulated by oscillation
        real_part = np.exp(-r**2 / scale**2) * np.cos(r * (t + 1))
        
        # Imaginary part: Simple phase variation
        imag_part = np.sin(r * (t + 1))
        
        # Combine into complex strength
        strengths[t, :] = real_part + 1j * imag_part
    
    return x, y, z, strengths

def main():
    """Main function demonstrating the analytical gradient calculation."""
    # Output directory
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    x, y, z, strengths = generate_sample_data(n_points=1000, n_trans=5)
    
    # 1. Calculate analytical gradient with averaging (default)
    print("Calculating analytical gradient with averaging...")
    results_with_avg = calculate_analytical_gradient(
        x=x, y=y, z=z, strengths=strengths,
        subject_id="example_with_avg",
        output_dir=output_dir,
        average=True  # Default, can be omitted
    )
    
    # Examine the results
    print("\nResults with averaging:")
    print(f"- Gradient map shape: {results_with_avg['gradient_map_nu'].shape}")
    print(f"- Average gradient shape: {results_with_avg['gradient_average_nu'].shape}")
    print(f"- Grid dimensions: {results_with_avg['coordinates']['nx']} x {results_with_avg['coordinates']['ny']} x {results_with_avg['coordinates']['nz']}")
    
    # 2. Calculate analytical gradient without averaging
    print("\nCalculating analytical gradient without averaging...")
    results_no_avg = calculate_analytical_gradient(
        x=x, y=y, z=z, strengths=strengths,
        subject_id="example_no_avg",
        output_dir=output_dir,
        average=False
    )
    
    # Examine the results
    print("\nResults without averaging:")
    print(f"- Gradient map shape: {results_no_avg['gradient_map_nu'].shape}")
    print("- Average gradient: Not computed")
    
    # 3. Display directory structure created
    print("\nDirectory structure created:")
    print(f"- {output_dir}/example_with_avg/")
    print(f"  - Analytical_FFT_Gradient_Maps/")
    print(f"    - average_gradient.h5")
    print(f"    - AllTimePoints/")
    print(f"      - all_gradients.h5")
    
    # 4. Load and verify data from files
    with_avg_file = output_dir / "example_with_avg" / "Analytical_FFT_Gradient_Maps" / "average_gradient.h5"
    
    if with_avg_file.exists():
        print("\nVerifying data in HDF5 files:")
        with h5py.File(with_avg_file, 'r') as f:
            keys = list(f.keys())
            print(f"- Keys in average_gradient.h5: {keys}")
            if 'gradient_average_nu' in f:
                print(f"- Average gradient shape in file: {f['gradient_average_nu'].shape}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 