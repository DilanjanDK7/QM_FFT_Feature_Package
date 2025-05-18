# Analytical Gradient Function Guide

This document provides a comprehensive guide to the standalone `calculate_analytical_gradient` function, including its theoretical background, implementation details, and usage examples.

## Overview

The `calculate_analytical_gradient` function computes the radial gradient of a function directly in k-space using the Fourier differentiation theorem. This method is significantly faster and more accurate than traditional numerical differentiation approaches, especially for non-uniform data.

## Theoretical Background

### Mathematical Foundation

The analytical gradient calculation is based on the Fourier differentiation theorem. For a function $f(\mathbf{x})$ with Fourier transform $F(\mathbf{k})$, the radial derivative can be computed as:

$$\frac{\partial f}{\partial r}(\mathbf{x}) = \mathcal{F}^{-1}\bigl\{\,i2\pi\,\|\mathbf{k}\|\,F(\mathbf{k})\bigr\}$$

Where:
- $\mathcal{F}^{-1}$ is the inverse Fourier transform
- $\|\mathbf{k}\|$ is the magnitude of the k-space vector
- $i2\pi$ is the complex coefficient from the differentiation property

This approach allows us to compute the gradient in a single step by weighting the k-space data by $i2\pi\|\mathbf{k}\|$ before applying the inverse FFT, which is mathematically equivalent to the limit of infinitesimal shell differences but computationally much faster.

### Advantages Over Numerical Methods

1. **Efficiency**: Computes the gradient in a single step, rather than requiring multiple derivatives and combinations.
2. **Accuracy**: Avoids numerical errors associated with finite-difference methods.
3. **Resolution Independence**: The accuracy doesn't depend on the grid spacing.
4. **Non-Uniform Data Support**: Works directly with non-uniform data through NUFFT.

## Function Signature

```python
calculate_analytical_gradient(
    x, y, z, strengths, 
    subject_id="subject", 
    output_dir=None,
    nx=None, ny=None, nz=None,
    eps=1e-6,
    dtype='complex128',
    estimate_grid=True,
    upsampling_factor=2,
    export_nifti=False,
    affine_transform=None,
    average=True,
    skip_interpolation=True
)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | ndarray | 1D arrays of coordinates for non-uniform points |
| `strengths` | ndarray | Complex array of source strengths. Shape can be (N,) for single transform or (n_trans, N) for multiple |
| `subject_id` | str | Identifier for the subject, used for output file naming |
| `output_dir` | str or Path | Directory to save outputs. If None, no files are saved |
| `nx`, `ny`, `nz` | int | Grid dimensions. If not provided and `estimate_grid=True`, these are automatically estimated |
| `eps` | float | FINUFFT precision. Defaults to 1e-6 |
| `dtype` | str | Data type for complex values. Must be 'complex128' for FINUFFT |
| `estimate_grid` | bool | Whether to automatically estimate grid dimensions based on data distribution |
| `upsampling_factor` | float | Factor for grid estimation. Higher values → finer grid |
| `export_nifti` | bool | Whether to export results as NIfTI files (requires `skip_interpolation=False`) |
| `affine_transform` | ndarray | 4x4 affine transformation matrix for NIfTI export |
| `average` | bool | Whether to compute and save the average gradient over time points |
| `skip_interpolation` | bool | Whether to skip interpolation to regular grid. Default is True for significantly improved performance. |

## Return Value

The function returns a dictionary containing different keys depending on the `skip_interpolation` parameter:

### With `skip_interpolation=True` (Default)
| Key | Description |
|-----|-------------|
| `gradient_map_nu` | Gradient map on non-uniform points (n_trans, n_points) |
| `gradient_average_nu` | Average gradient map on non-uniform points (n_points) if average=True |
| `fft_result` | Forward FFT result |
| `coordinates` | Dictionary with coordinates and grid information |
| `k_space_info` | Information about the k-space grid (max_k, k_resolution, etc.) |

### With `skip_interpolation=False`
All of the above, plus:
| Key | Description |
|-----|-------------|
| `gradient_map_grid` | Gradient map on regular grid (n_trans, nx, ny, nz) |
| `gradient_average_grid` | Average gradient map on regular grid (nx, ny, nz) if average=True |

## Skip Interpolation Mode

The `skip_interpolation` parameter controls a critical performance optimization in the gradient calculation workflow:

### How It Works

1. **skip_interpolation=True** (Default)
   - After the analytical gradient is computed in k-space, the gradient is transformed directly back to the original non-uniform points
   - The intermediate step of interpolating to a regular grid is skipped
   - Results are stored in the `gradient_map_nu` key
   - Provides maximum performance and lowest memory usage

2. **skip_interpolation=False**
   - After computing the gradient on non-uniform points, an additional interpolation step creates a gridded version
   - Results are stored in both `gradient_map_nu` (non-uniform) and `gradient_map_grid` (regular grid)
   - Required for NIfTI export and certain visualization methods
   - Uses more memory and is computationally more intensive

### Performance Benefits

Based on benchmark tests with different dataset sizes:

| Points | With Interpolation (s) | Without Interpolation (s) | Speedup |
|--------|------------------------|---------------------------|---------|
| 100    | 0.0219                 | 0.0241                    | 0.91x   |
| 500    | 0.1517                 | 0.0608                    | 2.50x   |
| 1000   | 0.2157                 | 0.1977                    | 1.09x   |
| 2000   | 0.6021                 | 0.0959                    | 6.28x   |
| 5000   | 0.7257                 | 0.0819                    | 8.86x   |

Key findings:
- For small datasets (~100 points), there's a slight overhead in skip-interpolation mode
- For medium datasets (500-1000 points), the speedup ranges from 1.1x to 2.5x
- For large datasets (2000+ points), the speedup is substantial (6-9x)
- The speedup increases with dataset size, suggesting even greater benefits for very large datasets

### When to Use Each Mode

Use **skip_interpolation=True** (Default) when:
- Processing large datasets (1000+ points)
- Running batch analyses where performance is critical
- Memory usage is a concern
- You don't need to export to NIfTI or other grid-based formats
- You'll analyze the results directly on non-uniform points

Use **skip_interpolation=False** when:
- You need to export to NIfTI format
- You need to visualize the results on a regular grid
- You'll use grid-based analysis tools on the results
- You're processing small datasets where performance is less critical

## Automatic K-Space Optimization

The function automatically optimizes k-space parameters based on your input data:

1. **Optimal Grid Size**: Automatically calculates the optimal grid size based on:
   - Spatial distribution of the input points
   - Minimum distance between points (for Nyquist frequency)
   - Desired upsampling factor

2. **K-Space Extent**: Ensures the k-space coverage is sufficient for the highest frequencies in your data.

3. **Warnings**: If the estimated k-space extent is insufficient, a warning is issued suggesting grid size adjustments.

## Output File Structure

When `output_dir` is provided, the function creates the following directory structure:

```
output_dir/
└── subject_id/
    └── Analytical_FFT_Gradient_Maps/
        ├── average_gradient.h5      # Average gradient (if average=True)
        └── AllTimePoints/
            └── all_gradients.h5     # All time points data
```

### File Contents

The contents of the output files will depend on the `skip_interpolation` parameter:

1. **With skip_interpolation=True (Default):**
   - `gradient_map_nu`: Gradient maps on non-uniform points
   - `gradient_average_nu`: Average gradient on non-uniform points (if average=True)
   - `coordinates`: Coordinate information
   - `k_space_info`: K-space parameters and quality metrics
   - `fft_result`: Forward FFT result

2. **With skip_interpolation=False:**
   - All of the above, plus:
   - `gradient_map_grid`: Gradient maps interpolated to regular grid
   - `gradient_average_grid`: Average gradient interpolated to regular grid (if average=True)
   - NIfTI files in the filesystem (if export_nifti=True)

## NIfTI Export

NIfTI export is only available when `skip_interpolation=False` because NIfTI files require data on a regular grid. If you try to export NIfTI files with `skip_interpolation=True`, the function will log a warning and no NIfTI files will be created.

To successfully export NIfTI files:
```python
results = calculate_analytical_gradient(
    x=x, y=y, z=z, strengths=strengths,
    subject_id="example",
    output_dir="./output",
    skip_interpolation=False,  # Must be False for NIfTI export
    export_nifti=True,
    affine_transform=affine  # Optional 4x4 affine transformation
)
```

## Examples

### Basic Usage (With Skip Interpolation)

```python
import numpy as np
from QM_FFT_Analysis.utils import calculate_analytical_gradient

# Generate sample data
n_points = 1000
n_trans = 5  # Number of time points

# Create coordinates
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)

# Create complex strengths with 5 time points
strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

# Calculate the analytical gradient (skip_interpolation=True by default)
results = calculate_analytical_gradient(
    x=x, y=y, z=z, strengths=strengths,
    subject_id="example",
    output_dir="./output"
)

# Access the gradient map (non-uniform points)
gradient_map = results['gradient_map_nu']
print(f"Gradient map shape: {gradient_map.shape}")
```

### With Interpolation (for NIfTI Export)

```python
# Calculate with interpolation to regular grid
results_with_interp = calculate_analytical_gradient(
    x=x, y=y, z=z, strengths=strengths,
    subject_id="example_with_interp",
    output_dir="./output",
    skip_interpolation=False,  # Enable interpolation to regular grid
    export_nifti=True         # Export as NIfTI (requires skip_interpolation=False)
)

# Access both non-uniform and grid data
gradient_map_nu = results_with_interp['gradient_map_nu']
gradient_map_grid = results_with_interp['gradient_map_grid']
print(f"Grid gradient map shape: {gradient_map_grid.shape}")
```

### Custom Grid Size

```python
# Specify a custom grid size
results = calculate_analytical_gradient(
    x=x, y=y, z=z, strengths=strengths,
    subject_id="example_custom_grid",
    output_dir="./output",
    nx=64, ny=64, nz=64,
    estimate_grid=False
)
```

### Without Time Averaging

```python
# Disable time averaging
results = calculate_analytical_gradient(
    x=x, y=y, z=z, strengths=strengths,
    subject_id="example_no_avg",
    output_dir="./output",
    average=False
)
```

## Memory Optimization

For very large datasets, the memory usage can be further reduced by processing time points in batches:

```python
import numpy as np
from QM_FFT_Analysis.utils import calculate_analytical_gradient

# For large datasets with many time points
n_points = 10000
n_trans = 100

# Processing in batches of 10 time points
batch_size = 10
n_batches = n_trans // batch_size

# Create example data
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)
all_strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

# Process each batch
for i in range(n_batches):
    print(f"Processing batch {i+1}/{n_batches}")
    batch_strengths = all_strengths[i*batch_size:(i+1)*batch_size]
    
    # Use skip_interpolation=True for maximum performance
    results = calculate_analytical_gradient(
        x=x, y=y, z=z, strengths=batch_strengths,
        subject_id=f"batch_{i+1}",
        output_dir="./output",
        skip_interpolation=True
    )
```

## Performance Considerations

For optimal performance:

1. **Grid Size**: The function automatically estimates an optimal grid size, but you can specify it manually for more control.

2. **Upsampling Factor**: Increasing the upsampling factor improves accuracy but increases computation time. A value of 2.0 is usually sufficient.

3. **Memory Usage**: Memory usage scales with the product of grid dimensions (nx * ny * nz) and the number of time points.

4. **FINUFFT Precision**: The `eps` parameter controls FINUFFT precision. Lower values increase accuracy but slow down computation.

5. **Skip Interpolation**: Using `skip_interpolation=True` (default) provides substantial performance benefits, especially for large datasets.

## Troubleshooting

### Common Issues

1. **Insufficient k-space extent warning**:
   ```
   WARNING - Estimated k-space extent may be insufficient for the minimum spatial scale. 
   Consider increasing grid size or upsampling factor.
   ```
   **Solution**: Increase the upsampling factor or specify larger grid dimensions.

2. **Memory errors**:
   **Solution**: Try reducing the grid size, processing fewer time points at once, or using `skip_interpolation=True`.

3. **NIfTI export warnings**:
   ```
   WARNING: NIfTI export requires interpolation to a regular grid. No NIfTI files will be created.
   ```
   **Solution**: Set `skip_interpolation=False` when you need to export NIfTI files.

4. **Missing NIfTI files**:
   **Solution**: 
   - Ensure that `skip_interpolation=False` and `export_nifti=True`
   - Check that the `nibabel` package is installed (`pip install nibabel`)

## References

1. Diyabalanage, D. (2023). "Multiscale k-Space Gradient Mapping in fMRI: Theory, Shell Selection, and Excitability Proxy"

2. Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). "A Parallel Nonuniform Fast Fourier Transform Library Based on an 'Exponential of Semicircle' Kernel." SIAM Journal on Scientific Computing, 41(5), C479–C504. 