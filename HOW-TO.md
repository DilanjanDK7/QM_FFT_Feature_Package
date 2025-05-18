# How to Install and Use QM_FFT_Analysis

This guide provides practical steps for installing and using the `QM_FFT_Analysis` package.

## Installation

We recommend using a virtual environment to manage dependencies.

1. **Clone the Repository:**
    ```bash
    # git clone <repository_url> # Replace with the actual URL
    # cd QM_FFT_Feature_Package
    ```

2. **Create and Activate Virtual Environment:**
    ```bash
    # Using venv (Python 3 standard library)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Or using Conda
    # conda create -n qmfft_env python=3.8 # Or your preferred Python version
    # conda activate qmfft_env
    ```

3. **Install the Package and Dependencies:**
    Navigate to the root directory of the project (where `pyproject.toml` is located) and run:
    
   * **Standard installation:**
        ```bash
        pip install .
        ```

   * **Development (editable) installation:**
        ```bash
        pip install -e .
        ```

## Basic Usage Example

Here's how to run the main processing pipeline with the `MapBuilder` class:

```python
import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import MapBuilder

# 1. Define inputs and prepare data
subject_id = "example_subject_01"
output_dir = Path("./output") 

# Generate example data (replace with your actual data)
n_points = 500  # Number of non-uniform points
n_trans = 3     # Number of time points
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)
strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

# 2. Initialize MapBuilder
map_builder = MapBuilder(
    subject_id=subject_id,
    output_dir=output_dir,
    x=x, y=y, z=z,
    strengths=strengths,
    eps=1e-6,               # FINUFFT precision
    dtype='complex128',     # Data type for calculations
    enable_enhanced_features=True  # Enable enhanced features
)

# 3. Run the full processing pipeline
analyses_to_run = [
    'magnitude', 'phase',   # Basic analyses
    'local_variance',       # Spatial heterogeneity
    'temporal_diff_magnitude', 'temporal_diff_phase',  # Temporal changes
    'spectral_slope', 'spectral_entropy', 'anisotropy'  # Enhanced metrics
]


map_builder.process_map(
    n_centers=3,              # Number of k-space masks
    radius=0.6,               # Radius of k-space masks
    analyses_to_run=analyses_to_run,
    k_neighbors_local_var=5,  # For local variance calculation
    skip_interpolation=True,  # Default is True for faster processing
    use_analytical_gradient=True  # Faster gradient method
)

print(f"Processing complete. Results saved in: {map_builder.output_dir / subject_id}")

# 4. Access Results (Optional)
# Results are stored in attributes and saved to files
# print("FFT Result shape:", map_builder.fft_result.shape)
# print("Number of inverse maps:", len(map_builder.inverse_maps))
# print("Shape of first inverse map:", map_builder.inverse_maps[0].shape)
# print("Number of gradient maps:", len(map_builder.gradient_maps))
# print("Shape of first gradient map:", map_builder.gradient_maps[0].shape)
# print("Analysis results for map 0:", map_builder.analysis_results.get('map_0', {}).keys())
```

## Understanding Skip Interpolation Mode

The package offers two workflow modes for gradient computation, controlled by the `skip_interpolation` parameter:

### 1. Skip-Interpolation Mode (`skip_interpolation=True`, Default)

This is the default, high-performance mode that operates directly on non-uniform points:

```python
# Default behavior - skip_interpolation=True
map_builder.compute_gradient_maps(use_analytical_method=True)  # Fastest option

# Or explicitly specify
map_builder.compute_gradient_maps(use_analytical_method=True, skip_interpolation=True)
```

**Advantages:**
- **Performance:** Up to 9x faster processing, with greater benefits for larger datasets
- **Memory Efficient:** Consumes less memory as it avoids creating large grid arrays
- **Original Resolution:** Preserves the exact resolution of the original non-uniform data
- **Lower Storage Requirements:** Outputs smaller files with only the necessary data

**Limitations:**
- **Not compatible with NIfTI export:** Cannot directly create NIfTI files (requires grid data)
- **Limited Visualization Options:** Some visualization methods require regular grids
- **Not compatible with grid-based analysis tools**

**When to use:**
- Processing large datasets (1000+ points)
- When you need maximum performance
- When you'll only analyze the results on the original non-uniform points
- For batch processing of many subjects

### 2. Grid-Interpolation Mode (`skip_interpolation=False`)

This mode interpolates the data to a regular grid for traditional gradient calculation:

```python
# Enable grid interpolation
map_builder.compute_gradient_maps(use_analytical_method=True, skip_interpolation=False)
```

**Advantages:**
- **Compatibility:** Required for NIfTI export and grid-based visualizations
- **Standard Format:** Results on regular grid are compatible with most neuroimaging tools
- **Visualization:** Easier to create volume renderings and slice views

**Limitations:**
- **Performance:** 1.1x-9x slower than skip_interpolation mode (depending on dataset size)
- **Memory Usage:** Requires more memory to store grid arrays
- **Storage:** Creates larger output files with both non-uniform and grid data

**When to use:**
- When you need to export to NIfTI format
- When you need to visualize results as volume renderings
- When you'll use the results with tools that require regular grids
- For small datasets where performance is less critical

### Performance Comparison

| Points | With Interpolation (s) | Without Interpolation (s) | Speedup |
|--------|------------------------|---------------------------|---------|
| 100    | 0.0219                 | 0.0241                    | 0.91x   |
| 500    | 0.1517                 | 0.0608                    | 2.50x   |
| 1000   | 0.2157                 | 0.1977                    | 1.09x   |
| 2000   | 0.6021                 | 0.0959                    | 6.28x   |
| 5000   | 0.7257                 | 0.0819                    | 8.86x   |

## Using the Standalone Analytical Gradient Function

If you only need to compute the analytical radial gradient directly, without using the full MapBuilder pipeline, you can use the standalone function:

```python
import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import calculate_analytical_gradient

# 1. Define Inputs
subject_id = "gradient_example_01"
output_dir = Path("./gradient_output") 

# --- Generate sample data ---
n_points = 1000  # Number of non-uniform points
n_trans = 5      # Number of time points or transforms

# Non-uniform coordinates
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)

# Complex strengths (for example, a simple Gaussian function)
r = np.sqrt(x**2 + y**2 + z**2)
strengths_base = np.exp(-r**2)  # Base pattern

# Create multiple time points with variations
strengths = np.zeros((n_trans, n_points), dtype=np.complex128)
for t in range(n_trans):
    # Add time-dependent variations
    phase_shift = np.exp(1j * t * r / 2)
    strengths[t] = strengths_base * phase_shift

# 2. Calculate the analytical gradient with skip_interpolation (default)
# This is up to 9x faster than traditional methods
print(f"Calculating analytical gradient for {subject_id}...")
results = calculate_analytical_gradient(
    x=x, y=y, z=z, 
    strengths=strengths,
    subject_id=subject_id,
    output_dir=output_dir,
    # Optional parameters with their defaults:
    # estimate_grid=True,       # Auto-estimate grid size
    # upsampling_factor=2.0,    # For grid estimation
    # average=True,             # Calculate time average
    # skip_interpolation=True,  # Skip interpolation for maximum performance (default)
)

# 3. Access Results
print(f"Calculation complete. Results saved in: {output_dir / subject_id / 'Analytical_FFT_Gradient_Maps'}")

# The gradient maps on the original non-uniform points
gradient_maps = results['gradient_map_nu']  # Shape: (n_trans, n_points)
print(f"Gradient maps shape: {gradient_maps.shape}")

# The average gradient over time (if average=True was used)
if 'gradient_average_nu' in results:
    avg_gradient = results['gradient_average_nu']  # Shape: (n_points,)
    print(f"Average gradient shape: {avg_gradient.shape}")

# Information about the k-space parameters used
k_info = results['k_space_info']
print(f"K-space extent: {k_info['max_k']:.4f}")
print(f"K-space resolution: {k_info['k_resolution']:.4f}")

# If you need interpolated grid data (useful for visualization or NIfTI export):
results_grid = calculate_analytical_gradient(
    x=x, y=y, z=z, 
    strengths=strengths,
    subject_id=f"{subject_id}_grid",
    output_dir=output_dir,
    skip_interpolation=False,  # Enable interpolation to regular grid
    export_nifti=True,         # Optionally export as NIfTI (requires skip_interpolation=False)
    # affine_transform=None     # Affine transform for NIfTI
)

# Access both non-uniform and grid data
gradient_maps_grid = results_grid['gradient_map_grid']  # Shape: (n_trans, nx, ny, nz)
print(f"Grid gradient maps shape: {gradient_maps_grid.shape}")
```

The standalone function provides several advantages:

1. **Ultra-Fast Processing**: With the default `skip_interpolation=True`, computation is up to 9x faster than traditional methods.
2. **Flexibility**: Option to interpolate to a regular grid when needed for visualization or integration with grid-based tools.
3. **Simplicity**: Direct computation without needing to set up the full MapBuilder pipeline.
4. **Accuracy**: More accurate gradient calculation using the analytical formula.
5. **Time Averaging**: Built-in support for calculating time-averaged gradients.
6. **Automatic Optimization**: Automatically determines the optimal grid size and k-space parameters.

For more details on the standalone function, its theory, and advanced usage, see the [Analytical Gradient Guide](docs/analytical_gradient_guide.md).

## Core Functionality Explained

*   **Forward FFT (`compute_forward_fft`)**: Transforms your signal from the non-uniform points (`x`, `y`, `z`) where `strengths` are defined onto a regular 3D grid in k-space (frequency space). The size of this grid is estimated automatically or can be specified (`nx`, `ny`, `nz`).
*   **K-Space Masking (`generate_kspace_masks`)**: Creates spherical masks centered at random locations in k-space. This allows you to select specific frequency components from the forward FFT result.
*   **Inverse Map (`compute_inverse_maps`)**: Takes the masked k-space data and transforms it *back* to the original non-uniform point locations. This shows the spatial representation of the selected frequency components.
*   **Gradient Map (`compute_gradient_maps`)**: Calculates the spatial rate of change (gradient magnitude) of the signal. With `skip_interpolation=True` (default), this is done directly on the non-uniform points for up to 9x faster performance. When `skip_interpolation=False`, the signal is interpolated onto a regular grid for traditional gradient calculation and to enable NIfTI export.
*   **Analysis (`analyze_inverse_maps`)**: Computes various metrics directly on the non-uniform inverse maps:
    *   `magnitude`/`phase`: Basic properties of the complex signal at each point.
    *   `local_variance`: Measures spatial heterogeneity of the magnitude signal.
    *   `temporal_difference`: Measures how magnitude/phase change between consecutive transforms (if `n_trans > 1`).

## Output Files

When running the package, three HDF5 files are created for each subject in the output directory. The content varies based on the `skip_interpolation` setting:

### With `skip_interpolation=True` (Default)

1. **data.h5**: Raw computational results
   * `/forward_fft`: Complex FFT result on regular grid
   * `/kspace_masks/{mask_id}`: Binary masks in k-space
   * `/inverse_maps/{mask_id}`: Complex inverse maps for each mask on non-uniform points
   * `/params`: Processing parameters and metadata

2. **analysis.h5**: Analysis results
   * `/magnitude/{mask_id}`: Magnitude values on non-uniform points
   * `/phase/{mask_id}`: Phase angle values on non-uniform points
   * `/local_variance/{mask_id}`: Local variance metrics on non-uniform points
   * `/temporal_diff_magnitude/{mask_id}`: Temporal derivatives on non-uniform points
   * `/temporal_diff_phase/{mask_id}`: Phase changes over time on non-uniform points
   * `/summary`: Summary statistics for each metric

3. **enhanced.h5**: Enhanced feature results (if enabled)
   * `/analytical_gradients/{mask_id}`: Analytically computed gradients on non-uniform points
   * `/spectral_slope`, `/spectral_entropy`, `/anisotropy`: Spectral metrics
   * `/higher_moments`: Skewness and kurtosis values
   * `/excitation`: Neural activity estimates
   * `/params`: Configuration parameters

### With `skip_interpolation=False`

All of the above, plus additional interpolated grid datasets:
1. **data.h5**:
   * `/grid_inverse_maps/{mask_id}`: Complex inverse maps interpolated to regular grid
   * `/grid_gradient_maps/{mask_id}`: Gradient magnitude maps on regular grid

2. **enhanced.h5** (if enabled):
   * `/grid_analytical_gradients/{mask_id}`: Analytically computed gradients on regular grid
   * NIfTI files in the filesystem (if `export_nifti=True`)

## Performance Considerations

When working with large datasets, consider the following:

1. **Memory Requirements**
   - Skip-interpolation mode (`skip_interpolation=True`) uses significantly less memory
   - Small scale (1K points, 5 times): ~100MB RAM with interpolation, ~60MB without
   - Medium scale (5K points, 10 times): ~500MB RAM with interpolation, ~200MB without
   - Large scale (50K points, 100 times): ~4GB RAM with interpolation, ~1GB without

2. **Storage Requirements**
   - Skip-interpolation mode reduces storage requirements by 30-60%
   - Small scale: ~2MB total with interpolation, ~1.2MB without
   - Medium scale: ~18MB total with interpolation, ~7MB without
   - Large scale: ~1.7GB total with interpolation, ~0.7GB without

3. **Processing Time**
   - Skip-interpolation mode provides significant speedups for larger datasets
   - Small scale: ~1-2 seconds with interpolation, ~1-2 seconds without (minimal difference)
   - Medium scale: ~5-10 seconds with interpolation, ~2-4 seconds without (2.5x faster)
   - Large scale: ~88 seconds with interpolation, ~10-15 seconds without (6-8x faster)

4. **NIfTI Export Compatibility**
   - If you need to export NIfTI files, you must use `skip_interpolation=False`
   - When `skip_interpolation=True` and `export_nifti=True`, a warning is generated and no NIfTI files are created

## Workflow Selection Guide

### Choose Skip-Interpolation Mode (`skip_interpolation=True`) when:
- Processing large datasets (1000+ points)
- Running batch analysis on multiple subjects
- Memory and performance are critical
- You don't need NIfTI export
- You're working directly with the non-uniform point data

### Choose Grid-Interpolation Mode (`skip_interpolation=False`) when:
- Creating visualizations that require regular grids
- Exporting to NIfTI format
- Using other grid-based analysis tools
- Working with small datasets where performance is less critical
- Performing comparison with other methods that use regular grids

## Troubleshooting

### Common Issues

1. **"NIfTI export requires interpolation" warnings**:
   ```
   WARNING: NIfTI export requires interpolation to regular grid. No NIfTI files will be created.
   ```
   **Solution**: Set `skip_interpolation=False` when you need to export NIfTI files.

2. **Memory errors with large datasets**:
   **Solution**: Use `skip_interpolation=True` to reduce memory usage, or process smaller batches of data.

3. **Missing grid data in outputs**:
   **Solution**: Check if you used `skip_interpolation=True` (default). Set to `False` if you need grid data.

4. **Tests failing due to missing keys**:
   **Solution**: Tests expecting grid data need `skip_interpolation=False`. Update test assertions to check for appropriate outputs based on the interpolation setting.

For more help and examples, refer to the documentation or contact the developer. 