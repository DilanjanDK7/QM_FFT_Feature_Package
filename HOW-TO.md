# How to Install and Use QM_FFT_Analysis

This guide provides practical steps for installing and using the `QM_FFT_Analysis` package.

## Installation

We recommend using a virtual environment to manage dependencies.

1.  **Clone the Repository:**
    If you haven't already, get the code from the repository.
    ```bash
    # git clone <repository_url> # Replace with the actual URL
    # cd QM_FFT_Feature_Package
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Using venv (Python 3 standard library)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Or using Conda
    # conda create -n qmfft_env python=3.8 # Or your preferred Python version
    # conda activate qmfft_env
    ```

3.  **Install the Package and Dependencies:**
    Navigate to the root directory of the project (where `pyproject.toml` is located) and run:
    
    *   **For regular use:**
        ```bash
        pip install .
        ```
        This command uses the information in `pyproject.toml` to install the package and its required dependencies (`numpy`, `scipy`, `finufft`, `plotly`).

    *   **For development (editable install):**
        If you plan to modify the code, use an editable install:
        ```bash
        pip install -e .
        ```
        This installs the package in a way that changes you make to the source code are immediately reflected when you import and use the package.

## Basic Usage Example

Here's how to run the main processing pipeline using the `MapBuilder` class:

```python
import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import MapBuilder

# 1. Define Inputs
subject_id = "example_subject_01"
output_dir = Path("./example_output") 

# --- Generate some random example data ---
# Usually, you would load your actual data here
n_points = 500  # Number of non-uniform points
n_trans = 3     # Number of transforms (e.g., time points)

# Non-uniform coordinates (replace with your actual coordinates)
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)

# Complex strength values at each point for each transform
strengths_real = np.random.randn(n_trans, n_points)
strengths_imag = np.random.randn(n_trans, n_points)
strengths = strengths_real + 1j * strengths_imag
# --- End example data generation ---

# 2. Initialize MapBuilder
# It will estimate grid dimensions automatically by default
print(f"Initializing MapBuilder for {subject_id}...")
map_builder = MapBuilder(
    subject_id=subject_id,
    output_dir=output_dir,
    x=x, 
    y=y, 
    z=z,
    strengths=strengths,
    eps=1e-6,           # Desired FINUFFT precision
    dtype='complex128'  # Data type for calculations
)

# 3. Generate K-Space Masks
# The process_map method below uses spherical masks by default.
# Alternatively, you can generate masks manually first:
print("Generating K-Space masks...")
# map_builder.generate_kspace_masks(n_centers=2, radius=0.5) # Spherical (default in process_map)
# map_builder.generate_cubic_mask(kx_min=-0.1, kx_max=0.1, ky_min=-0.1, ky_max=0.1, kz_min=-0.1, kz_max=0.1)
# map_builder.generate_slice_mask(axis='z', k_value=0)
# map_builder.generate_slab_mask(axis='x', k_min=0.05, k_max=0.15)
# See docs/map_builder_guide.md for details on these methods.
# If generating manually, you might call compute_inverse_maps, etc., separately
# or call process_map with n_centers=0 if you don't want extra spherical masks.

# 4. Run the Full Processing Pipeline (Using Default Spherical Masks)
# This performs: forward FFT -> spherical mask generation -> inverse map -> gradient -> analysis
print("Running processing pipeline...")
analyses_to_run = [
    'magnitude', 
    'phase', 
    'local_variance', 
    'temporal_diff_magnitude', # Only computes if n_trans >= 2
    'temporal_diff_phase'      # Only computes if n_trans >= 2
]

map_builder.process_map(
    n_centers=3,        # Number of *spherical* k-space masks to generate by default
    radius=0.6,         # Radius of spherical k-space masks
    analyses_to_run=analyses_to_run, # Specify which analyses to run
    k_neighbors_local_var=5, # k for local variance calculation
    skip_interpolation=True  # Default is True for faster processing
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
# This is up to 245x faster than traditional methods
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

# If you need interpolated grid data (useful for visualization):
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

1. **Ultra-Fast Processing**: With the default `skip_interpolation=True`, computation is up to 245x faster than traditional methods.
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
*   **Gradient Map (`compute_gradient_maps`)**: Calculates the spatial rate of change (gradient magnitude) of the signal. With `skip_interpolation=True` (default), this is done directly on the non-uniform points for up to 245x faster performance. When `skip_interpolation=False`, the signal is interpolated onto a regular grid for traditional gradient calculation.
*   **Analysis (`analyze_inverse_maps`)**: Computes various metrics directly on the non-uniform inverse maps:
    *   `magnitude`/`phase`: Basic properties of the complex signal at each point.
    *   `local_variance`: Measures spatial heterogeneity of the magnitude signal.
    *   `temporal_difference`: Measures how magnitude/phase change between consecutive transforms (if `n_trans > 1`).

## Output Files

When running the package, three HDF5 files are created for each subject in the output directory:

1. `data.h5`: Contains raw computational results
   - Forward FFT results
   - K-space masks
   - Inverse maps
   - Gradient maps (on grid if skip_interpolation=False, otherwise just references to non-uniform data)

2. `analysis.h5`: Contains analysis results
   - Magnitude and phase calculations
   - Local variance metrics (with k-neighbor specification)
   - Temporal difference calculations
   - Analysis summary group

3. `enhanced.h5`: Contains enhanced feature results (if enabled)
   - Spectral metrics (slope, entropy)
   - Analytical gradient maps
   - Higher-order moments
   - Excitation maps

## Performance Considerations

When working with large datasets, consider the following:

1. **Memory Requirements**
   - Small scale (1K points, 5 times): ~100MB RAM
   - Medium scale (5K points, 10 times): ~500MB RAM
   - Large scale (50K points, 100 times): ~4GB RAM
   - Using skip_interpolation=True reduces memory usage considerably

2. **Storage Requirements**
   - Small scale: ~2MB total
   - Medium scale: ~18MB total
   - Large scale: ~1.7GB total
   - Using skip_interpolation=True reduces storage requirements


3. **Processing Time**
   - Small scale: ~1-2 seconds
   - Medium scale: ~5-10 seconds
   - Large scale: ~88 seconds

4. **Optimization Tips**
   - Use analytical gradient method for faster processing
   - Enable HDF5 compression for efficient storage
   - Consider batch processing for very large datasets 

5. **Performance Tips**
- Always use skip_interpolation=True when you don't specifically need grid-interpolated data
- This provides up to 245x speedup for gradient calculations
- Only use skip_interpolation=False when you need to:
    - Visualize data on a regular grid
    - Export to NIfTI format (which requires regular grid data)
    - Use tools that specifically require data on a regular grid 