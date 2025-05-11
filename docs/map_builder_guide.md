# MapBuilder Guide

This guide explains the `MapBuilder` class found in `QM_FFT_Analysis.utils.map_builder`.

## Overview

The `MapBuilder` class provides a pipeline for analyzing 3D scattered data using Non-Uniform Fast Fourier Transforms (NUFFT), specifically leveraging the FINUFFT library. It handles multiple "transforms" (e.g., time points) associated with the same set of non-uniform spatial coordinates.

The primary workflow involves:
1.  Transforming strength data at non-uniform points to a uniform k-space grid (Forward FFT).
2.  Generating and applying masks in k-space to isolate frequency components of interest.
3.  Transforming the masked k-space data back to the original non-uniform points (Inverse Map).
4.  Calculating the spatial gradient magnitude of the inverse map (interpolated onto the uniform grid).
5.  Calculating additional analysis metrics (magnitude, phase, local variance, temporal difference) directly on the non-uniform inverse map data.

## Initialization

```python
from QM_FFT_Analysis.utils import MapBuilder
import numpy as np

# Example data
subject_id = "subj01"
output_dir = "./results"
n_points = 1000
n_trans = 5 # e.g., 5 time points
x = np.random.rand(n_points) * 10 - 5
y = np.random.rand(n_points) * 10 - 5
z = np.random.rand(n_points) * 10 - 5
strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)

map_builder = MapBuilder(
    subject_id=subject_id,
    output_dir=output_dir,
    x=x, y=y, z=z,
    strengths=strengths,
    nx=32, ny=32, nz=32, # Optional: Specify grid dimensions
    estimate_grid=False, # Set to False if providing nx, ny, nz
    eps=1e-6, # FINUFFT precision
    upsampling_factor=2, # Factor used if estimate_grid=True
    dtype='complex128', # Data type for calculations
    normalize_fft_result=False, # Optionally normalize FFT power
    padding=0, # Deprecated, for test compatibility
    stride=1 # Stride for k-space sampling
)
```

**Parameters:**

*   `subject_id` (str): A unique identifier for this dataset/run.
*   `output_dir` (str | Path): The base directory where results will be saved. A subdirectory named after `subject_id` will be created.
*   `x`, `y`, `z` (ndarray): 1D arrays containing the coordinates of the non-uniform points.
*   `strengths` (ndarray): An array containing the complex strength values at each point. Can be 1D (shape `(N,)` for a single transform) or 2D (shape `(n_trans, N)` for multiple transforms, e.g., time points). `N` must equal `len(x)`.
*   `nx`, `ny`, `nz` (int, optional): The desired dimensions of the uniform grid. If `estimate_grid` is `True` or these are not provided, the grid size will be estimated based on the number of points and `upsampling_factor`.
*   `eps` (float, optional): The requested precision for FINUFFT calculations (default: `1e-6`).
*   `upsampling_factor` (int, optional): Used when `estimate_grid=True`. Multiplies the cube root of the number of points to estimate grid size (default: `2`).
*   `dtype` (str, optional): The numpy data type for complex calculations (default: `'complex128'`).
*   `estimate_grid` (bool, optional): If `True` (default), estimate `nx, ny, nz`. If `False`, `nx, ny, nz` must be provided.
*   `normalize_fft_result` (bool, optional): If `True`, compute and save the normalized power spectrum density of the forward FFT result (default: `False`).
*   `padding` (int, optional): Deprecated parameter, kept for test compatibility (default: `0`).
*   `stride` (int, optional): The step size used for calculating k-space frequency coordinates (`kx, ky, kz`) via `np.fft.fftfreq` (default: `1`).

## Core Methods

### `compute_forward_fft()`

*   Performs the Type 1 NUFFT (non-uniform points to uniform grid).
*   Takes `self.strengths` (shape `(n_trans, n_points)`) as input.
*   Stores the result in `self.fft_result` (shape `(n_trans, nx, ny, nz)`).
*   Saves the result to `data/forward_fft.npy`.
*   If `self.normalize_fft_result` is `True`, calculates the power spectrum density, stores it in `self.fft_prob_density`, and saves it to `data/fft_prob_density.npy`.

### Mask Generation Methods

The following methods generate boolean masks (shape `(nx, ny, nz)`) representing regions in k-space. Each method appends the generated mask to the `self.kspace_masks` list and saves it to `data/kspace_mask_{i}.npy`.

*   **`generate_kspace_masks(n_centers=2, radius=0.5)`**
    *   Generates `n_centers` spherical masks.
    *   Each sphere is centered at a randomly chosen k-space grid point.
    *   `radius` (float): The radius of the spheres in the k-space units defined by `self.kx`, `self.ky`, `self.kz`.

*   **`generate_cubic_mask(kx_min, kx_max, ky_min, ky_max, kz_min, kz_max)`**
    *   Generates a single cubic (box) mask.
    *   Selects points where `kx_min <= Kx < kx_max` AND `ky_min <= Ky < ky_max` AND `kz_min <= Kz < kz_max`.
    *   Boundaries are in the k-space units defined by `self.kx`, `self.ky`, `self.kz`.

*   **`generate_slice_mask(axis, k_value)`**
    *   Generates a single mask selecting a 2D slice.
    *   `axis` (str): The axis normal to the slice ('x', 'y', or 'z').
    *   `k_value` (float): The target k-coordinate. The slice corresponding to the grid frequency (`self.kx`, `self.ky`, or `self.kz`) closest to this value is selected.

*   **`generate_slab_mask(axis, k_min, k_max)`**
    *   Generates a single mask selecting a 3D slab (a range of slices).
    *   `axis` (str): The axis normal to the slab ('x', 'y', or 'z').
    *   `k_min`, `k_max` (float): The minimum and maximum k-coordinates (inclusive) defining the slab boundaries along the specified axis.

**Note:** You can call these methods multiple times or mix different mask types before proceeding to the inverse transform.

### `compute_inverse_maps()`

*   Performs the Type 2 NUFFT (uniform grid to non-uniform points) for **each mask** currently stored in `self.kspace_masks`.
*   Applies each mask in `self.kspace_masks` to `self.fft_result`.
*   Transforms the masked k-space data back to the original non-uniform point locations defined by `self.x`, `self.y`, `self.z`.
*   Stores the results (list of arrays, each shape `(n_trans, n_points)`) in `self.inverse_maps`.
*   Saves each inverse map to `data/inverse_map_{i}.npy`.

### `compute_gradient_maps(use_analytical_method=None, skip_interpolation=Trueuse_analytical_method=None, skip_interpolation=True)`

*   Calculates the spatial gradient magnitude for each inverse map.
*   **Parameters:**
    * `use_analytical_method` (bool, optional): If True, uses the analytical k-space method for calculating gradients (only available if enhanced features are enabled). Default is None (uses config setting if enhanced features enabled).
    * `skip_interpolation` (bool, optional): Whether to skip interpolation to regular grid. Default is True for significantly improved performance.
*   **When skip_interpolation=True (Default)**:
    * Operates directly on non-uniform inverse map data (dramatically faster)
    * Can be up to 245x faster than using interpolation
    * Stores the gradient on non-uniform points, matching the original data coordinates
    * Uses minimal memory since no grid interpolation is performed
    * Most suitable for subsequent processing that works with non-uniform data
*   **When skip_interpolation=False**:
    * Interpolates the complex data from each inverse map onto the uniform grid (`nx, ny, nz`) using `scipy.interpolate.griddata` (linear interpolation)
    * Calculates the gradient components (`dx`, `dy`, `dz`) of the *interpolated* data on the grid using `np.gradient`
    * Computes the gradient magnitude `sqrt(|dx|^2 + |dy|^2 + |dz|^2)` on the grid
    * Requires significantly more processing time and memory
    * Suitable for visualization or when working with tools that require grid-based data
*   **Storage Differences:**
    * With skip_interpolation=True, references to original non-uniform data are stored in `self.gradient_maps`
    * With skip_interpolation=False, grid-interpolated data is stored in `self.gradient_maps` (shape `(n_trans, nx, ny, nz)`)
*   Saves each gradient map to the appropriate format in `data/gradient_map_{i}.h5`
*   **Parameters:**
    * `use_analytical_method` (bool, optional): If True, uses the analytical k-space method for calculating gradients (only available if enhanced features are enabled). Default is None (uses config setting if enhanced features enabled).
    * `skip_interpolation` (bool, optional): Whether to skip interpolation to regular grid. Default is True for significantly improved performance.
*   **When skip_interpolation=True (Default)**:
    * Operates directly on non-uniform inverse map data (dramatically faster)
    * Can be up to 245x faster than using interpolation
    * Stores the gradient on non-uniform points, matching the original data coordinates
    * Uses minimal memory since no grid interpolation is performed
    * Most suitable for subsequent processing that works with non-uniform data
*   **When skip_interpolation=False**:
    * Interpolates the complex data from each inverse map onto the uniform grid (`nx, ny, nz`) using `scipy.interpolate.griddata` (linear interpolation)
    * Calculates the gradient components (`dx`, `dy`, `dz`) of the *interpolated* data on the grid using `np.gradient`
    * Computes the gradient magnitude `sqrt(|dx|^2 + |dy|^2 + |dz|^2)` on the grid
    * Requires significantly more processing time and memory
    * Suitable for visualization or when working with tools that require grid-based data
*   **Storage Differences:**
    * With skip_interpolation=True, references to original non-uniform data are stored in `self.gradient_maps`
    * With skip_interpolation=False, grid-interpolated data is stored in `self.gradient_maps` (shape `(n_trans, nx, ny, nz)`)
*   Saves each gradient map to the appropriate format in `data/gradient_map_{i}.h5`

### `analyze_inverse_maps(analyses_to_run=['magnitude', 'phase'], k_neighbors=5, save_format='hdf5', compute_enhanced=None, enable_local_variance=False)`

*   Performs selected analyses directly on the non-uniform inverse maps (`self.inverse_maps`).
*   `analyses_to_run` (list): Specifies which analyses to perform. Options:
    *   `'magnitude'`: Computes `np.abs(inverse_map)`. Saved as `analysis/map_{i}_magnitude.npy`.
    *   `'phase'`: Computes `np.angle(inverse_map)`. Saved as `analysis/map_{i}_phase.npy`.
    *   `'local_variance'`: Computes the variance of the magnitude within the `k_neighbors` nearest neighbors for each point. Requires `scipy`. Saved as `analysis/map_{i}_local_variance_k{k}.npy`. **Note:** `enable_local_variance` must be set to `True` for this analysis to run, even if it's included in `analyses_to_run`.
    *   `'temporal_diff_magnitude'`: Computes the difference in magnitude between consecutive transforms (`t` and `t-1`). Requires `n_trans >= 2`. Saved as `analysis/map_{i}_temporal_diff_magnitude.npy`.
    *   `'temporal_diff_phase'`: Computes the difference in phase between consecutive transforms. Requires `n_trans >= 2`. Saved as `analysis/map_{i}_temporal_diff_phase.npy`.
*   `k_neighbors` (int): The number of neighbors `k` to use for the `local_variance` calculation (default: `5`).
*   `save_format` (str, optional): Specifies how to save the summary of analysis results (default: `'hdf5'`).
    *   `'hdf5'`: Saves the entire `self.analysis_results` dictionary structure into a single HDF5 file named `analysis/analysis_summary.h5`. This provides a convenient way to store all analysis outputs together.
    *   `'npz'`: Skips saving the summary file. Logs a message indicating that individual `.npy` files were saved and the calling script is responsible for creating any summary files (e.g., `.npz`).
    *   Other values: Logs a warning and does not save a summary file.
*   `compute_enhanced` (bool, optional): Whether to compute enhanced metrics if available (default: `None`). If `None`, uses the value from `self.enable_enhanced_features`.
*   `enable_local_variance` (bool, optional): Whether to calculate local variance (default: `False`). Set to `True` to enable this computationally expensive calculation when 'local_variance' is in `analyses_to_run`.
*   Stores results in the `self.analysis_results` dictionary, keyed by map index (e.g., `'map_0'`) and analysis type (e.g., `'magnitude'`).
*   **Important:** Regardless of `save_format`, each calculated analysis result (magnitude, phase, etc.) is **always** saved to its individual `.npy` file in the `analysis/` subdirectory.

### `generate_volume_plot(data, filename, ...)`

*   Helper function to create interactive 3D volume plots using Plotly.
*   Requires 3D input data (e.g., one transform from `self.gradient_maps` or an interpolated inverse map).
*   Saves the plot as an HTML file in the `plots/` directory.
*   *Note:* This currently needs manual calls if you want plots for specific outputs or transforms.

### `process_map(n_centers=1, radius=0.5, analyses_to_run=['magnitude'], k_neighbors_local_var=5, use_analytical_gradient=None, skip_interpolation=True, use_analytical_gradient=None, calculate_local_variance=False, use_analytical_gradient=None, skip_interpolation=True)`

*   Orchestrates the main pipeline:
    1.  `compute_forward_fft()`
    2.  `generate_kspace_masks(n_centers, radius)` (Note: Only calls the spherical mask generator by default)
    3.  `compute_inverse_maps()`
    4.  `compute_gradient_maps(use_analytical_method=use_analytical_gradient, skip_interpolation=skip_interpolationuse_analytical_method=use_analytical_gradientuse_analytical_method=use_analytical_gradient, skip_interpolation=skip_interpolation)`
    5.  `analyze_inverse_maps(analyses_to_run, k_neighbors=k_neighbors_local_var, enable_local_variance=calculate_local_variance)`
*   **Parameters:**
    * `n_centers` (int): Number of spherical masks to generate (default: 2)
    * `radius` (float): Radius of the spherical masks (default: 0.5)
    * `analyses_to_run` (list): Analyses to perform on inverse maps (default: ['magnitude'])
    * `k_neighbors_local_var` (int): Number of neighbors for local variance calculation (default: 5)
    * `use_analytical_gradient` (bool): Whether to use analytical gradient method (default: None, uses config setting)
    * `skip_interpolation` (bool): Whether to skip interpolation to regular grid (default: True)
*   **Performance:**
    * Using the default `skip_interpolation=True` provides significantly faster processing (up to 245x speedup)
    * Only set `skip_interpolation=False` when you specifically need grid-interpolated data for visualization or other grid-based tools
*   Provides a convenient way to run the standard sequence of operations **with spherical masks**.
*   If you want to use cubic, slice, or slab masks, you should typically call the specific generation methods *before* calling `process_map` (and potentially set `n_centers=0` if you don't want additional spherical masks), or call the steps individually.
*   Parameters:
    *   `n_centers` (int): Number of spherical mask centers (default: `1`).
    *   `radius` (float): Radius of spherical masks (default: `0.5`).
    *   `analyses_to_run` (list): Analyses to run (default: `['magnitude']`).
    *   `k_neighbors_local_var` (int): k for local variance calculation (default: `5`).
    *   `use_analytical_gradient` (bool): Whether to use analytical gradient calculation. If `None`, uses the value from config if enhanced features are enabled (default: `None`).
    *   `calculate_local_variance` (bool): Whether to calculate local variance. This is passed as `enable_local_variance` to `analyze_inverse_maps` (default: `False`).

## Attributes

Key attributes storing results:

*   `fft_result`: The raw forward FFT result (uniform grid), shape `(n_trans, nx, ny, nz)`.
*   `fft_prob_density`: Normalized power spectrum density (if calculated), shape `(n_trans, nx, ny, nz)`.
*   `kspace_masks`: List of boolean k-space masks, each shape `(nx, ny, nz)`.
*   `inverse_maps`: List of complex inverse maps (non-uniform points), each shape `(n_trans, n_points)`.
*   `gradient_maps`: 
    * With `skip_interpolation=True`: List of gradient maps on non-uniform points, each shape `(n_trans, n_points)`.
    * With `skip_interpolation=False`: List of gradient maps on uniform grid, each shape `(n_trans, nx, ny, nz)`.
*   `analysis_results`: Dictionary holding the results from `analyze_inverse_maps`. Structure: `{'map_0': {'magnitude': ndarray, 'phase': ndarray, ...}, 'map_1': {...}, ...}`.

## Output Files

Results are saved in the specified `output_dir/<subject_id>/`:

*   `data/`: Contains raw numerical results (forward FFT, masks, inverse maps, gradient maps).
    * When `skip_interpolation=True`, gradient data references the non-uniform data.
    * When `skip_interpolation=False`, grid-interpolated gradient maps are stored.
*   `plots/`: Contains any generated Plotly visualizations as `.html` files.
*   `analysis/`: Contains results from `analyze_inverse_maps` (magnitude, phase, local variance, etc.).
*   `analysis/analysis_summary.h5`: (Optional) Contains a structured summary of all analysis results if `save_format='hdf5'` was used in `analyze_inverse_maps`.
*   `map_builder.log`: Log file for the run.

## Performance Optimization

The `skip_interpolation` parameter provides a significant performance optimization:

* **With skip_interpolation=True (Default)**:
  * Processing time: Up to 245x faster for gradient calculations
  * Memory usage: Significantly reduced (no grid interpolation)
  * Storage: Smaller files (only non-uniform data)
  * Best for: Most processing scenarios, especially large datasets

* **With skip_interpolation=False**:
  * Processing time: Much slower (requires interpolation)
  * Memory usage: Higher (needs to store full grid)
  * Storage: Larger files (full grid data)
  * Best for: Visualization, integration with grid-based tools, or when NIfTI export is needed

### Example usage with performance comparison:

```python
import time
from QM_FFT_Analysis.utils import MapBuilder

# Initialize MapBuilder...
map_builder = MapBuilder(
    subject_id="performance_test",
    output_dir="./output",
    x=x, y=y, z=z,
    strengths=strengths
)

# Run with skip_interpolation=True (default)
start_time = time.time()
map_builder.process_map(n_centers=2, radius=0.5, skip_interpolation=True)
fast_time = time.time() - start_time
print(f"Processing time with skip_interpolation=True: {fast_time:.2f} seconds")

# Run with skip_interpolation=False (slower but provides grid data)
start_time = time.time()
map_builder.process_map(n_centers=2, radius=0.5, skip_interpolation=False)
slow_time = time.time() - start_time
print(f"Processing time with skip_interpolation=False: {slow_time:.2f} seconds")
print(f"Speedup factor: {slow_time/fast_time:.2f}x") 