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
    k_neighbors_local_var=5,,  # For local variance calculation
    skip_interpolation=True  # Default is True for faster processing
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

# 2. Calculate the analytical gradient
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
    # export_nifti=False,       # Export to NIfTI format
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

# The function creates the following files:
# - {output_dir}/{subject_id}/Analytical_FFT_Gradient_Maps/average_gradient.h5 (if average=True)
# - {output_dir}/{subject_id}/Analytical_FFT_Gradient_Maps/AllTimePoints/all_gradients.h5
```

The standalone function provides several advantages:

1. **Simplicity**: Direct computation without needing to set up the full MapBuilder pipeline.
2. **Speed**: Much faster than the traditional gradient computation that requires interpolation.
3. **Accuracy**: More accurate gradient calculation using the analytical formula.
4. **Time Averaging**: Built-in support for calculating time-averaged gradients.
5. **Automatic Optimization**: Automatically determines the optimal grid size and k-space parameters.

For more details on the standalone function, its theory, and advanced usage, see the [Analytical Gradient Guide](docs/analytical_gradient_guide.md).

## Core Functionality Explained

*   **Forward FFT (`compute_forward_fft`)**: Transforms your signal from the non-uniform points (`x`, `y`, `z`) where `strengths` are defined onto a regular 3D grid in k-space (frequency space). The size of this grid is estimated automatically or can be specified (`nx`, `ny`, `nz`).
*   **K-Space Masking (`generate_kspace_masks`)**: Creates spherical masks centered at random locations in k-space. This allows you to select specific frequency components from the forward FFT result.
*   **Inverse Map (`compute_inverse_maps`)**: Takes the masked k-space data and transforms it *back* to the original non-uniform point locations. This shows the spatial representation of the selected frequency components.
*   **Gradient Map (`compute_gradient_maps`)**: Calculates the spatial rate of change (gradient magnitude) of the signal. Since the inverse map is on non-uniform points, the signal is first interpolated onto the regular grid before the gradient is calculated using standard numerical methods.
*   **Analysis (`analyze_inverse_maps`)**: Computes various metrics directly on the non-uniform inverse maps:
    *   `magnitude`/`phase`: Basic properties of the complex signal at each point.
    *   `local_variance`: Measures spatial heterogeneity of the magnitude signal.
    *   `temporal_difference`: Measures how magnitude/phase change between consecutive transforms (if `n_trans > 1`).

## Output Files

When running the package, three HDF5 files are created for each subject in the output directory:

### 1. `data.h5`: Raw computational results
   * `/forward_fft`: Complex FFT result on regular grid
   * `/kspace_masks/{mask_id}`: Binary masks in k-space
   * `/inverse_maps/{mask_id}`: Complex inverse maps for each mask
   * `/gradient_maps/{mask_id}`: Gradient magnitude maps
   * `/params`: Processing parameters and metadata

### 2. `analysis.h5`: Analysis results
   * `/magnitude/{mask_id}`: Magnitude values
   * `/phase/{mask_id}`: Phase angle values
   * `/local_variance/{mask_id}`: Local variance metrics
   * `/temporal_diff_magnitude/{mask_id}`: Temporal derivatives
   * `/temporal_diff_phase/{mask_id}`: Phase changes over time
   * `/summary`: Summary statistics for each metric

### 3. `enhanced.h5`: Enhanced feature results
   * `/spectral_slope`: Power law exponents
   * `/spectral_entropy`: Entropy of k-space distribution
   * `/anisotropy`: Directional preference metrics
   * `/higher_moments`: Skewness and kurtosis values
   * `/excitation`: Neural activity estimates
   * `/analytical_gradients/{mask_id}`: Analytically computed gradients
   * `/params`: Configuration parameters for enhanced features

## Performance Considerations

When working with large datasets, consider the following:

1. **Memory Requirements**
   - Small scale (1K points, 5 times): ~100MB RAM
   - Medium scale (5K points, 10 times): ~500MB RAM
   - Large scale (50K points, 100 times): ~4GB RAM

2. **Storage Requirements**
   - Small scale: ~2MB total
   - Medium scale: ~18MB total
   - Large scale: ~1.7GB total

3. **Processing Time**
   - Small scale: ~1-2 seconds
   - Medium scale: ~5-10 seconds
   - Large scale: ~88 seconds

4. **Optimization Tips**
   - Use analytical gradient method for faster processing
   - Enable HDF5 compression for efficient storage
   - Consider batch processing for very large datasets 
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
    use_analytical_gradient=True  # Faster gradient method
)

print(f"Processing complete. Results saved in: {map_builder.output_dir / subject_id}")

# 4. Alternative: Compute only enhanced metrics (much faster)
enhanced_metrics = map_builder.compute_enhanced_metrics(
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy']
)
print("Enhanced metrics computed:", list(enhanced_metrics.keys()))
```

## Processing Pipeline Explained

### Core Steps of the Pipeline

1. **Initialize MapBuilder**: Set up the computation environment
2. **Forward FFT**: Transform from non-uniform points to regular k-space grid
3. **Generate k-space masks**: Create spherical regions of interest in k-space
4. **Compute inverse maps**: Transform masked k-space data back to original points
5. **Calculate gradients**: Compute spatial rate of change (gradient magnitude)
6. **Analyze maps**: Calculate metrics on the inverse maps
7. **Enhanced metrics**: Compute advanced spectral and spatial metrics

### Key Methods and Their Functions

#### Main Pipeline Methods

* **`process_map()`**: Runs the complete pipeline
  ```python
  map_builder.process_map(
      n_centers=2,                # Number of k-space mask centers
      radius=0.5,                 # Radius of spherical masks
      analyses_to_run=['magnitude', 'phase', 'spectral_slope'],
      use_analytical_gradient=True,  # Faster gradient calculation method
      k_neighbors_local_var=5,    # Number of neighbors for local variance
      compression='gzip',         # HDF5 file compression
  )
  ```

* **`compute_forward_fft()`**: Calculates the FFT of non-uniform data
  ```python
  map_builder.compute_forward_fft(
      eps=1e-6,       # FINUFFT precision
      dtype='complex128'  # Data type
  )
  ```

* **`generate_kspace_masks()`**: Creates spherical masks in k-space
  ```python
  map_builder.generate_kspace_masks(
      n_centers=3,    # Number of mask centers
      radius=0.5,     # Radius of each mask
      random_seed=42  # For reproducible center locations
  )
  ```

* **`compute_inverse_maps()`**: Inverse FFT of masked k-space data
  ```python
  map_builder.compute_inverse_maps()  # Uses previously generated masks
  ```

* **`compute_gradient_maps()`**: Calculates spatial gradients
  ```python
  map_builder.compute_gradient_maps(
      use_analytical_method=True  # Use faster k-space method
  )
  ```

* **`analyze_inverse_maps()`**: Performs analyses on inverse maps
  ```python
  map_builder.analyze_inverse_maps(
      analyses_to_run=['magnitude', 'phase', 'local_variance'],
      k_neighbors=5  # For local variance
  )
  ```

#### Enhanced Features

* **`compute_enhanced_metrics()`**: Calculates advanced metrics directly from k-space
  ```python
  enhanced_metrics = map_builder.compute_enhanced_metrics(
      metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy', 
                      'higher_moments', 'excitation'],
      excitation_params={
          'hrf_type': 'canonical',  # HRF model to use
          'tr': 2.0,                # Repetition time in seconds 
          'oversampling': 10        # Temporal oversampling factor
      }
  )
  ```

## Enhanced Features in Detail

### Available Enhanced Metrics

1. **Spectral Slope**:
   * Measures the power-law exponent of the k-space power spectrum
   * Negative values indicate smoother signals with rapid decay of high frequencies
   ```python
   # Configure spectral slope calculation
   map_builder.compute_enhanced_metrics(
       metrics_to_run=['spectral_slope'],
       spectral_slope_params={
           'k_min': 0.1,  # Minimum k-value for fitting
           'k_max': 0.8   # Maximum k-value for fitting
       }
   )
   ```

2. **Spectral Entropy**:
   * Quantifies the diversity of frequency components in the signal
   * Higher values indicate more uniform distribution of power
   ```python
   # Configure spectral entropy calculation
   map_builder.compute_enhanced_metrics(
       metrics_to_run=['spectral_entropy'],
       spectral_entropy_params={
           'n_bins': 64  # Number of bins for histogram
       }
   )
   ```

3. **K-space Anisotropy**:
   * Measures directional preference in k-space distribution
   * Higher values indicate stronger directional patterns
   ```python
   # Configure anisotropy calculation
   map_builder.compute_enhanced_metrics(
       metrics_to_run=['anisotropy'],
       anisotropy_params={
           'moment': 2  # Order of the k-space tensor moment
       }
   )
   ```

4. **Higher Moments**:
   * Computes skewness and kurtosis of the inverse map values
   * Provides information about asymmetry and tailedness of value distribution
   ```python
   # Configure higher moments calculation
   map_builder.compute_enhanced_metrics(
       metrics_to_run=['higher_moments']
   )
   ```

5. **Excitation Maps**:
   * Estimates neural activity through HRF deconvolution
   * Requires at least 3 time points for effective calculation
   ```python
   # Configure excitation map calculation
   map_builder.compute_enhanced_metrics(
       metrics_to_run=['excitation'],
       excitation_params={
           'hrf_type': 'canonical',  # 'canonical', 'gamma', or 'boxcar'
           'tr': 2.0,                # Repetition time in seconds
           'oversampling': 10,       # Temporal oversampling factor
           'basis_functions': 1      # Number of HRF basis functions
       }
   )
   ```

## Output Files Structure

The package creates three HDF5 files in the output directory for each subject:

### 1. `data.h5`: Raw computational results
   * `/forward_fft`: Complex FFT result on regular grid
   * `/kspace_masks/{mask_id}`: Binary masks in k-space
   * `/inverse_maps/{mask_id}`: Complex inverse maps for each mask
   * `/gradient_maps/{mask_id}`: Gradient magnitude maps (on grid if skip_interpolation=False, otherwise just references to non-uniform data)
   * `/params`: Processing parameters and metadata

### 2. `analysis.h5`: Analysis results
   * `/magnitude/{mask_id}`: Magnitude values
   * `/phase/{mask_id}`: Phase angle values
   * `/local_variance/{mask_id}`: Local variance metrics
   * `/temporal_diff_magnitude/{mask_id}`: Temporal derivatives
   * `/temporal_diff_phase/{mask_id}`: Phase changes over time
   * `/summary`: Summary statistics for each metric

### 3. `enhanced.h5`: Enhanced feature results
   * `/spectral_slope`: Power law exponents
   * `/spectral_entropy`: Entropy of k-space distribution
   * `/anisotropy`: Directional preference metrics
   * `/higher_moments`: Skewness and kurtosis values
   * `/excitation`: Neural activity estimates
   * `/analytical_gradients/{mask_id}`: Analytically computed gradients
   * `/params`: Configuration parameters for enhanced features

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