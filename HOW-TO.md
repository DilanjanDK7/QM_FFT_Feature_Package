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

### Memory and Storage Requirements

| Dataset Size | RAM Usage | Storage Required |
|--------------|-----------|------------------|
| Small (1K pts, 5 times) | ~100MB | ~2MB |
| Medium (5K pts, 10 times) | ~500MB | ~18MB |
| Large (10K pts, 15 times) | ~1GB | ~200MB |
| Extreme (22K pts, 15+ times) | ~1-2GB | ~400MB+ |

### Performance Optimizations

1. **Analytical Gradient Method**:
   * Enable with `use_analytical_gradient=True` in `process_map()`
   * Speedup factors:
     - 1.4x for small datasets (1,000 points)
     - 2.3x for medium datasets (5,000 points)
     - 1.7x for large datasets (10,000+ points)

2. **Enhanced-Only Processing**:
   * Use `compute_enhanced_metrics()` for spectral metrics without full analysis
   * Speedup factors:
     - 5.9x for small datasets (1,000 points, 5 times)
     - 7.8x for medium datasets (5,000 points, 10 times)
     - 8.6x for large datasets (10,000+ points, 15 times)

3. **Other Optimization Tips**:
   * Enable HDF5 compression with `compression='gzip'`
   * For large batch jobs, use logging with appropriate verbosity
   * Use the smallest necessary grid dimensions for k-space

## Benchmarks

To evaluate performance on your system, use the benchmark scripts in the `benchmarks/` directory:

```bash
# Compare enhanced-only vs full pipeline
python benchmarks/benchmark_enhanced_only.py

# Evaluate analytical gradient performance
python benchmarks/benchmark_analytical_gradient.py

# Test with extremely large datasets
python benchmarks/benchmark_extreme_gradient.py
```

See `benchmarks/README.md` for details on all available benchmark scripts.

## Visualization Examples

```python
import matplotlib.pyplot as plt
import h5py

# Load previously computed results
with h5py.File('output/subject_01/analysis.h5', 'r') as f:
    magnitude = f['/magnitude/mask_0'][:]
    phase = f['/phase/mask_0'][:]

with h5py.File('output/subject_01/enhanced.h5', 'r') as f:
    spectral_slope = f['/spectral_slope'][:]
    anisotropy = f['/anisotropy'][:]

# Create visualization (simplified example)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(x, y, c=magnitude[0], cmap='viridis')
axes[0, 0].set_title('Magnitude (t=0)')

axes[0, 1].scatter(x, y, c=phase[0], cmap='hsv')
axes[0, 1].set_title('Phase (t=0)')

axes[1, 0].plot(spectral_slope)
axes[1, 0].set_title('Spectral Slope Over Time')

axes[1, 1].plot(anisotropy)
axes[1, 1].set_title('Anisotropy Over Time')

plt.tight_layout()
plt.savefig('output/subject_01/summary_plot.png')
plt.show()
```

## Troubleshooting

Common issues and solutions:

* **Memory errors**: Reduce grid dimensions or process data in smaller batches
* **Performance issues**: Enable analytical gradient method and use enhanced-only processing when possible
* **HDF5 errors**: Ensure proper file closing with context managers or explicitly close files
* **FINUFFT errors**: Verify input data is non-uniform and within expected range

For more help, please check the benchmarks or open an issue in the repository. 