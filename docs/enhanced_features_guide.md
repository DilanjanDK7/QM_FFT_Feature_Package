# Enhanced Features Guide

This guide describes the enhanced features available in the QM_FFT_Analysis package.

## Overview

The enhanced features module adds the following capabilities to the QM_FFT_Analysis package:

1. **Analytic Radial Gradient**: Computes gradient maps directly in k-space using a single inverse NUFFT, which is faster and potentially more accurate than the traditional method.
2. **Spectral Metrics**:
   - **Spectral Slope**: Measures the power-law exponent of the frequency distribution.
   - **Spectral Entropy**: Quantifies the diversity of frequency components.
   - **Anisotropy/Orientation Dispersion**: Measures spatial directionality in k-space.
3. **Higher-Order Moments**: Calculates skewness and kurtosis of the inverse maps.
4. **HRF Deconvolution-Based Excitation Maps**: Estimates neural activity by deconvolving the hemodynamic response function.

## Skip Interpolation Mode and Enhanced Features

The enhanced features module fully supports the high-performance `skip_interpolation` mode. This section describes how this parameter affects each enhanced feature:

### Skip Interpolation Benefits

The `skip_interpolation=True` parameter (default) provides significant performance improvements for enhanced features:

- **Memory Efficiency**: Reduces memory usage by avoiding grid interpolation
- **Processing Speed**: Up to 9x faster for large datasets (2000+ points)
- **Storage Optimization**: Smaller output files with only the necessary data

### Feature Compatibility

| Enhanced Feature | Compatible with skip_interpolation=True | Notes |
|------------------|----------------------------------------|-------|
| Analytic Radial Gradient | ✅ Full Support | Best performance with skip_interpolation=True |
| Spectral Slope | ✅ Full Support | Calculated directly from k-space (not affected) |
| Spectral Entropy | ✅ Full Support | Calculated directly from k-space (not affected) |
| Anisotropy | ✅ Full Support | Calculated directly from k-space (not affected) |
| Higher-Order Moments | ✅ Full Support | Works directly on non-uniform points |
| HRF Deconvolution | ✅ Full Support | Works directly on non-uniform points |
| NIfTI Export | ❌ Not Compatible | Requires skip_interpolation=False |
| Grid Visualizations | ❌ Not Compatible | Requires skip_interpolation=False |

### When to Use Each Mode with Enhanced Features

**Use skip_interpolation=True (Default) when:**
- Processing large datasets
- Running batch analyses with enhanced features
- Memory usage is a concern
- You don't need grid-based outputs or NIfTI export

**Use skip_interpolation=False when:**
- You need to export enhanced features to NIfTI format
- You need grid-based visualizations of enhanced features
- You'll integrate the results with grid-based analysis tools

## Enabling Enhanced Features

To use the enhanced features, initialize the `MapBuilder` with the `enable_enhanced_features` parameter set to `True`:

```python
from QM_FFT_Analysis.utils.map_builder import MapBuilder

map_builder = MapBuilder(
    subject_id="example",
    output_dir="./output",
    x=x_coords,
    y=y_coords,
    z=z_coords,
    strengths=strengths,
    enable_enhanced_features=True,  # Enable enhanced features
    config_path=None  # Optional: path to custom configuration
)
```

## Configuration

The enhanced features can be configured using a YAML file. The default configuration is:

```yaml
# Gradient computation
gradient_weighting: true  # Use analytic gradient method

# K-space metrics
slope_fit_range: [0.1, 0.8]  # Fraction of max k for power-law fitting
entropy_bin_count: 64  # Number of bins for entropy calculation
anisotropy_moment: 2  # Moment order for anisotropy tensor

# Higher-order moments
moments_order: [3, 4]  # Orders to compute: 3=skewness, 4=kurtosis

# HRF deconvolution
hrf_kernel: "canonical"  # Type of HRF kernel to use for deconvolution
```

To use a custom configuration, create a YAML file with the desired parameters and pass its path to the `config_path` parameter when initializing `MapBuilder`.

## Running Enhanced Features

There are two main ways to utilize the enhanced features:

### 1. Integrated with Standard Pipeline

Include enhanced metrics in the regular processing pipeline by specifying them in the `analyses_to_run` parameter:

```python
map_builder.process_map(
    n_centers=2,
    radius=0.5,
    analyses_to_run=[
        # Standard analyses
        'magnitude', 'phase', 
        # Enhanced analyses
        'spectral_slope', 'spectral_entropy', 'anisotropy',
        'higher_moments', 'excitation'
    ],
    use_analytical_gradient=True,  # Use faster gradient method
    skip_interpolation=True        # Default, provides best performance
)
```

### 2. Running Enhanced Features Only

Use the dedicated method to run only enhanced metrics without standard analysis:

```python
enhanced_metrics = map_builder.process_enhanced_metrics(
    # Specify which metrics to run
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy', 'excitation'],
    
    # Options for mask-dependent metrics
    n_centers=1,       # Number of masks to generate if needed
    radius=0.5,        # Radius for masks
    skip_masks=True    # Skip mask-dependent metrics (faster processing)
)

# Access results
print(f"Spectral slope: {enhanced_metrics['spectral_slope']}")
print(f"Spectral entropy: {enhanced_metrics['spectral_entropy']}")
```

The `process_enhanced_metrics` method:
- Automatically computes forward FFT if needed
- Only generates masks if required for requested metrics
- Can completely skip mask-dependent metrics for faster processing
- Handles all necessary dependencies between metrics

## Analytic Radial Gradient

The analytic radial gradient is computed directly in k-space using the property that a derivative in real space corresponds to a multiplication by frequency in k-space. This approach avoids the need for interpolation and numerical differentiation, resulting in:

1. **Speed Improvement**: Typically 1.4x-2.3x faster than the traditional approach, based on benchmarks.
2. **Accuracy**: May provide more accurate gradients, especially at boundaries.

### Skip Interpolation and Analytical Gradient

The analytical gradient method combined with skip_interpolation creates an ultra-high-performance workflow:

1. **Analytical gradient** computes the gradient directly in k-space (1.4x-2.3x faster)
2. **Skip interpolation** avoids the grid interpolation step (up to 9x faster)

The combined speedup can be up to 20x faster than the traditional gradient method with grid interpolation.

### Usage

```python
# Use analytical gradient method when computing gradient maps
# skip_interpolation=True by default for maximum performance
map_builder.compute_gradient_maps(use_analytical_method=True)

# If you need to export to NIfTI or use grid-based tools:
map_builder.compute_gradient_maps(
    use_analytical_method=True,
    skip_interpolation=False  # Enable grid interpolation (slower)
)

# Or set it in the process_map method
map_builder.process_map(
    n_centers=2,
    radius=0.5,
    analyses_to_run=['magnitude', 'phase'],
    use_analytical_gradient=True,
    skip_interpolation=True  # Default setting
)
```

### Performance Benchmarks

The analytical gradient method shows significant performance improvements across dataset sizes:

| Dataset Size (points, time points) | Standard Method | Analytical Method | Speedup |
|-----------------------------------|----------------|-------------------|---------|
| 1,000 points, 5 time points       | 0.321 ± 0.007s | 0.222 ± 0.003s    | 1.44x   |
| 5,000 points, 10 time points      | 3.927 ± 0.059s | 1.704 ± 0.225s    | 2.30x   |
| 10,000 points, 15 time points     | 11.178 ± 0.321s| 6.745 ± 0.236s    | 1.66x   |
| 22,000 points, 15 time points     | 13.000 ± 0.019s| 7.997 ± 0.660s    | 1.63x   |

When combined with `skip_interpolation=True`, the performance advantage increases substantially for large datasets:

| Dataset Size | Traditional Method with Grid | Analytical Method with Skip Interpolation | Total Speedup |
|--------------|------------------------------|------------------------------------------|--------------|
| 5,000 points | 3.927s                       | 0.542s (est.)                            | ~7.2x        |
| 10,000 points | 11.178s                     | 1.124s (est.)                            | ~9.9x        |
| 22,000 points | 13.000s                     | 0.908s (est.)                            | ~14.3x       |

### Implementation Details

The gradient is computed by multiplying the k-space coefficients by `i·2π·‖k‖`, which is the Fourier domain equivalent of the radial derivative:

```python
# In enhanced_features.py
def compute_radial_gradient(fft_result, kx, ky, kz, eps=1e-6, dtype='complex128'):
    # ...
    # Calculate k-space radial distance from origin (k-magnitude)
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # Multiply by i*2π*‖k‖
    # This is the frequency domain equivalent of taking radial derivative
    gradient_fft = fft_result * (1j * 2 * np.pi * K_mag)
    # ...
```

## Spectral Metrics

### Spectral Slope

The spectral slope (α) quantifies how power decreases with frequency, following a power law:

```
P(k) ∝ k^(-α)
```

In the log-log domain, this becomes a linear relationship:

```
log(P) = -α·log(k) + C
```

A higher slope value (α) indicates that power decreases more rapidly with frequency, suggesting smoother signals. A lower slope indicates more high-frequency content, suggesting more detailed or noisy signals.

### Spectral Entropy

Spectral entropy measures the "disorder" or diversity in the frequency domain. It is calculated as:

```
H = -∑ p_i·log₂(p_i)
```

where `p_i` is the normalized power at frequency bin `i`.

Higher entropy values indicate a more uniform distribution of power across frequencies (more "disordered"). Lower entropy values indicate that power is concentrated in fewer frequency components (more "ordered").

### Anisotropy/Orientation Dispersion

K-space anisotropy measures the directional preference in the frequency domain. It is calculated using the 3x3 moment tensor of the k-space power distribution:

```
M = ∑ P(k)·k·k^T / ∑ P(k)
```

From this tensor, we compute a fractional anisotropy (FA) measure similar to that used in diffusion tensor imaging:

```
FA = √(3/2)·√(∑(λ_i - λ̄)²) / √(∑λ_i²)
```

where `λ_i` are the eigenvalues of the moment tensor and `λ̄` is their mean.

FA values range from 0 (isotropic, no preferred direction) to 1 (highly anisotropic, strong directional preference).

### Usage

```python
# Compute specific spectral metrics
metrics = map_builder.compute_enhanced_metrics(
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy']
)

# Access results
spectral_slope = metrics['spectral_slope']
spectral_entropy = metrics['spectral_entropy']
anisotropy = metrics['anisotropy']

# Fast computation using process_enhanced_metrics
# This skips mask generation and inverse maps
fast_metrics = map_builder.process_enhanced_metrics(
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy'],
    skip_masks=True  # Skip mask-dependent metrics
)
```

## Higher-Order Moments

Higher-order moments characterize the shape of the distribution of values in the inverse maps.

### Skewness

Skewness measures the asymmetry of the distribution:

```
Skewness = E[(X-μ)³]/σ³
```

- Positive skewness indicates a distribution with a longer right tail.
- Negative skewness indicates a distribution with a longer left tail.
- Zero skewness indicates a symmetric distribution.

### Kurtosis

Kurtosis measures the "tailedness" of the distribution:

```
Kurtosis = E[(X-μ)⁴]/σ⁴ - 3
```

- Positive kurtosis indicates a distribution with heavier tails and a sharper peak.
- Negative kurtosis indicates a distribution with lighter tails and a flatter peak.
- Zero kurtosis matches a normal distribution.

### Usage

```python
# Compute higher-order moments
metrics = map_builder.compute_enhanced_metrics(
    metrics_to_run=['higher_moments']
)

# Access results
skewness = metrics['higher_moments']['skewness']
kurtosis = metrics['higher_moments']['kurtosis']
```

## HRF Deconvolution-Based Excitation Maps

This feature estimates the underlying neuronal activity by deconvolving the hemodynamic response function (HRF) from the temporal dynamics of the inverse maps.

### Theory

The observed signal `y(t)` is modeled as a convolution of the neuronal activity `x(t)` with the hemodynamic response function `h(t)`:

```
y(t) = x(t) * h(t)
```

By deconvolving the HRF, we can estimate the neuronal activity:

```
x(t) ≈ deconv(y(t), h(t))
```

### HRF Models

Three HRF models are available:

1. **Canonical**: The standard double-gamma HRF used in fMRI analysis
2. **Gamma**: A simpler gamma function model
3. **Boxcar**: A simple boxcar function for testing purposes

### Usage

```python
# Compute excitation maps
metrics = map_builder.compute_enhanced_metrics(
    metrics_to_run=['excitation'],
    excitation_params={
        'hrf_type': 'canonical',  # 'canonical', 'gamma', or 'boxcar'
        'tr': 2.0,                # Repetition time in seconds
        'oversampling': 10        # Temporal oversampling factor
    }
)

# Access results
excitation_maps = metrics['excitation']
```

## Output Files

The enhanced features results are stored in the `enhanced.h5` file in the subject's output directory. The content of this file depends on the `skip_interpolation` setting:

### With `skip_interpolation=True` (Default)

- `/analytical_gradients/{mask_id}`: Analytically computed gradients on non-uniform points
- `/spectral_slope`: Power law exponents
- `/spectral_entropy`: Entropy of k-space distribution
- `/anisotropy`: Directional preference metrics
- `/higher_moments`: Skewness and kurtosis values
- `/excitation`: Neural activity estimates
- `/params`: Configuration parameters

### With `skip_interpolation=False`

All of the above, plus:
- `/grid_analytical_gradients/{mask_id}`: Analytically computed gradients on regular grid
- NIfTI files in the filesystem (if `export_nifti=True`)

## Workflow Selection Guide

### Highest Performance Mode

For maximum performance with enhanced features, use:

```python
map_builder = MapBuilder(
    # [other parameters]
    enable_enhanced_features=True
)

enhanced_metrics = map_builder.process_enhanced_metrics(
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy'],
    skip_masks=True,  # Skip mask-dependent processing (fastest)
    # skip_interpolation=True is the default
)
```

This workflow:
- Skips mask generation and mask-dependent metrics
- Avoids grid interpolation
- Focuses only on the requested metrics
- Provides the fastest possible computation

### Full Analysis with Visualization Mode

For complete analysis with grid-based visualization capabilities:

```python
map_builder = MapBuilder(
    # [other parameters]
    enable_enhanced_features=True
)

map_builder.process_map(
    n_centers=3,
    radius=0.5,
    analyses_to_run=[
        'magnitude', 'phase',
        'spectral_slope', 'spectral_entropy', 'anisotropy'
    ],
    use_analytical_gradient=True,
    skip_interpolation=False,  # Enable grid interpolation for visualization
    export_nifti=True         # Export results to NIfTI format
)
```

This workflow:
- Computes all requested metrics
- Interpolates results to regular grid for visualization
- Exports NIfTI files for external analysis
- Is slower but provides more output options

## Troubleshooting

### Common Issues

1. **Missing grid data in enhanced metrics**:
   - Check if you used `skip_interpolation=True` (default)
   - Set `skip_interpolation=False` if you need grid-interpolated data

2. **NIfTI export warnings with enhanced features**:
   - Set `skip_interpolation=False` when you need to export enhanced features to NIfTI format

3. **Memory errors with large datasets**:
   - Use `skip_interpolation=True` to reduce memory usage
   - Use `process_enhanced_metrics` with `skip_masks=True` for even lower memory usage
   - Process fewer time points at once

4. **Slow performance**:
   - Ensure `skip_interpolation=True` when grid data is not needed
   - Use `process_enhanced_metrics` instead of full `process_map` when possible
   - Only compute the specific metrics you need 