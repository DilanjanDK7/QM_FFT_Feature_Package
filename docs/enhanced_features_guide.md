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

## Analytic Radial Gradient

The analytic radial gradient is computed directly in k-space using the property that a derivative in real space corresponds to a multiplication by frequency in k-space. This approach avoids the need for interpolation and numerical differentiation, resulting in:

1. **Speed Improvement**: Typically 2-5x faster than the traditional approach.
2. **Accuracy**: May provide more accurate gradients, especially at boundaries.

### Usage

```python
# Use analytical gradient method when computing gradient maps
map_builder.compute_gradient_maps(use_analytical_method=True)

# Or set it in the process_map method
map_builder.process_map(
    n_centers=2,
    radius=0.5,
    analyses_to_run=['magnitude', 'phase'],
    use_analytical_gradient=True
)
```

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

The "-3" term makes the kurtosis of a normal distribution equal to 0 (Fisher's definition).

- Positive kurtosis indicates a distribution with heavier tails than a normal distribution.
- Negative kurtosis indicates a distribution with lighter tails than a normal distribution.
- Zero kurtosis corresponds to a normal distribution.

### Usage

```python
# Compute higher-order moments
metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['higher_moments'])

# Access results
skewness = metrics['map_0']['skewness']
kurtosis = metrics['map_0']['kurtosis']
```

## HRF Deconvolution-Based Excitation Maps

HRF deconvolution aims to recover the underlying neural activity by removing the effect of the hemodynamic response function (HRF), which is a delayed and dispersed response to neural activity.

### Canonical HRF

The canonical HRF is modeled as the difference of two gamma functions:

```
h(t) = (t/d₁)ᵃ¹·exp(-(t-d₁)/b₁) - c·(t/d₂)ᵃ²·exp(-(t-d₂)/b₂)
```

This produces a characteristic shape with an initial peak followed by an undershoot.

### Deconvolution

Deconvolution is performed in the frequency domain:

```
E(f) = Y(f) / H(f)
```

where `Y(f)` is the Fourier transform of the measured signal, `H(f)` is the Fourier transform of the HRF, and `E(f)` is the Fourier transform of the estimated neural activity.

### Usage

```python
# Note: Requires at least 3 time points
metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['excitation'])

# Access results
excitation_map = metrics['excitation_map']
```

## Integration with the Processing Pipeline

All enhanced features can be included in the processing pipeline by specifying them in the `analyses_to_run` parameter:

```python
map_builder.process_map(
    n_centers=2,
    radius=0.5,
    analyses_to_run=[
        # Standard analyses
        'magnitude', 'phase', 'local_variance',
        # Enhanced analyses
        'spectral_slope', 'spectral_entropy', 'anisotropy',
        'higher_moments', 'excitation'
    ],
    use_analytical_gradient=True
)
```

## Output Files

Enhanced feature results are saved in the subject's `enhanced` directory:

- `spectral_slope.npy`: Spectral slope values for each transform.
- `spectral_entropy.npy`: Spectral entropy values for each transform.
- `anisotropy.npy`: K-space anisotropy values for each transform.
- `map_i_skewness.npy`: Skewness values for inverse map `i`.
- `map_i_kurtosis.npy`: Kurtosis values for inverse map `i`.
- `excitation_map.npy`: Excitation map computed via HRF deconvolution.
- `analytical_gradient_map_i.npy`: Gradient map computed analytically for mask `i`.
- `enhanced_metrics.h5`: All enhanced metrics in a single HDF5 file.

## Backward Compatibility

The enhanced features are strictly opt-in: all existing functionality works exactly as before unless `enable_enhanced_features=True` is specified. When analytical gradient computation is used, the results are also saved in the standard gradient map format for compatibility with existing code. 