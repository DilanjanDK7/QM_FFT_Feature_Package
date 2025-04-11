\
---
jupytext:
  formats: mystnb:myst
  text_representation:
    extension: .myst.md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Demo: Preprocessing Time Series and MapBuilder Workflow (Multi-Time Point)

## Overview

This notebook demonstrates the updated end-to-end workflow for processing time-series data using the `QM_FFT_Analysis` package, now optimized for **multiple time points**. We will:
1. Generate synthetic time-series data associated with spatial locations.
2. Apply the Hilbert transform and normalize the resulting complex wavefunctions **vectorially for multiple selected time points** using `get_normalized_wavefunctions_at_times`.
3. Use the stacked normalized wavefunctions as input `strengths` for the `MapBuilder`.
4. Compute the forward Fast Non-uniform Fourier Transform (FINUFFT) **simultaneously** for all time points using `MapBuilder` (`n_trans`).
5. Optionally normalize the resulting k-space power spectrum density **for each time point**.
6. Verify the outputs at each stage.

Refer to the updated [Workflow Documentation](QM_FFT_Analysis/docs/workflow_preprocessing_fft.md) for more detailed explanations.

## 1. Setup and Imports

First, let\'s import the necessary libraries and our custom functions. We also set up paths and basic logging.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Ensure the package directory is in the Python path
# Adjust the path if the notebook is run from a different location
package_dir = Path(\'.\').resolve() 
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

# Import our custom modules
from QM_FFT_Analysis.utils.preprocessing import get_normalized_wavefunctions_at_times
from QM_FFT_Analysis.utils.map_builder import MapBuilder

# Configure basic logging for feedback
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

# Define output directory for this demo
output_dir = Path(\'./tests/demo_output\')
output_dir.mkdir(parents=True, exist_ok=True)

print(f\"Imports successful.\")
print(f\"Package directory: {package_dir}\")
print(f\"Output will be saved in: {output_dir}\")
```

## 2. Generate Synthetic Test Data

We need some data to work with. Let\'s create:
*   Time series data for multiple sources.
*   **1D** spatial coordinates (x, y, z) for each source (required by updated `MapBuilder`).

```python
# Parameters for synthetic data
n_sources = 50
n_time_points = 200
sampling_freq = 100  # Hz (for context, not directly used in Hilbert/FFT normalization)
signal_freq_1 = 5    # Hz
signal_freq_2 = 15   # Hz

# Time vector
time_vec = np.arange(n_time_points) / sampling_freq

# Create synthetic time series data (e.g., sum of sinusoids with noise)
np.random.seed(42) # for reproducibility
time_series_data = np.zeros((n_sources, n_time_points))
for i in range(n_sources):
    amp1 = np.random.rand() * 2
    amp2 = np.random.rand() * 1.5
    phase1 = np.random.rand() * 2 * np.pi
    phase2 = np.random.rand() * 2 * np.pi
    noise = np.random.randn(n_time_points) * 0.2
    time_series_data[i, :] = (amp1 * np.sin(2 * np.pi * signal_freq_1 * time_vec + phase1) +
                              amp2 * np.sin(2 * np.pi * signal_freq_2 * time_vec + phase2) + 
                              noise)

# Create synthetic spatial coordinates (non-uniform)
# Coordinates must now be 1D arrays
x_coords = np.random.uniform(-np.pi, np.pi, n_sources)
y_coords = np.random.uniform(-np.pi, np.pi, n_sources)
z_coords = np.random.uniform(-np.pi, np.pi, n_sources)

print(f\"Generated time_series_data with shape: {time_series_data.shape}\")
print(f\"Generated 1D coordinates (x,y,z) with shape: {x_coords.shape}\")

# Visualize the time series for one source
plt.figure(figsize=(10, 3))
plt.plot(time_vec, time_series_data[0, :])
plt.title(\"Time Series for Source 0\")
plt.xlabel(\"Time (s)\")
plt.ylabel(\"Amplitude\")
plt.grid(True)
plt.show()
```

## 3. Preprocessing: Hilbert Transform and Multi-Time Point Wavefunction Normalization

Now, we apply the vectorized preprocessing step. We choose **multiple time points** and use `get_normalized_wavefunctions_at_times` to get the stack of complex, normalized wavefunctions.

```python
# Parameters for preprocessing
time_indices_to_analyze = [50, 100, 150] # Choose multiple time point indices
time_axis = 1               # Time is the second axis (index 1)
source_axis = 0             # Sources are the first axis (index 0)

# Perform preprocessing
normalized_wavefunctions_stack = get_normalized_wavefunctions_at_times(
    time_series_data=time_series_data,
    time_indices=time_indices_to_analyze,
    time_axis=time_axis,
    source_axis=source_axis
)

# --- Verification ---
print(f\"Preprocessing complete for time indices: {time_indices_to_analyze}\")
expected_shape = (len(time_indices_to_analyze), n_sources)
print(f\"Shape of normalized_wavefunctions_stack: {normalized_wavefunctions_stack.shape}\") # Should be (n_times, n_sources)
print(f\"Data type: {normalized_wavefunctions_stack.dtype}\") # Should be complex
assert normalized_wavefunctions_stack.shape == expected_shape

# Check normalization for each time point (each row)
total_probabilities = np.sum(np.abs(normalized_wavefunctions_stack)**2, axis=1) # Sum over sources (axis 1)
print(f\"Sum of probability density (|wf|^2) across sources for each time point: {total_probabilities}\")
assert np.allclose(total_probabilities, 1.0), \"Wavefunction normalization failed for one or more time points!\"

print(\"Verification successful.\")
```
The output `normalized_wavefunctions_stack` is a 2D complex array `(n_times, n_sources)` representing the normalized state across all sources for each chosen time point. This stack will serve as the `strengths` input for the `MapBuilder`.

## 4. Map Building: FINUFFT and Optional Normalization

We initialize the `MapBuilder` with the **1D** spatial coordinates and the **stacked** `normalized_wavefunctions_stack` as the `strengths`. `MapBuilder` will detect `n_trans` from the input shape. We\'ll also enable the FFT result normalization (`normalize_fft_result=True`).

```python
# MapBuilder parameters
subject_id = \"DemoSubject\"
finufft_eps = 1e-6
# Use complex64 as the preprocessing function outputs complex64
finufft_dtype = \'complex64\' 
normalize_fft = True

# Initialize MapBuilder
builder = MapBuilder(
    subject_id=subject_id, 
    output_dir=output_dir, # Use the specific demo output dir
    x=x_coords, 
    y=y_coords, 
    z=z_coords, 
    strengths=normalized_wavefunctions_stack, # Use the preprocessed stack
    eps=finufft_eps, 
    dtype=finufft_dtype, 
    normalize_fft_result=normalize_fft 
)

print(f\"MapBuilder initialized for subject: {subject_id}\")
print(f\"Detected n_trans (number of time points): {builder.n_trans}\")
print(f\"Target grid dimensions (nx, ny, nz): ({builder.nx}, {builder.ny}, {builder.nz})\")
print(f\"FFT result normalization enabled: {builder.normalize_fft_result}\")
```

Now, compute the forward FFT. This will generate the raw FFT result (`forward_fft.npy`) and, because we set `normalize_fft_result=True`, also the normalized power spectrum density (`fft_prob_density.npy`).

```python
# Compute the forward FFT
builder.compute_forward_fft()

# --- Verification ---
print(f\"Forward FFT computation complete.\")

# Check if output files exist
raw_fft_file = builder.data_dir / \"forward_fft.npy\"
norm_fft_file = builder.data_dir / \"fft_prob_density.npy\"

assert raw_fft_file.exists(), f\"Raw FFT file not found: {raw_fft_file}\"
print(f\"Raw FFT result saved: {raw_fft_file}\")

if normalize_fft:
    assert norm_fft_file.exists(), f\"Normalized FFT file not found: {norm_fft_file}\"
    print(f\"Normalized FFT probability density saved: {norm_fft_file}\")
    
    # Load the normalized result and verify its sum
    fft_prob_density = np.load(norm_fft_file)

    # Verify shape (n_trans, nx, ny, nz)
    expected_fft_shape = (builder.n_trans, builder.nx, builder.ny, builder.nz)
    print(f\"Shape of loaded fft_prob_density: {fft_prob_density.shape}\")
    assert fft_prob_density.shape == expected_fft_shape

    # Verify normalization for each transform (sum over spatial axes)
    total_k_prob_per_trans = np.sum(fft_prob_density, axis=(1, 2, 3))
    print(f\"Sum of FFT probability density across k-space for each transform: {total_k_prob_per_trans}\")
    assert np.allclose(total_k_prob_per_trans, 1.0), \"FFT Probability Density normalization failed!\"

# Check the shapes of the results stored in the builder object
expected_fft_shape = (builder.n_trans, builder.nx, builder.ny, builder.nz)
print(f\"Shape of builder.fft_result (raw complex FFT): {builder.fft_result.shape}\")
assert builder.fft_result.shape == expected_fft_shape
if normalize_fft:
    print(f\"Shape of builder.fft_prob_density (real probability): {builder.fft_prob_density.shape}\")
    assert builder.fft_prob_density.shape == expected_fft_shape

print(\"Verification successful.\")
```

## 5. Subsequent Steps (Optional)

From here, you can proceed with the rest of the `MapBuilder` pipeline, such as generating k-space masks and computing inverse maps and gradients. These steps will operate on all `n_trans` time points simultaneously. The results (e.g., `builder.inverse_maps`, `builder.gradient_maps`) will be lists (over masks), where each element is an array with the first dimension corresponding to the time points (`n_trans`).

```python
# Example of generating masks and inverse maps (optional)
# builder.generate_kspace_masks(n_centers=3, radius=1.0)
# builder.compute_inverse_maps()
# builder.compute_gradient_maps()
# print(f\"Generated {len(builder.inverse_maps)} inverse maps and {len(builder.gradient_maps)} gradient maps.\")

# Accessing results for a specific mask and time point:
# if builder.gradient_maps:
#   mask_index = 0
#   time_point_index = 1 # Corresponds to the second time index in time_indices_to_analyze
#   gradient_map_slice = builder.gradient_maps[mask_index][time_point_index, :, :, :]
#   print(f"Shape of single gradient map slice: {gradient_map_slice.shape}") # Should be (nx, ny, nz)

#   # Visualize this 3D slice
#   # builder.generate_volume_plot(gradient_map_slice, f\"gradient_map_{mask_index}_t{time_indices_to_analyze[time_point_index]}.html\")
```

## Conclusion

This notebook demonstrated the **updated multi-time point** workflow from raw time-series data to a normalized k-space representation using the `preprocessing` and `MapBuilder` components. Key steps included Hilbert transform, **vectorized** wavefunction normalization for multiple time points, **simultaneous** FINUFFT (`n_trans`) using `MapBuilder`, and optional normalization of the k-space power spectrum per time point. The verification steps confirmed the expected shapes, data types, and normalization properties. Output files (now often 4D) were generated in the `./tests/demo_output/DemoSubject` directory. 