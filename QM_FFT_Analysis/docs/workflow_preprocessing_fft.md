# Workflow Documentation: Time Series to Normalized FFT (Multi-Time Point)

## Overview
This document outlines the workflow for processing time-series data associated with spatial locations, applying a Hilbert transform, selecting **multiple time points**, normalizing the *complex wavefunction* for each point, using the `MapBuilder` to compute the FINUFFT **simultaneously** for all selected time points (`n_trans`), and optionally normalizing the resulting k-space data for each time point.

## Preprocessing (Using `QM_FFT_Analysis/utils/preprocessing.py`)

**Goal:** Prepare the complex `strengths` input (shape `(n_times, n_sources)`) for the `MapBuilder` from multi-dimensional time-series data by normalizing the wavefunction **vectorially** across multiple specified time points.

1.  **Input Data:** Start with a NumPy array representing the time-series data, e.g., `(n_sources, n_time_points)`, and corresponding **1D** spatial coordinates `x, y, z` for each source (`n_sources`).
2.  **Identify Axes:** Determine the index of the axis representing time (`time_axis`) and the axis representing sources (`source_axis`).
3.  **Select Time Points:** Choose the specific time indices (a list or array `time_indices`) you want to analyze.
4.  **Function Call:** Use the `get_normalized_wavefunctions_at_times` function (plural):
    ```python
    from QM_FFT_Analysis.utils.preprocessing import get_normalized_wavefunctions_at_times

    # Example:
    # time_series_data: shape (10, 100) -> 10 sources, 100 time points
    # time_indices = [50, 60, 70]
    # time_axis = 1
    # source_axis = 0

    normalized_wavefunctions_stack = get_normalized_wavefunctions_at_times(
        time_series_data=your_time_data, 
        time_indices=[50, 60, 70], 
        time_axis=1, 
        source_axis=0
    )
    # Result `normalized_wavefunctions_stack` will have shape (3, 10) and be complex
    ```
5.  **Process within `get_normalized_wavefunctions_at_times`:**
    *   Applies the Hilbert transform along the `time_axis` once.
    *   Selects the complex wavefunction data for all specified `time_indices` using vectorized indexing.
    *   Calculates the probability density (`abs(...)**2`) for the selected time points.
    *   Sums the density across the `source_axis` for each time point independently to get the total probability per time point.
    *   Calculates the normalization factor for each time point (`sqrt(total_probability)`).
    *   Divides the selected complex wavefunction data by the corresponding normalization factors using broadcasting.
6.  **Output:** The function returns the `normalized_wavefunctions_stack` array (shape `(n_selected_times, n_sources)`). Each row represents a normalized complex wavefunction for one time point. This 2D array is passed as the `strengths` argument to `MapBuilder`.

## Map Building (Using `QM_FFT_Analysis/utils/map_builder.py`)

**Goal:** Compute the non-uniform FFT **simultaneously** for multiple time points (`n_trans`) using the stacked normalized wavefunctions as input, and potentially normalize the FFT result for each transform.

1.  **Initialization:** Instantiate the `MapBuilder` class, providing the preprocessed `normalized_wavefunctions_stack` (shape `(n_times, n_sources)`) as the `strengths` argument. `MapBuilder` automatically detects `n_trans` from the first dimension of `strengths`. Ensure the `dtype` matches the preprocessing output (e.g., `complex64`). Ensure `x, y, z` are 1D arrays of length `n_sources`. Optionally, set `normalize_fft_result=True`.
    ```python
    from QM_FFT_Analysis.utils.map_builder import MapBuilder

    # x, y, z are 1D coordinates corresponding to sources
    builder = MapBuilder(
        subject_id='Subject01', 
        output_dir='./output', 
        x=x_coords, 
        y=y_coords, 
        z=z_coords, 
        strengths=normalized_wavefunctions_stack, # Use the (n_times, n_sources) stack
        eps=1e-6, 
        dtype='complex64', # Ensure this matches the output of preprocessing
        normalize_fft_result=True # Optional: Normalize FFT result
    )
    # builder.n_trans will be equal to len(time_indices)
    ```
2.  **Compute Forward FFT:** Call the `compute_forward_fft` method.
    ```python
    builder.compute_forward_fft()
    ```
3.  **Process within `compute_forward_fft`:**
    *   Executes the FINUFFT Type 1 transform using the input `strengths` (shape `(n_trans, N)`).
    *   Reshapes the result into a 4D k-space grid (`builder.fft_result`, shape `(n_trans, nx, ny, nz)`).
    *   Saves the raw 4D FFT result to `forward_fft.npy`.
    *   **If `normalize_fft_result` is `True`:**
        *   Calculates the power spectrum density (`abs(fft_result)**2`, shape `(n_trans, nx, ny, nz)`).
        *   Computes the total power for each transform by summing over spatial axes `(1, 2, 3)`.
        *   Divides the power spectrum density by the corresponding total power (using broadcasting) to get normalized probability densities (`builder.fft_prob_density`). Each 3D slice along the first axis sums to 1.
        *   Saves the 4D normalized density to `fft_prob_density.npy`.
4.  **Subsequent Steps:** Proceed with generating k-space masks (`generate_kspace_masks`, masks are 3D), computing inverse maps (`compute_inverse_maps`, output shape `(n_masks, n_trans, n_points)`), and gradient maps (`compute_gradient_maps`, output shape `(n_masks, n_trans, nx, ny, nz)`) using the `builder` object. All computations involving transforms are now vectorized across the `n_trans` dimension.

## Summary
This workflow allows for **efficiently analyzing** the spatial frequency representation of time-series data at **multiple specific moments simultaneously**. It involves:
- Preprocessing to isolate multiple time points and normalize the complex wavefunction for each point using vectorized operations.
- Using `MapBuilder` to simultaneously transform these normalized complex wavefunctions into k-space using FINUFFT's `n_trans` capability.
Optionally normalizing the k-space power spectrum for each time point to represent probability distributions. 