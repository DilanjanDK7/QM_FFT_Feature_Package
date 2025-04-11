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

# Comprehensive Guide: QM-Inspired FINUFFT Analysis for Brain Mapping

## 1. Introduction: Bridging Quantum Concepts and Brain Dynamics

The `QM_FFT_Analysis` package offers a novel computational pipeline designed to analyze spatio-temporal patterns in neuroimaging data, drawing inspiration from quantum mechanical principles for signal representation and leveraging the power of the Non-uniform Fast Fourier Transform (NUFFT).

**Core Purpose:** The primary goal is to transform non-uniformly distributed brain activity signals (e.g., from source-localized EEG or MEG) into a spatial frequency domain (k-space), analyze specific frequency components, transform them back to the source space, and compute spatial gradients. This process aims to generate maps that may offer insights into the brain\'s functional organization and dynamics, particularly concerning neural **excitability**.

## 2. Background: Excitability Mapping and the QM-FFT Approach

### Brain Excitability
Brain excitability refers to the readiness of neurons or neural populations to fire or respond to stimulation. It\'s a fundamental property underlying brain function and plasticity. Aberrant excitability is implicated in various neurological and psychiatric disorders (e.g., epilepsy, migraine, depression).

Mapping brain excitability typically involves techniques like Transcranial Magnetic Stimulation (TMS) combined with EEG or fMRI, or analyzing specific features of electrophysiological signals (e.g., evoked potentials, oscillation power). These methods provide valuable but often spatially or temporally limited views.

### The Rationale for QM-FFT
This package explores a different approach by hypothesizing that rapid spatial changes in specific spatio-temporal patterns of brain activity might correlate with underlying excitability states. The pipeline uses the following concepts:

1.  **Hilbert Transform:** To represent the time-varying brain signal as a complex analytic signal (often termed a "wavefunction" in this context), capturing both amplitude and phase information.
2.  **Quantum-Inspired Normalization:** At a specific time point, the complex analytic signal across different brain sources is normalized such that the sum of its squared magnitudes (probability density) equals one. This treats the instantaneous brain state as a normalized "wavefunction" in the source space.
3.  **FINUFFT (Forward):** Transforms this normalized, complex "wavefunction" from the non-uniform source locations into a uniform grid in spatial frequency (k-space).
4.  **K-Space Masking:** Isolates specific spatial frequency components by applying masks in k-space. This allows focusing on patterns of a particular spatial scale.
5.  **FINUFFT (Inverse):** Transforms the masked k-space data back to the original source space, reconstructing the spatial pattern associated with the selected frequency components.
6.  **Gradient Calculation:** Computes the spatial gradient magnitude of the reconstructed pattern. Regions with high gradients indicate rapid changes in the reconstructed signal\'s amplitude across space.

**The Hypothesis:** The resulting **gradient maps** are proposed as potential proxies for excitability maps. The idea is that regions exhibiting sharp spatial changes in the reconstructed neural activity patterns (high gradient) might correspond to areas with high functional contrast or potential boundaries between different activity states, possibly reflecting transitions in local excitability.

## 3. Pipeline Overview

The process involves two main stages, optimized for processing multiple time points simultaneously:

1.  **Preprocessing:** Raw time-series data is processed to extract a stack of normalized complex "wavefunctions", one for each specified time point.
2.  **Map Building:** This stack of wavefunctions (shape `(n_times, n_sources)`) is fed into the `MapBuilder`, which performs the FINUFFT transforms (`n_trans = n_times`), masking, and gradient calculations in a vectorized manner across time points.

```mermaid
graph LR
    A[Time Series Data + 1D Coordinates] --> B(Preprocessing: Hilbert + QM Norm @ multiple times t1..tn);
    B --> C{MapBuilder};
    C -- Strengths Input (n_times, n_sources) --> D(Forward FINUFFT n_trans=n_times);
    D --> E[K-Space Data (n_times, nx, ny, nz)];
    E --> F(Apply K-Space Masks);
    F --> G[Masked K-Space (n_times, nx, ny, nz)];
    G --> H(Inverse FINUFFT n_trans=n_times);
    H --> I[Reconstructed Maps (n_masks, n_times, n_points)];
    I --> J(Compute Spatial Gradients);
    J --> K[Gradient Maps (n_masks, n_times, nx, ny, nz)];
```

## 4. Detailed Workflow & Functioning

This workflow utilizes the functions and classes within the `QM_FFT_Analysis` package.

**(Step 1) Preprocessing (`QM_FFT_Analysis.utils.preprocessing`)**

*   **Input:**
    *   `time_series_data`: A NumPy array (e.g., `n_sources x n_timepoints`).
    *   `x, y, z`: 1D coordinates corresponding to each source (length `n_sources`).
    *   `time_indices`: A list or 1D array of the specific time points to analyze.
    *   `time_axis`, `source_axis`: Indices specifying the time and source dimensions.
*   **Function:** `get_normalized_wavefunctions_at_times()`
*   **Process:**
    1.  Applies the Hilbert transform (`scipy.signal.hilbert`) along the `time_axis` once.
    2.  Selects the complex signal slices for all specified `time_indices` using vectorized indexing.
    3.  Calculates the probability density (squared magnitude) for all selected time points.
    4.  Sums the density across the `source_axis` independently for each time point.
    5.  Calculates the normalization factor for each time point (`sqrt(total_probability)`).
    6.  Divides the complex signal slices by their corresponding normalization factors (vectorized).
*   **Output:**
    *   `normalized_wavefunctions_stack`: A complex NumPy array (shape `(n_selected_times, n_sources)`), ready as input for `MapBuilder`.

**(Step 2) Map Building (`QM_FFT_Analysis.utils.map_builder.MapBuilder`)**

*   **Initialization:**
    *   Instantiate `MapBuilder` with `subject_id`, `output_dir`, 1D coordinates (`x, y, z`), and the `normalized_wavefunctions_stack` as the `strengths` argument.
    *   `n_trans` is automatically inferred from `strengths.shape[0]`.
    *   Crucially, set the `dtype` parameter (e.g., `'complex64'`) to match the output of the preprocessing step.
    *   Optionally set `normalize_fft_result=True` to normalize the k-space power spectrum density for each transform.
*   **Core Methods:**
    1.  `compute_forward_fft()`: Executes the FINUFFT Type 1 transform simultaneously for all `n_trans` inputs. Stores the 4D result `(n_trans, nx, ny, nz)` in `self.fft_result`. Optionally normalizes and saves the 4D power spectrum density `self.fft_prob_density`.
    2.  `generate_kspace_masks(n_centers, radius)`: Creates 3D spherical boolean masks in k-space. Stores masks in `self.kspace_masks`.
    3.  `compute_inverse_maps()`: Applies each 3D mask to the 4D `self.fft_result` (using broadcasting) and performs the inverse FINUFFT (Type 2) simultaneously for all `n_trans`. Stores the reconstructed complex maps (shape `(n_trans, n_points)`) in `self.inverse_maps` (list over masks).
    4.  `compute_gradient_maps()`: Reshapes the inverse maps to an assumed grid `(n_trans, nx, ny, nz)`, calculates the spatial gradient magnitude for each transform along spatial axes. Stores 4D results `(n_trans, nx, ny, nz)` in `self.gradient_maps` (list over masks).
    5.  `process_map(...)`: A convenience method to run steps 1-4 sequentially.
*   **Output Files (in `output_dir/subject_id/`):**
    *   `data/forward_fft.npy`: Raw 4D complex k-space data `(n_trans, nx, ny, nz)`.
    *   `data/fft_prob_density.npy`: (Optional) Normalized 4D k-space power spectrum density.
    *   `data/kspace_mask_<i>.npy`: 3D Boolean k-space masks.
    *   `data/inverse_map_<i>.npy`: Complex reconstructed maps `(n_trans, n_points)`.
    *   `data/gradient_map_<i>.npy`: Real-valued 4D gradient magnitude maps `(n_trans, nx, ny, nz)`.
    *   `plots/*.html`: Interactive Plotly visualizations (generated by separate calls to `generate_volume_plot` with 3D slices of data).

## 5. Inputs and Outputs Summary

*   **Primary Inputs:**
    *   Time-series neuroimaging data (e.g., source-localized EEG/MEG) as a NumPy array.
    *   Corresponding 1D coordinates (`x, y, z`) for each source/sensor.
    *   A list or array of specific time indices (`time_indices`) to analyze.
*   **Key Outputs:**
    *   **Gradient Magnitude Maps:** (`gradient_map_<i>.npy`) 4D NumPy arrays `(n_trans, nx, ny, nz)` representing the spatial rate of change in the reconstructed patterns for specific k-space components *for each selected time point*. Slices along the first dimension (`gradient_map_<i>[t, :, :, :]`) are the candidates for interpretation as excitability proxies at time `t`.
    *   Intermediate results (k-space data, masks, inverse maps) for detailed analysis.
    *   Helper function for interactive visualizations (requires slicing the 4D data).

## 6. Use Cases

This pipeline could be applied in various research scenarios:

*   **Comparing Brain States:** Analyze data from different task conditions (e.g., rest vs. task) or time periods (e.g., pre- vs. post-stimulus) to see if gradient map patterns differ.
*   **Clinical Group Comparisons:** Compare gradient maps between healthy controls and patient groups (e.g., epilepsy, migraine) to identify potential differences in spatial excitability patterns.
*   **Exploring Effects of Interventions:** Assess how brain stimulation (TMS, tDCS) or pharmacological agents alter the computed gradient maps.
*   **Analyzing Spontaneous Activity:** Investigate dynamic changes in gradient patterns during resting-state recordings by analyzing a sequence of time points to understand fluctuations in brain states.

## 7. Interpreting Gradient Maps as Excitability Proxies

As mentioned, slices of the **gradient magnitude maps** (representing specific time points), derived from specific spatial frequency components of the normalized wavefunction, might serve as a **proxy for brain excitability maps** at those moments.

*   **Rationale:** High gradient values indicate regions where the reconstructed signal pattern changes rapidly over space. This could imply functional boundaries, areas of high spatial contrast in neural activity synchronization, or zones transitioning between different states. Such dynamic spatial features *might* be related to the underlying capacity of the neural tissue to respond or transition, i.e., its excitability.
*   **Caveats and Assumptions:**
    *   **Indirect Measure:** This is an indirect, computationally derived measure, not a direct physiological measurement of excitability like TMS-MEP amplitudes.
    *   **Dependence on Preprocessing:** The interpretation relies heavily on the validity of the Hilbert transform for capturing relevant dynamics and the quantum-inspired normalization for representing the instantaneous state.
    *   **Dependence on K-Space Components:** The resulting maps depend strongly on *which* k-space masks (spatial frequencies) are chosen for the inverse transform.
    *   **Temporal Resolution:** While multiple time points can be processed efficiently, each map still represents a snapshot. Analyzing the *evolution* of these maps requires further steps.
    *   **Need for Validation:** The link between these gradient maps and established measures of brain excitability requires empirical validation through correlational studies (e.g., comparing gradient maps with TMS-derived excitability measures, fMRI activation patterns, or clinical outcomes).

## 8. Potential Enhancements for Excitability Mapping

To strengthen the connection between the computed gradient maps and biological brain excitability, several avenues could be explored:

1.  **Integration with Anatomical Priors:** Constrain the analysis or interpretation using structural MRI data (e.g., gray matter segmentation, diffusion tensor imaging for connectivity).
2.  **Advanced K-Space Masking:** Instead of random spherical masks, use data-driven approaches or masks based on specific frequency bands known to be relevant for certain brain functions or pathologies.
3.  **Temporal Dynamics Analysis:** Leverage the `n_trans` output to explicitly model or analyze the temporal evolution of gradient patterns (e.g., compute temporal derivatives, statistics over time windows, fit dynamic models).
4.  **Alternative Normalization Schemes:** Explore different methods for normalizing the initial signal or the k-space data, potentially considering temporal consistency.
5.  **Source Modeling Improvements:** The quality of the input coordinates and time series (from source localization) significantly impacts the results. Using more accurate source modeling techniques is crucial.
6.  **Direct Correlation Studies:** Design experiments to directly compare the pipeline\'s output gradient maps with gold-standard measures of excitability (e.g., TMS-EEG/MEP mapping) in the same subjects.
7.  **Feature Extraction:** Instead of directly interpreting the map, extract quantitative features from the gradient maps (e.g., peak gradient, spatial entropy, regional averages) and correlate these features with behavioral or clinical variables.

By pursuing these directions, the `QM_FFT_Analysis` pipeline can be further refined and validated as a tool for investigating the complex landscape of brain excitability. 