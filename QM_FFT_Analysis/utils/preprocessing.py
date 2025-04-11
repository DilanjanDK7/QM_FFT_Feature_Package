import numpy as np
from scipy.signal import hilbert
import logging
from typing import Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hilbert_transform_axis(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Applies the Hilbert transform along the specified axis.

    Args:
        data (np.ndarray): Input data array.
        axis (int): The axis along which to compute the Hilbert transform. Default is -1 (last axis).

    Returns:
        np.ndarray: Hilbert transformed data (complex).
    """
    logger.debug(f"Applying Hilbert transform to data with shape {data.shape} along axis {axis}")
    if not np.isrealobj(data):
        logger.warning("Input data is complex; taking real part for Hilbert transform.")
        data = data.real
    try:
        hilbert_transformed = hilbert(data, axis=axis)
        # Ensure output is complex64 for consistency if needed downstream, adjust if complex128 preferred
        return hilbert_transformed.astype(np.complex64)
    except Exception as e:
        logger.error(f"Hilbert transform failed: {e}", exc_info=True)
        raise

def get_normalized_wavefunction_at_time(
    time_series_data: np.ndarray,
    time_index: int,
    time_axis: int = -1,
    source_axis: int = 0
) -> np.ndarray:
    """
    Applies Hilbert transform, selects a time point, and normalizes the complex
    wavefunction across sources such that the total probability density is 1.

    Args:
        time_series_data (np.ndarray): Input data (e.g., sources, time).
        time_index (int): Index of the time point to select.
        time_axis (int): Axis representing time. Default is -1.
        source_axis (int): Axis representing sources. Default is 0.

    Returns:
        np.ndarray: Normalized complex wavefunction across sources at the specified time index.
                     Shape will be the same as the input data slice excluding the time axis.
                     The sum of the squared magnitudes along the source axis will be close to 1.
    """
    if time_series_data.ndim < 2:
        raise ValueError("Input data must be at least 2D.")
    if time_axis < 0:
        time_axis += time_series_data.ndim
    if source_axis < 0:
        source_axis += time_series_data.ndim
    if time_axis == source_axis:
        raise ValueError("time_axis and source_axis cannot be the same.")

    logger.info(f"Processing data shape {time_series_data.shape} at time index {time_index} (time_axis={time_axis}, source_axis={source_axis})")

    # 1. Hilbert Transform along time axis
    wavefunction = hilbert_transform_axis(time_series_data, axis=time_axis)

    # 2. Select the specific time point from the complex wavefunction
    slicer = [slice(None)] * wavefunction.ndim
    try:
        slicer[time_axis] = time_index
        wavefunction_at_t = wavefunction[tuple(slicer)]
    except IndexError:
        logger.error(f"time_index {time_index} is out of bounds for time_axis with size {wavefunction.shape[time_axis]}")
        raise IndexError(f"time_index {time_index} out of bounds for axis {time_axis} with size {wavefunction.shape[time_axis]}")

    logger.debug(f"Wavefunction shape at time index {time_index}: {wavefunction_at_t.shape}")

    # 3. Calculate Probability Density
    prob_density_at_t = np.abs(wavefunction_at_t)**2

    # Ensure source_axis index is adjusted if time_axis was before it
    adjusted_source_axis = source_axis if source_axis < time_axis else source_axis - 1

    # 4. Calculate Total Probability across the source axis
    total_prob = np.sum(prob_density_at_t, axis=adjusted_source_axis, keepdims=True)

    # 5. Calculate Normalization Factor (sqrt of total probability)
    # Prevent division by zero or issues with zero probability
    norm_factor = np.sqrt(total_prob)
    norm_factor[norm_factor == 0] = 1.0 # Avoid division by zero

    # 6. Normalize the complex wavefunction
    normalized_wavefunction_at_t = wavefunction_at_t / norm_factor

    # Verification check (optional)
    # final_prob_sum = np.sum(np.abs(normalized_wavefunction_at_t)**2, axis=adjusted_source_axis)
    # logger.debug(f"Sum of probability density after normalization: {final_prob_sum}")

    logger.info("Wavefunction normalization at specified time point complete.")
    return normalized_wavefunction_at_t

def get_normalized_wavefunctions_at_times(
    time_series_data: np.ndarray,
    time_indices: Union[List[int], np.ndarray],
    time_axis: int = -1,
    source_axis: int = 0
) -> np.ndarray:
    """
    Applies Hilbert transform, selects multiple time points, and normalizes
    the complex wavefunction for each time point independently across sources using
    vectorized operations.

    Args:
        time_series_data (np.ndarray): Input data (e.g., sources, time).
        time_indices (Union[List[int], np.ndarray]): Indices of the time points to select.
        time_axis (int): Axis representing time. Default is -1.
        source_axis (int): Axis representing sources. Default is 0.

    Returns:
        np.ndarray: Stacked normalized complex wavefunctions.
                     Shape: (len(time_indices), n_sources) assuming time_axis and source_axis
                     are the primary dimensions handled.
                     The sum of squared magnitudes for each row (time point) along
                     the source dimension will be close to 1.
    """
    if time_series_data.ndim < 2:
        raise ValueError("Input data must be at least 2D.")
    # Ensure time_indices is an array for fancy indexing
    time_indices = np.asarray(time_indices)
    if time_indices.ndim != 1:
        raise ValueError("time_indices must be a 1D list or array.")

    # Normalize axis indices
    if time_axis < 0:
        time_axis += time_series_data.ndim
    if source_axis < 0:
        source_axis += time_series_data.ndim
    if time_axis == source_axis:
        raise ValueError("time_axis and source_axis cannot be the same.")

    logger.info(f"Processing data shape {time_series_data.shape} at {len(time_indices)} time indices (time_axis={time_axis}, source_axis={source_axis})")

    # 1. Hilbert Transform along time axis (once)
    wavefunction_full = hilbert_transform_axis(time_series_data, axis=time_axis)

    # 2. Select multiple time points using fancy indexing
    # We need to construct a multi-dimensional index tuple
    slicer = [slice(None)] * wavefunction_full.ndim
    try:
        slicer[time_axis] = time_indices
        wavefunctions_at_t = wavefunction_full[tuple(slicer)]
        # Output shape depends on original dims and axis order.
        # Example: if input (src, time), time_axis=1, src_axis=0, output is (src, n_times)
        # Example: if input (time, src), time_axis=0, src_axis=1, output is (n_times, src)
    except IndexError as e:
        logger.error(f"One or more time_indices are out of bounds for time_axis with size {wavefunction_full.shape[time_axis]}: {e}")
        raise IndexError(f"Invalid time index found for axis {time_axis} with size {wavefunction_full.shape[time_axis]}")

    logger.debug(f"Selected wavefunctions shape: {wavefunctions_at_t.shape}")

    # Determine the axis corresponding to sources *in the sliced array*
    # If time_axis was removed, the source_axis index might shift down by 1 if it was after time_axis
    # However, fancy indexing doesn't remove the axis, it selects along it.
    # The source axis should retain its original index relative to the *other* non-time axes.
    # Let's assume the output shape is effectively (..., sources, ..., n_selected_times, ...) or (..., n_selected_times, ..., sources, ...)
    # We need the axis index corresponding to the original source_axis in the new shape
    
    # Simpler approach: Reorder axes so that sources and time are first, then select, then normalize
    # Move source_axis to 0, time_axis to 1
    original_axes_order = list(range(time_series_data.ndim))
    other_axes = [i for i in original_axes_order if i not in (source_axis, time_axis)]
    new_order = [source_axis, time_axis] + other_axes
    inv_order = np.argsort(new_order) # To restore original order later if needed
    
    wavefunction_reordered = np.transpose(wavefunction_full, axes=new_order)
    # Shape is now (n_sources, n_time, ...)
    
    # Select time points - now time is always axis 1
    selected_wavefunctions = wavefunction_reordered[:, time_indices, ...]
    # Shape is now (n_sources, n_selected_times, ...)
    
    logger.debug(f"Selected wavefunctions shape after reorder: {selected_wavefunctions.shape}")
    
    # Now, source is axis 0
    prob_density = np.abs(selected_wavefunctions)**2
    total_prob = np.sum(prob_density, axis=0, keepdims=True) # Sum across sources (axis 0)
    norm_factor = np.sqrt(total_prob)
    norm_factor[norm_factor == 0] = 1.0
    
    normalized_wavefunctions = selected_wavefunctions / norm_factor
    # Shape is still (n_sources, n_selected_times, ...)

    # Transpose to (n_selected_times, n_sources, ...) for consistency with n_trans usage
    # Swap axis 0 and 1
    axes_for_final_transpose = list(range(normalized_wavefunctions.ndim))
    axes_for_final_transpose[0], axes_for_final_transpose[1] = axes_for_final_transpose[1], axes_for_final_transpose[0]
    final_normalized_wavefunctions = np.transpose(normalized_wavefunctions, axes=axes_for_final_transpose)
    # Expected shape (n_selected_times, n_sources) if input was 2D

    # Verification (optional)
    # check_prob = np.sum(np.abs(final_normalized_wavefunctions)**2, axis=1) # Sum over source axis (now axis 1)
    # assert np.allclose(check_prob, 1.0)

    logger.info(f"Stacked normalized wavefunctions created with shape: {final_normalized_wavefunctions.shape}")
    return final_normalized_wavefunctions

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Example: 10 sources, 100 time points
    n_sources = 10
    n_time = 100
    dummy_data = np.random.randn(n_sources, n_time)
    target_time_index = 50

    normalized_strengths = get_normalized_wavefunction_at_time(
        dummy_data,
        time_index=target_time_index,
        time_axis=1,
        source_axis=0
    )
    print("Shape of normalized strengths:", normalized_strengths.shape) # Should be (n_sources,)
    # Check if it's complex
    print("Data type:", normalized_strengths.dtype)
    # Check normalization: Sum of squared magnitudes should be close to 1
    total_probability = np.sum(np.abs(normalized_strengths)**2)
    print(f"Total probability (sum |wf|^2) should be close to 1: {total_probability:.6f}")

    # Example with different axis order: (time, sources)
    dummy_data_alt = np.random.randn(n_time, n_sources)
    normalized_strengths_alt = get_normalized_wavefunction_at_time(
        dummy_data_alt,
        time_index=target_time_index,
        time_axis=0,
        source_axis=1
    )
    print("Shape of normalized strengths (alt):", normalized_strengths_alt.shape) # Should be (n_sources,)
    print("Data type (alt):", normalized_strengths_alt.dtype)
    total_probability_alt = np.sum(np.abs(normalized_strengths_alt)**2)
    print(f"Total probability (alt) (sum |wf|^2) should be close to 1: {total_probability_alt:.6f}")

    # --- New Example for Multiple Time Points --- 
    print("\n--- Multi-Time Point Example ---")
    n_sources = 10
    n_time = 100
    dummy_data = np.random.randn(n_sources, n_time)
    target_time_indices = [50, 60, 70]

    normalized_wavefunctions_multi = get_normalized_wavefunctions_at_times(
        dummy_data,
        time_indices=target_time_indices,
        time_axis=1,
        source_axis=0
    )
    print(f"Shape of multi-time normalized wavefunctions: {normalized_wavefunctions_multi.shape}") 
    # Should be (len(target_time_indices), n_sources)
    print(f"Data type: {normalized_wavefunctions_multi.dtype}")
    # Check normalization for each time point (each row)
    total_probabilities = np.sum(np.abs(normalized_wavefunctions_multi)**2, axis=1) # Sum over sources (axis 1)
    print(f"Total probability per time point (should all be ~1): {total_probabilities}")
    assert np.allclose(total_probabilities, 1.0)
    print("Multi-time normalization verified.")

    # Example with different axis order: (time, sources)
    dummy_data_alt = np.random.randn(n_time, n_sources)
    normalized_wavefunctions_multi_alt = get_normalized_wavefunctions_at_times(
        dummy_data_alt,
        time_indices=target_time_indices,
        time_axis=0,
        source_axis=1
    )
    print(f"\nShape of multi-time normalized wavefunctions (alt): {normalized_wavefunctions_multi_alt.shape}") 
    # Should be (len(target_time_indices), n_sources)
    print(f"Data type (alt): {normalized_wavefunctions_multi_alt.dtype}")
    total_probabilities_alt = np.sum(np.abs(normalized_wavefunctions_multi_alt)**2, axis=1) # Sum over sources (axis 1)
    print(f"Total probability per time point (alt) (should all be ~1): {total_probabilities_alt}")
    assert np.allclose(total_probabilities_alt, 1.0)
    print("Multi-time normalization (alt) verified.") 