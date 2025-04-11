# Analysis Module Guide (`map_analysis.py`)

This guide describes the functions available in the `QM_FFT_Analysis.utils.map_analysis` module. These functions are designed to calculate various metrics directly from the complex-valued inverse maps computed at the original non-uniform point locations (typically having a shape of `(n_trans, n_points)`).

These functions are primarily called by the `MapBuilder.analyze_inverse_maps` method but can also be used independently if needed.

## Functions

### `calculate_magnitude(inverse_map_nu)`

*   **Purpose:** Computes the element-wise magnitude (absolute value) of the complex inverse map.
*   **Args:**
    *   `inverse_map_nu` (np.ndarray): The complex inverse map array, shape `(n_trans, n_points)`.
*   **Returns:** (np.ndarray) A real-valued array of the same shape containing the magnitudes.
*   **Usage:** Useful for understanding the strength or intensity of the signal reconstructed at each non-uniform point for each transform.

### `calculate_phase(inverse_map_nu)`

*   **Purpose:** Computes the element-wise phase angle (in radians) of the complex inverse map.
*   **Args:**
    *   `inverse_map_nu` (np.ndarray): The complex inverse map array, shape `(n_trans, n_points)`.
*   **Returns:** (np.ndarray) A real-valued array of the same shape containing the phase angles (typically in the range `[-pi, pi]`).
*   **Usage:** Useful for analyzing phase relationships or coherence in the reconstructed signal.

### `calculate_local_variance(inverse_map_nu, points_nu, k=5)`

*   **Purpose:** Calculates the variance of the *magnitude* within a local neighborhood around each non-uniform point. This gives a measure of how much the signal intensity fluctuates spatially in the immediate vicinity of each point.
*   **Args:**
    *   `inverse_map_nu` (np.ndarray): The complex inverse map array, shape `(n_trans, n_points)`.
    *   `points_nu` (np.ndarray): The coordinates of the non-uniform points, shape `(n_points, 3)`. Used to determine neighborhoods.
    *   `k` (int, optional): The number of nearest neighbors (excluding the point itself) to include in the variance calculation (default: 5). Requires `scipy`.
*   **Returns:** (np.ndarray) A real-valued array of the same shape `(n_trans, n_points)` containing the local variance of the magnitude.
*   **Details:** Uses `scipy.spatial.KDTree` to efficiently find the `k` nearest neighbors for each point based on the provided `points_nu` coordinates. It then calculates the variance of the magnitudes (`np.abs()`) of the `inverse_map_nu` values corresponding to those neighbors.
*   **Usage:** Can highlight regions of high spatial fluctuation or heterogeneity in the signal magnitude.

### `calculate_temporal_difference(map_data)`

*   **Purpose:** Computes the difference between consecutive transforms (along `axis=0`). This is useful if `n_trans` represents time points, allowing calculation of the change between `t` and `t-1`.
*   **Args:**
    *   `map_data` (np.ndarray): The input data array, shape `(n_trans, n_points)`. This can be the raw complex `inverse_map_nu`, or derived real data like magnitude or phase maps.
*   **Returns:** (np.ndarray | None) An array containing the differences, shape `(n_trans - 1, n_points)`. Returns `None` if `n_trans < 2`.
*   **Usage:** Directly measures the change in a quantity (magnitude, phase, etc.) between successive transforms (e.g., time steps) at each non-uniform point. 