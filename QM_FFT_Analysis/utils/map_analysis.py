import numpy as np
from scipy.spatial import KDTree
import logging
import functools

# Try to import numba for JIT compilation
try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Configure basic logging if run standalone (optional)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use module-level logger


def calculate_magnitude(inverse_map_nu):
    """Calculates the magnitude of the complex inverse map.

    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).

    Returns:
        np.ndarray: Real array of magnitudes, shape (n_trans, n_points).
    """
    logger.debug(f"Calculating magnitude for map shape {inverse_map_nu.shape}")
    # Use float32 for memory efficiency if appropriate
    return np.abs(inverse_map_nu).astype(np.float32)


def calculate_phase(inverse_map_nu):
    """Calculates the phase angle (in radians) of the complex inverse map.

    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).

    Returns:
        np.ndarray: Real array of phase angles, shape (n_trans, n_points).
    """
    logger.debug(f"Calculating phase for map shape {inverse_map_nu.shape}")
    # Use float32 for memory efficiency if appropriate
    return np.angle(inverse_map_nu).astype(np.float32)


# Cache for KDTree to avoid rebuilding it for the same point set
@functools.lru_cache(maxsize=8)
def _get_kdtree(points_tuple):
    """Get a KDTree for the given points.
    
    Args:
        points_tuple (tuple): Tuple of coordinates as a hashable type for caching.
        
    Returns:
        KDTree: The KDTree for the provided points.
    """
    # Convert tuple back to numpy array
    points = np.array(points_tuple)
    return KDTree(points)


def calculate_local_variance(inverse_map_nu, points_nu, k=5):
    """Calculates the local variance of the magnitude around each non-uniform point.

    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).
        points_nu (np.ndarray): Array of non-uniform point coordinates, shape (n_points, 3).
        k (int): Number of nearest neighbors to consider for variance calculation.

    Returns:
        np.ndarray: Real array of local variances, shape (n_trans, n_points).
    """
    n_trans, n_points = inverse_map_nu.shape
    logger.debug(f"Calculating local variance for map shape {inverse_map_nu.shape} with k={k}")

    if n_points <= k:
        logger.warning(f"Number of points ({n_points}) is less than or equal to k ({k}). Cannot compute local variance reliably. Returning zeros.")
        return np.zeros_like(inverse_map_nu, dtype=np.float32)  # Using float32 for memory efficiency
        
    try:
        # Precompute magnitude for all points
        magnitude_map = np.abs(inverse_map_nu)
        local_variance_map = np.zeros_like(magnitude_map, dtype=np.float32)  # Using float32 for memory efficiency

        # Convert points to a hashable type for caching
        points_tuple = tuple(map(tuple, points_nu))
        
        # Get KDTree (cached if available)
        tree = _get_kdtree(points_tuple)
        
        # Query all points at once for their k+1 nearest neighbors
        distances, indices = tree.query(points_nu, k=k+1)
        
        # Process each time point
        for t in range(n_trans):
            # Get magnitude for current time point
            magnitudes_t = magnitude_map[t]
            
            # For each point, get the variance of its neighbors' magnitudes
            for i in range(n_points):
                # Skip the point itself (first element) if distance is 0
                if distances[i, 0] == 0:
                    neighbor_indices = indices[i, 1:k+1]
                else:
                    neighbor_indices = indices[i, :k]
                
                if len(neighbor_indices) < 2:  # Need at least 2 points for variance
                    local_variance_map[t, i] = 0
                else:
                    # Calculate variance efficiently
                    neighbor_magnitudes = magnitudes_t[neighbor_indices]
                    local_variance_map[t, i] = np.var(neighbor_magnitudes, dtype=np.float32)
                    
        logger.debug("Local variance calculation complete.")
        return local_variance_map
    
    except Exception as e:
        logger.error(f"Error during local variance calculation: {e}", exc_info=True)
        # Return zeros or raise, depending on desired robustness
        return np.zeros_like(inverse_map_nu, dtype=np.float32)


def calculate_local_variance_vectorized(inverse_map_nu, points_nu, k=5):
    """Vectorized version for calculating the local variance of the magnitude.
    
    This function performs the same calculation as calculate_local_variance
    but using more vectorized operations for better performance with large datasets.

    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).
        points_nu (np.ndarray): Array of non-uniform point coordinates, shape (n_points, 3).
        k (int): Number of nearest neighbors to consider for variance calculation.

    Returns:
        np.ndarray: Real array of local variances, shape (n_trans, n_points).
    """
    n_trans, n_points = inverse_map_nu.shape
    logger.debug(f"Calculating local variance (vectorized) for map shape {inverse_map_nu.shape} with k={k}")

    if n_points <= k:
        logger.warning(f"Number of points ({n_points}) is less than or equal to k ({k}). Cannot compute local variance reliably. Returning zeros.")
        return np.zeros((n_trans, n_points), dtype=np.float32)
        
    try:
        # Precompute magnitude for all points
        magnitude_map = np.abs(inverse_map_nu).astype(np.float32)
        
        # Convert points to a hashable type for caching
        points_tuple = tuple(map(tuple, points_nu))
        
        # Get KDTree (cached if available)
        tree = _get_kdtree(points_tuple)
        
        # Query all points at once for their k+1 nearest neighbors
        distances, indices = tree.query(points_nu, k=k+1)
        
        # Create output array
        local_variance_map = np.zeros((n_trans, n_points), dtype=np.float32)
        
        # Process each transform/time point
        for t in range(n_trans):
            # Get magnitudes for current time point
            magnitudes_t = magnitude_map[t]
            
            # Initialize a mask for points where the first neighbor is the point itself
            self_is_first = distances[:, 0] == 0
            
            # Handle points where self is first neighbor
            for i in np.where(self_is_first)[0]:
                neighbor_idx = indices[i, 1:k+1]
                if len(neighbor_idx) >= 2:
                    local_variance_map[t, i] = np.var(magnitudes_t[neighbor_idx], dtype=np.float32)
                    
            # Handle points where self is not first neighbor
            for i in np.where(~self_is_first)[0]:
                neighbor_idx = indices[i, :k]
                if len(neighbor_idx) >= 2:
                    local_variance_map[t, i] = np.var(magnitudes_t[neighbor_idx], dtype=np.float32)
        
        logger.debug("Vectorized local variance calculation complete.")
        return local_variance_map
    
    except Exception as e:
        logger.error(f"Error during vectorized local variance calculation: {e}", exc_info=True)
        return np.zeros((n_trans, n_points), dtype=np.float32)


if _NUMBA_AVAILABLE:
    @numba.njit(parallel=True)
    def _calculate_variances_jit(magnitudes, indices, n_trans, n_points, k, self_is_first):
        """JIT-compiled inner loop for variance calculation.
        
        Args:
            magnitudes (ndarray): Magnitude values, shape (n_trans, n_points)
            indices (ndarray): Indices of neighbors from KDTree, shape (n_points, k+1)
            n_trans (int): Number of transforms
            n_points (int): Number of points
            k (int): Number of neighbors
            self_is_first (ndarray): Boolean mask indicating where self is first neighbor
            
        Returns:
            ndarray: Local variance values, shape (n_trans, n_points)
        """
        result = np.zeros((n_trans, n_points), dtype=np.float32)
        
        for t in range(n_trans):
            magnitudes_t = magnitudes[t]
            
            for i in numba.prange(n_points):
                if self_is_first[i]:
                    # Skip the point itself (first index)
                    neighbor_indices = indices[i, 1:k+1]
                else:
                    # Use all indices since point itself isn't included
                    neighbor_indices = indices[i, :k]
                
                if len(neighbor_indices) >= 2:
                    # Extract values and calculate variance
                    values = np.array([magnitudes_t[idx] for idx in neighbor_indices])
                    mean_val = np.mean(values)
                    squared_diff_sum = 0.0
                    for val in values:
                        squared_diff_sum += (val - mean_val) ** 2
                    result[t, i] = squared_diff_sum / len(values)
                
        return result


def calculate_local_variance_fully_vectorized(inverse_map_nu, points_nu, k=5):
    """Fully vectorized implementation of local variance calculation with optional JIT acceleration.
    
    This function is optimized for large datasets and will use Numba JIT compilation
    if available for maximum performance.
    
    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).
        points_nu (np.ndarray): Array of non-uniform point coordinates, shape (n_points, 3).
        k (int): Number of nearest neighbors to consider for variance calculation.
        
    Returns:
        np.ndarray: Real array of local variances, shape (n_trans, n_points).
    """
    n_trans, n_points = inverse_map_nu.shape
    logger.debug(f"Calculating fully vectorized local variance for map shape {inverse_map_nu.shape} with k={k}")
    
    if n_points <= k:
        logger.warning(f"Number of points ({n_points}) is less than or equal to k ({k}). Cannot compute local variance reliably. Returning zeros.")
        return np.zeros((n_trans, n_points), dtype=np.float32)
    
    try:
        # Convert to float32 for efficiency and consistency
        magnitude_map = np.abs(inverse_map_nu).astype(np.float32)
        
        # Convert points to a hashable type for caching
        points_tuple = tuple(map(tuple, points_nu))
        
        # Get KDTree (cached if available)
        tree = _get_kdtree(points_tuple)
        
        # Query all points at once for their k+1 nearest neighbors
        distances, indices = tree.query(points_nu, k=k+1)
        
        # Identify points where the first neighbor is the point itself
        self_is_first = distances[:, 0] == 0
        
        # Use JIT-accelerated implementation if available
        if _NUMBA_AVAILABLE:
            logger.debug("Using Numba-accelerated implementation for local variance calculation")
            local_variance_map = _calculate_variances_jit(
                magnitude_map, indices, n_trans, n_points, k, self_is_first
            )
        else:
            logger.debug("Using pure NumPy implementation for local variance calculation")
            # Initialize output array
            local_variance_map = np.zeros((n_trans, n_points), dtype=np.float32)
            
            # For each time point
            for t in range(n_trans):
                magnitudes_t = magnitude_map[t]
                
                # Create a mask for valid points (having at least 2 neighbors)
                valid_self_first = np.ones(n_points, dtype=bool)
                valid_not_self_first = np.ones(n_points, dtype=bool)
                
                # Process points with self as first neighbor
                # Extract neighbor indices for all points with self as first neighbor
                neighbor_indices_self_first = indices[self_is_first, 1:k+1]
                
                # For each point where self is first neighbor
                for i, idx in enumerate(np.where(self_is_first)[0]):
                    if len(neighbor_indices_self_first[i]) >= 2:
                        # Extract neighbor magnitudes
                        neighbor_mags = magnitudes_t[neighbor_indices_self_first[i]]
                        # Calculate variance
                        local_variance_map[t, idx] = np.var(neighbor_mags, dtype=np.float32)
                    else:
                        valid_self_first[i] = False
                        
                # Process points where self is not first neighbor
                # Extract neighbor indices for all points without self as first neighbor
                neighbor_indices_not_self_first = indices[~self_is_first, :k]
                
                # For each point where self is not first neighbor
                for i, idx in enumerate(np.where(~self_is_first)[0]):
                    if len(neighbor_indices_not_self_first[i]) >= 2:
                        # Extract neighbor magnitudes
                        neighbor_mags = magnitudes_t[neighbor_indices_not_self_first[i]]
                        # Calculate variance
                        local_variance_map[t, idx] = np.var(neighbor_mags, dtype=np.float32)
                    else:
                        valid_not_self_first[i] = False
        
        logger.debug("Fully vectorized local variance calculation complete.")
        return local_variance_map
        
    except Exception as e:
        logger.error(f"Error during fully vectorized local variance calculation: {e}", exc_info=True)
        return np.zeros((n_trans, n_points), dtype=np.float32)


def calculate_temporal_difference(map_data):
    """Calculates the difference between consecutive transforms (e.g., time points).

    Args:
        map_data (np.ndarray): Real or complex array of shape (n_trans, n_points). 
                               Can be magnitude, phase, or the raw complex map.

    Returns:
        np.ndarray: Array of differences, shape (n_trans - 1, n_points). Returns
                    None if n_trans < 2.
    """
    n_trans = map_data.shape[0]
    logger.debug(f"Calculating temporal difference for map shape {map_data.shape}")
    
    if n_trans < 2:
        logger.warning("Temporal difference requires at least 2 transforms (time points). Returning None.")
        return None
        
    # Calculate difference between t and t-1
    # Use numpy's optimized diff function
    dtype = np.float32 if np.isrealobj(map_data) else np.complex64
    temporal_diff = np.diff(map_data, axis=0).astype(dtype)
    
    logger.debug(f"Temporal difference calculated, output shape {temporal_diff.shape}")
    return temporal_diff


def calculate_temporal_difference_vectorized(map_data, window_size=1):
    """Calculates temporal differences with vectorized operations and optional window size.
    
    Args:
        map_data (np.ndarray): Real or complex array of shape (n_trans, n_points).
        window_size (int): Size of the window for difference calculation. Default is 1
                           (difference between consecutive time points).
    
    Returns:
        np.ndarray: Array of differences with shape (n_trans - window_size, n_points).
                    Returns None if n_trans <= window_size.
    """
    n_trans = map_data.shape[0]
    logger.debug(f"Calculating vectorized temporal difference for map shape {map_data.shape} with window={window_size}")
    
    if n_trans <= window_size:
        logger.warning(f"Temporal difference requires at least {window_size+1} transforms, but got {n_trans}. Returning None.")
        return None
    
    # Use float32 for real data, complex64 for complex data
    output_dtype = np.float32 if np.isrealobj(map_data) else np.complex64
    
    # Create a view into the original array for the end points
    later_points = map_data[window_size:].astype(output_dtype)
    # Create a view into the original array for the start points
    earlier_points = map_data[:-window_size].astype(output_dtype)
    
    # Directly compute the difference
    temporal_diff = later_points - earlier_points
    
    logger.debug(f"Vectorized temporal difference calculated, output shape {temporal_diff.shape}")
    return temporal_diff 