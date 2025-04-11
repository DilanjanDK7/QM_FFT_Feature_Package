import numpy as np
from scipy.spatial import KDTree
import logging

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
    return np.abs(inverse_map_nu)


def calculate_phase(inverse_map_nu):
    """Calculates the phase angle (in radians) of the complex inverse map.

    Args:
        inverse_map_nu (np.ndarray): Complex array of shape (n_trans, n_points).

    Returns:
        np.ndarray: Real array of phase angles, shape (n_trans, n_points).
    """
    logger.debug(f"Calculating phase for map shape {inverse_map_nu.shape}")
    return np.angle(inverse_map_nu)


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
        return np.zeros_like(inverse_map_nu, dtype=np.float64)
        
    try:
        magnitude_map = np.abs(inverse_map_nu)
        local_variance_map = np.zeros_like(magnitude_map)

        # Build KDTree for efficient neighbor lookup
        tree = KDTree(points_nu)

        for t in range(n_trans):
            for i in range(n_points):
                # Query for k+1 neighbors (includes the point itself)
                distances, indices = tree.query(points_nu[i], k=k + 1)
                
                # Exclude the point itself (usually the first index if distance is 0)
                neighbor_indices = indices[1:] if distances[0] == 0 else indices[:k] 
                
                if len(neighbor_indices) < 2: # Need at least 2 points for variance
                    local_variance_map[t, i] = 0 # Or np.nan, depending on desired handling
                else:
                     # Calculate variance of magnitude among neighbors
                    neighbor_magnitudes = magnitude_map[t, neighbor_indices]
                    local_variance_map[t, i] = np.var(neighbor_magnitudes)
        
        logger.debug("Local variance calculation complete.")
        return local_variance_map
    except Exception as e:
        logger.error(f"Error during local variance calculation: {e}", exc_info=True)
        # Return zeros or raise, depending on desired robustness
        return np.zeros_like(inverse_map_nu, dtype=np.float64) 


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
    temporal_diff = np.diff(map_data, axis=0)
    
    logger.debug(f"Temporal difference calculated, output shape {temporal_diff.shape}")
    return temporal_diff 