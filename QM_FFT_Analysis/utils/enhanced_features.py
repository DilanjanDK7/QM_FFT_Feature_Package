"""
Enhanced Features Module for QM_FFT_Analysis.

This module provides advanced spectral and k-space metrics for quantum mechanical analysis,
including analytic radial-derivative, spectral slope, spectral entropy, anisotropy/orientation
dispersion, higher-order moments, and HRF deconvolution-based excitation maps.
"""

import numpy as np
import logging
import finufft
from scipy import stats
from scipy import signal
from scipy.optimize import curve_fit
import yaml
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load configuration for enhanced features.
    
    Args:
        config_path (str, optional): Path to custom configuration YAML file.
        
    Returns:
        dict: Configuration parameters with defaults applied if not specified.
    """
    # Default configuration
    default_config = {
        "gradient_weighting": True,
        "hrf_kernel": "canonical",
        "slope_fit_range": [0.1, 0.8],  # Fraction of max k
        "entropy_bin_count": 64,
        "anisotropy_moment": 2,
        "moments_order": [3, 4]  # 3=skewness, 4=kurtosis
    }
    
    # If a config path is provided, load and update defaults
    if config_path:
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    default_config.update(custom_config)
            logger.info(f"Loaded custom configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            logger.warning("Using default configuration")
    
    return default_config

def compute_radial_gradient(fft_result, kx, ky, kz, eps=1e-6, dtype='complex128'):
    """
    Compute the radial gradient analytically in k-space using a single inverse NUFFT.
    
    This is faster than computing multiple inverse maps and calculating spatial gradients.
    
    Args:
        fft_result (np.ndarray): Forward FFT result, shape (n_trans, nx, ny, nz).
        kx (np.ndarray): 1D array of k-space coordinates in x direction.
        ky (np.ndarray): 1D array of k-space coordinates in y direction.
        kz (np.ndarray): 1D array of k-space coordinates in z direction.
        eps (float, optional): FINUFFT precision. Defaults to 1e-6.
        dtype (str, optional): Data type for complex values. Defaults to 'complex128'.
        
    Returns:
        np.ndarray: Gradient map with shape (n_trans, nx, ny, nz).
    """
    n_trans, nx, ny, nz = fft_result.shape
    
    # Create 3D k-space coordinate grids
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate k-space radial distance from origin (k-magnitude)
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # Multiply by i*2π*‖k‖
    # This is the frequency domain equivalent of taking radial derivative
    gradient_fft = fft_result * (1j * 2 * np.pi * K_mag)
    
    # Note: The inverse transform is typically done in the MapBuilder class
    # This function returns the modified k-space coefficients
    
    return gradient_fft

def calculate_spectral_slope(fft_result, kx, ky, kz, k_min=None, k_max=None, nbins=50):
    """
    Calculate the spectral slope (power-law exponent) from the radial power spectrum.
    
    Args:
        fft_result (np.ndarray): Forward FFT result, shape (n_trans, nx, ny, nz).
        kx (np.ndarray): 1D array of k-space coordinates in x direction.
        ky (np.ndarray): 1D array of k-space coordinates in y direction.
        kz (np.ndarray): 1D array of k-space coordinates in z direction.
        k_min (float, optional): Minimum k value for fitting. Defaults to 0.1*k_max.
        k_max (float, optional): Maximum k value for fitting. Defaults to 0.8*k_nyquist.
        nbins (int, optional): Number of radial bins. Defaults to 50.
        
    Returns:
        np.ndarray: Spectral slope (alpha) for each transform, shape (n_trans,).
    """
    n_trans = fft_result.shape[0]
    slopes = np.zeros(n_trans)
    
    # Create 3D k-space coordinate grids
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate k-space radial distance from origin
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # Determine k_min and k_max if not provided
    k_nyquist = np.max(K_mag)
    if k_min is None:
        k_min = 0.1 * k_nyquist
    if k_max is None:
        k_max = 0.8 * k_nyquist
    
    # Define radial bins
    r_bins = np.linspace(k_min, k_max, nbins + 1)
    bin_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # Power-law fit function
    def power_law(x, alpha, C):
        return C - alpha * x  # Log-log form: log(P) = -alpha*log(k) + C
    
    for t in range(n_trans):
        # Calculate power spectrum
        power = np.abs(fft_result[t])**2
        
        # Bin power by radial k value
        power_binned = np.zeros(nbins)
        count_binned = np.zeros(nbins, dtype=int)
        
        for i in range(nbins):
            mask = (K_mag >= r_bins[i]) & (K_mag < r_bins[i+1])
            if np.sum(mask) > 0:
                power_binned[i] = np.mean(power[mask])
                count_binned[i] = np.sum(mask)
        
        # Only use bins with data
        valid_bins = count_binned > 0
        if np.sum(valid_bins) >= 3:  # Need at least 3 points for a meaningful fit
            log_k = np.log(bin_centers[valid_bins])
            log_P = np.log(power_binned[valid_bins])
            
            # Linear fit on log-log scale
            try:
                params, _ = curve_fit(power_law, log_k, log_P)
                slopes[t] = params[0]  # Extract alpha
            except:
                logger.warning(f"Failed to fit power law for transform {t}, setting slope to 0")
                slopes[t] = 0
        else:
            logger.warning(f"Insufficient data points for power law fit in transform {t}")
            slopes[t] = 0
    
    return slopes

def calculate_spectral_entropy(fft_result, kx, ky, kz, nbins=64):
    """
    Calculate spectral entropy of k-space distribution.
    
    Args:
        fft_result (np.ndarray): Forward FFT result, shape (n_trans, nx, ny, nz).
        kx (np.ndarray): 1D array of k-space coordinates in x direction.
        ky (np.ndarray): 1D array of k-space coordinates in y direction.
        kz (np.ndarray): 1D array of k-space coordinates in z direction.
        nbins (int, optional): Number of bins for entropy calculation. Defaults to 64.
        
    Returns:
        np.ndarray: Spectral entropy for each transform, shape (n_trans,).
    """
    n_trans = fft_result.shape[0]
    entropy = np.zeros(n_trans)
    
    # Create 3D k-space coordinate grids
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate k-space radial distance from origin
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    k_max = np.max(K_mag)
    
    for t in range(n_trans):
        # Calculate power spectrum
        power = np.abs(fft_result[t])**2
        
        # Bin the power spectrum
        hist, _ = np.histogram(power.flatten(), bins=nbins, range=(0, np.max(power)))
        
        # Normalize to get probability distribution
        p = hist / np.sum(hist)
        
        # Remove zeros to avoid log(0)
        p = p[p > 0]
        
        # Calculate entropy
        entropy[t] = -np.sum(p * np.log2(p))
    
    return entropy

def calculate_kspace_anisotropy(fft_result, kx, ky, kz, moment=2):
    """
    Calculate anisotropy/orientation dispersion using the k-space power distribution.
    
    Args:
        fft_result (np.ndarray): Forward FFT result, shape (n_trans, nx, ny, nz).
        kx (np.ndarray): 1D array of k-space coordinates in x direction.
        ky (np.ndarray): 1D array of k-space coordinates in y direction.
        kz (np.ndarray): 1D array of k-space coordinates in z direction.
        moment (int, optional): Moment order for the tensor calculation. Defaults to 2.
        
    Returns:
        np.ndarray: Fractional anisotropy for each transform, shape (n_trans,).
    """
    n_trans = fft_result.shape[0]
    fa = np.zeros(n_trans)
    
    # Create 3D k-space coordinate grids
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    for t in range(n_trans):
        # Get power as weights
        power = np.abs(fft_result[t])**2
        power_flat = power.flatten()
        
        # Flatten coordinate grids
        kx_flat = Kx.flatten()
        ky_flat = Ky.flatten()
        kz_flat = Kz.flatten()
        
        # Calculate 3x3 moment tensor
        M = np.zeros((3, 3))
        
        for i, j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
            if i == 0:
                ki = kx_flat
            elif i == 1:
                ki = ky_flat
            else:
                ki = kz_flat
                
            if j == 0:
                kj = kx_flat
            elif j == 1:
                kj = ky_flat
            else:
                kj = kz_flat
                
            # Weight by power
            M[i, j] = np.sum(power_flat * ki * kj) / np.sum(power_flat)
        
        # Make tensor symmetric
        M[1, 0] = M[0, 1]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        
        # Calculate eigenvalues
        eigvals = np.linalg.eigvalsh(M)
        
        # Calculate fractional anisotropy (FA)
        mean_eigval = np.mean(eigvals)
        numerator = np.sqrt(np.sum((eigvals - mean_eigval)**2))
        denominator = np.sqrt(np.sum(eigvals**2))
        
        if denominator > 0:
            fa[t] = np.sqrt(1.5) * numerator / denominator
        else:
            fa[t] = 0
    
    return fa

def calculate_higher_order_moments(inverse_map_nu):
    """
    Calculate skewness and kurtosis from inverse maps.
    
    Args:
        inverse_map_nu (np.ndarray): Inverse maps on non-uniform points, shape (n_trans, n_points).
        
    Returns:
        tuple: (skewness, kurtosis) arrays, each with shape (n_trans,).
    """
    n_trans, n_points = inverse_map_nu.shape
    
    # Compute magnitude
    magnitude = np.abs(inverse_map_nu)
    
    # Initialize arrays
    skewness = np.zeros(n_trans)
    kurtosis = np.zeros(n_trans)
    
    for t in range(n_trans):
        # Compute skewness
        skewness[t] = stats.skew(magnitude[t])
        
        # Compute kurtosis (using Fisher's definition: normal = 0)
        kurtosis[t] = stats.kurtosis(magnitude[t])
    
    return skewness, kurtosis

def generate_canonical_hrf(duration=30, tr=1.0):
    """
    Generate a canonical hemodynamic response function (HRF).
    
    Args:
        duration (int, optional): Duration of HRF in seconds. Defaults to 30.
        tr (float, optional): Repetition time in seconds. Defaults to 1.0.
        
    Returns:
        np.ndarray: HRF timeseries.
    """
    # Time vector
    t = np.arange(0, duration, tr)
    
    # Parameters
    peak1 = 6.0      # Peak time for first gamma function
    peak2 = 16.0     # Peak time for second gamma function
    scale1 = 1.0     # Scale for first gamma function
    scale2 = 0.1     # Scale for second gamma function
    
    # Compute gamma functions
    gamma1 = stats.gamma.pdf(t, peak1 / tr, scale=scale1 * tr)
    gamma2 = stats.gamma.pdf(t, peak2 / tr, scale=scale2 * tr)
    
    # Canonical HRF
    hrf = gamma1 - gamma2
    
    # Normalize
    hrf = hrf / np.max(hrf)
    
    return hrf

def calculate_excitation_map(fft_result, time_axis=0, hrf_type='canonical'):
    """
    Calculate excitation map by deconvolving the HRF from the time series.
    
    Args:
        fft_result (np.ndarray): FFT result with time dimension, shape (n_trans, nx, ny, nz).
        time_axis (int, optional): Axis representing time. Defaults to 0.
        hrf_type (str, optional): Type of HRF to use. Defaults to 'canonical'.
        
    Returns:
        np.ndarray: Excitation map, same shape as input.
    """
    # Get time dimension size
    n_time = fft_result.shape[time_axis]
    
    if n_time < 3:
        logger.warning("Insufficient time points for HRF deconvolution (need at least 3)")
        return None
    
    # Generate HRF
    if hrf_type == 'canonical':
        hrf = generate_canonical_hrf(duration=min(30, n_time), tr=1.0)
    else:
        logger.error(f"Unknown HRF type: {hrf_type}")
        return None
    
    # Pad HRF to match n_time
    if len(hrf) < n_time:
        hrf = np.pad(hrf, (0, n_time - len(hrf)))
    else:
        hrf = hrf[:n_time]
    
    # FFT of HRF
    hrf_fft = np.fft.fft(hrf)
    
    # Create output array
    output_shape = list(fft_result.shape)
    excitation_map = np.zeros(output_shape, dtype=fft_result.dtype)
    
    # Process each voxel/point
    # Reshape to make time the first dimension
    shape = fft_result.shape
    dims = list(range(len(shape)))
    dims.remove(time_axis)
    dims = [time_axis] + dims
    temp = np.transpose(fft_result, dims)
    
    # Original shape without time dimension
    spatial_shape = temp.shape[1:]
    temp = temp.reshape(n_time, -1)
    
    # Deconvolve each spatial point
    for i in range(temp.shape[1]):
        time_series = temp[:, i]
        
        # FFT of time series
        ts_fft = np.fft.fft(time_series)
        
        # Deconvolution in frequency domain
        # Add regularization to avoid division by small values
        epsilon = 1e-10 * np.max(np.abs(hrf_fft))
        deconv_fft = ts_fft / (hrf_fft + epsilon)
        
        # Inverse FFT to get excitation
        excitation = np.real(np.fft.ifft(deconv_fft))
        
        # Store result
        temp[:, i] = excitation
    
    # Reshape back to original dimensions
    temp = temp.reshape((n_time,) + spatial_shape)
    
    # Transpose back to original axis order
    inverse_dims = np.zeros(len(dims), dtype=int)
    for i, d in enumerate(dims):
        inverse_dims[d] = i
    excitation_map = np.transpose(temp, inverse_dims)
    
    return excitation_map 