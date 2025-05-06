import numpy as np
import finufft
import h5py
from pathlib import Path
import logging
from scipy.interpolate import griddata
from scipy.spatial import distance

def setup_logging(name, output_dir=None):
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if output_dir:
        fh = logging.FileHandler(output_dir / "analytical_gradient.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def estimate_optimal_grid_size(x, y, z, upsampling_factor=2.0):
    """
    Estimate optimal grid size based on spatial distribution and Nyquist frequency.
    
    Args:
        x, y, z (ndarray): Coordinate arrays
        upsampling_factor (float): Factor to oversample beyond Nyquist limit
        
    Returns:
        tuple: (nx, ny, nz) grid dimensions
    """
    # Stack coordinates for distance calculation
    coords = np.stack((x, y, z), axis=1)
    
    # Calculate approximate spatial extent in each dimension
    x_extent = np.max(x) - np.min(x)
    y_extent = np.max(y) - np.min(y)
    z_extent = np.max(z) - np.min(z)
    
    # Estimate minimum distance between points (approximation of nearest neighbor)
    # This is more efficient than calculating all pairwise distances
    n_sample = min(1000, len(x))  # Use at most 1000 points for efficiency
    if n_sample < len(x):
        indices = np.random.choice(len(x), n_sample, replace=False)
        sample_coords = coords[indices]
    else:
        sample_coords = coords
    
    # Compute pairwise distances and find minimum non-zero distance
    dist_matrix = distance.pdist(sample_coords)
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    
    # Calculate Nyquist frequency (2 * max_freq where max_freq = 1/(2*min_dist))
    nyquist_freq = 1.0 / min_dist
    
    # Calculate grid dimensions based on Nyquist and spatial extent
    # The factor of 2π comes from the FFT frequency spacing formula: 2π/L
    nx = int(np.ceil(x_extent * nyquist_freq * upsampling_factor / (2 * np.pi)))
    ny = int(np.ceil(y_extent * nyquist_freq * upsampling_factor / (2 * np.pi)))
    nz = int(np.ceil(z_extent * nyquist_freq * upsampling_factor / (2 * np.pi)))
    
    # Ensure dimensions are even (for FFT efficiency)
    nx += nx % 2
    ny += ny % 2
    nz += nz % 2
    
    # Enforce minimum grid size
    min_grid = 10
    nx = max(nx, min_grid)
    ny = max(ny, min_grid)
    nz = max(nz, min_grid)
    
    return nx, ny, nz

def calculate_analytical_gradient(
    x, y, z, strengths, 
    subject_id="subject", 
    output_dir=None,
    nx=None, ny=None, nz=None,
    eps=1e-6,
    dtype='complex128',
    estimate_grid=True,
    upsampling_factor=2,
    export_nifti=False,
    affine_transform=None,
    average=True
):
    """
    Calculate analytical radial gradient directly from non-uniform points.
    
    This function implements the radial derivative method from the paper:
    "Multiscale k-Space Gradient Mapping in fMRI: Theory, Shell Selection, and Excitability Proxy"
    
    It calculates the gradient using the equation:
    ∂f/∂r(x) = F^-1{i2π‖k‖F(k)}
    
    Args:
        x (ndarray): 1D X coordinates of non-uniform points
        y (ndarray): 1D Y coordinates of non-uniform points
        z (ndarray): 1D Z coordinates of non-uniform points
        strengths (ndarray): Strength values at each coordinate.
                           Shape can be (N,) for single transform or (n_trans, N) for multiple.
                           N must be equal to len(x).
        subject_id (str): Unique identifier for the subject, used for output file naming
        output_dir (str or Path, optional): Directory to save outputs. If None, no files are saved.
        nx, ny, nz (int, optional): Number of grid points for each dimension.
                                   If estimate_grid is True, these are estimated automatically.
        eps (float): FINUFFT precision. Defaults to 1e-6.
        dtype (str): Data type for complex values. Must be 'complex128' for FINUFFT.
        estimate_grid (bool): Whether to estimate grid dimensions from point density.
        upsampling_factor (float): Factor for grid estimation. Higher values → finer grid.
        export_nifti (bool): Whether to export results as NIfTI files.
        affine_transform (ndarray, optional): 4x4 affine transformation matrix for NIfTI output.
        average (bool): Whether to compute and save the average gradient over time points.
                        Default is True.
    
    Returns:
        dict: A dictionary containing:
            - 'gradient_map_nu': Gradient map on non-uniform points (n_trans, n_points)
            - 'gradient_map_grid': Gradient map on regular grid (n_trans, nx, ny, nz) if interpolated
            - 'gradient_average_nu': Average gradient map on non-uniform points (n_points) if average=True
            - 'gradient_average_grid': Average gradient map on regular grid (nx, ny, nz) if average=True
            - 'fft_result': Forward FFT result
            - 'coordinates': Dictionary with coordinates and grid information
            - 'k_space_info': Information about the k-space grid (max_k, k_resolution)
    """
    # Set up logging
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the main directory for analytical gradient results
        subject_dir = output_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the Analytical_FFT_Gradient_Maps directory
        gradient_dir = subject_dir / "Analytical_FFT_Gradient_Maps"
        gradient_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the AllTimePoints subdirectory
        all_timepoints_dir = gradient_dir / "AllTimePoints"
        all_timepoints_dir.mkdir(parents=True, exist_ok=True)
    else:
        gradient_dir = None
        all_timepoints_dir = None
    
    logger = setup_logging(f"AnalyticalGradient_{subject_id}", output_dir)
    logger.info("Starting analytical gradient calculation")
    
    # Input validation and preparation
    # Convert coordinates to expected format
    x_in = np.asarray(x, dtype=np.float32).ravel()
    y_in = np.asarray(y, dtype=np.float32).ravel()
    z_in = np.asarray(z, dtype=np.float32).ravel()
    n_points = x_in.size
    
    if not (x_in.shape == y_in.shape == z_in.shape):
        raise ValueError("x, y, and z must have the same number of points after flattening")
    
    # Handle strengths and convert to complex128 for FINUFFT
    strengths_in = np.asarray(strengths, dtype=np.complex128)
    if strengths_in.ndim == 1:
        if strengths_in.size != n_points:
            raise ValueError(f"1D strengths size ({strengths_in.size}) must match number of points ({n_points})")
        strengths_data = strengths_in.reshape(1, n_points)
        n_trans = 1
    elif strengths_in.ndim == 2:
        if strengths_in.shape[1] != n_points:
            raise ValueError(f"Strengths last dimension ({strengths_in.shape[1]}) must match number of points ({n_points})")
        strengths_data = strengths_in
        n_trans = strengths_in.shape[0]
    else:
        raise ValueError("strengths must be a 1D or 2D array")
    
    logger.info(f"Processing data with n_trans={n_trans}, n_points={n_points}")
    
    # Store coordinates for FINUFFT (must be float64)
    x_coords = x_in.astype(np.float64)  # Required by FINUFFT
    y_coords = y_in.astype(np.float64)  # Required by FINUFFT
    z_coords = z_in.astype(np.float64)  # Required by FINUFFT
    
    # Estimate grid dimensions if needed
    if estimate_grid:
        # Use new optimal grid size estimation
        nx, ny, nz = estimate_optimal_grid_size(
            x_coords, y_coords, z_coords, 
            upsampling_factor=upsampling_factor
        )
        logger.info(f"Estimated optimal grid dimensions (nx, ny, nz): ({nx}, {ny}, {nz})")
    else:
        if not all([nx, ny, nz]):
            raise ValueError("If estimate_grid is False, nx, ny, and nz must be provided")
        logger.info(f"Using provided grid dimensions (nx, ny, nz): ({nx}, {ny}, {nz})")
    
    n_modes = (nx, ny, nz)
    
    # Initialize k-space grids (1D)
    kx = np.fft.fftfreq(nx).astype(np.float32)
    ky = np.fft.fftfreq(ny).astype(np.float32)
    kz = np.fft.fftfreq(nz).astype(np.float32)
    
    # Calculate maximum k-space extent and resolution
    k_max = np.sqrt(
        max(np.max(np.abs(kx)), np.max(np.abs(ky)), np.max(np.abs(kz)))**2 * 3
    )
    k_res = min(
        np.diff(np.sort(kx))[0] if nx > 1 else 1,
        np.diff(np.sort(ky))[0] if ny > 1 else 1,
        np.diff(np.sort(kz))[0] if nz > 1 else 1
    )
    
    # Convert to true frequency (2π*normalized frequency)
    # This is needed because fftfreq returns normalized frequencies
    k_max = k_max * 2 * np.pi  
    k_res = k_res * 2 * np.pi
    
    # Check if k-space sampling is sufficient
    # Based on minimum spatial distance between points
    coords = np.stack((x_coords, y_coords, z_coords), axis=1)
    dist_matrix = distance.pdist(coords[:min(1000, n_points)])
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    max_freq_required = 1.0 / min_dist
    
    if k_max < max_freq_required:
        logger.warning(
            f"Estimated k-space extent ({k_max:.4f}) may be insufficient for the minimum "
            f"spatial scale ({min_dist:.4f}). Consider increasing grid size or upsampling factor."
        )
    else:
        logger.info(
            f"K-space parameters: max_k={k_max:.4f}, k_resolution={k_res:.4f}, "
            f"minimum spatial distance={min_dist:.4f}"
        )
    
    # Step 1: Initialize FINUFFT plans
    logger.info("Initializing FINUFFT plans")
    forward_plan = finufft.Plan(1, n_modes, n_trans=n_trans, eps=eps, dtype=dtype)
    forward_plan.setpts(x_coords, y_coords, z_coords)
    
    inverse_plan = finufft.Plan(2, n_modes, n_trans=n_trans, eps=eps, dtype=dtype)
    inverse_plan.setpts(x_coords, y_coords, z_coords)
    
    # Step 2: Compute forward FFT
    logger.info("Computing forward FFT")
    fft_result_flat = forward_plan.execute(strengths_data)
    fft_result = fft_result_flat.reshape(n_trans, nx, ny, nz)
    
    # Step 3: Compute analytical gradient in k-space
    logger.info("Computing analytical radial gradient in k-space")
    
    # Create 3D k-space coordinate grids
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Calculate k-space radial distance from origin (k-magnitude) - this is ||k||
    K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # Multiply by i*2π*||k|| (convert to complex128 for FINUFFT compatibility)
    # This implements the equation from the paper: F^-1{i*2π*||k||*F(k)}
    gradient_fft = fft_result * (1j * 2 * np.pi * K_mag).astype(np.complex128)
    
    # Step 4: Compute inverse FFT to get the gradient map
    logger.info("Computing inverse FFT for gradient map")
    gradient_map_flat = inverse_plan.execute(gradient_fft)
    
    # Reshape result to (n_trans, n_points)
    gradient_map_nu = gradient_map_flat.reshape(n_trans, n_points)
    
    # Calculate the magnitude of the complex-valued gradient
    gradient_magnitude_nu = np.abs(gradient_map_nu)
    
    # Initialize results dictionary
    results = {
        'gradient_map_nu': gradient_magnitude_nu,
        'fft_result': fft_result,
        'coordinates': {
            'x': x_coords,
            'y': y_coords, 
            'z': z_coords,
            'nx': nx,
            'ny': ny,
            'nz': nz
        },
        'k_space_info': {
            'max_k': k_max,
            'k_resolution': k_res,
            'min_spatial_distance': min_dist,
            'max_freq_required': max_freq_required
        }
    }
    
    # Compute average gradient if requested
    if average and n_trans > 1:
        logger.info("Computing average gradient magnitude across time points")
        gradient_average_nu = np.mean(gradient_magnitude_nu, axis=0)
        results['gradient_average_nu'] = gradient_average_nu
    
    # Optionally, interpolate onto a regular grid for visualization/compatibility
    logger.info("Interpolating gradient map onto regular grid")
    points_nu = np.stack((x_coords, y_coords, z_coords), axis=-1)
    
    # Define the regular grid for interpolation
    grid_x = np.linspace(x_coords.min(), x_coords.max(), nx)
    grid_y = np.linspace(y_coords.min(), y_coords.max(), ny)
    grid_z = np.linspace(z_coords.min(), z_coords.max(), nz)
    
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    points_u = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    
    # Interpolate gradient magnitude onto regular grid
    gradient_magnitude_grid = np.zeros((n_trans, nx, ny, nz), dtype=np.float32)
    
    for t in range(n_trans):
        # Determine fill value (mean is typically a reasonable choice)
        fill_val = np.mean(gradient_magnitude_nu[t])
        
        # Interpolate using scipy's griddata
        interpolated_data_flat = griddata(
            points_nu, 
            gradient_magnitude_nu[t],
            points_u, 
            method='linear',
            fill_value=fill_val
        )
        
        gradient_magnitude_grid[t] = interpolated_data_flat.reshape(nx, ny, nz)
    
    # Add gridded result to return dictionary
    results['gradient_map_grid'] = gradient_magnitude_grid
    
    # Interpolate average gradient if it exists
    if average and n_trans > 1:
        # Determine fill value for average
        fill_val_avg = np.mean(gradient_average_nu)
        
        # Interpolate average gradient
        interpolated_avg_flat = griddata(
            points_nu,
            gradient_average_nu,
            points_u,
            method='linear',
            fill_value=fill_val_avg
        )
        
        gradient_average_grid = interpolated_avg_flat.reshape(nx, ny, nz)
        results['gradient_average_grid'] = gradient_average_grid
    
    # Save outputs if output_dir is provided
    if output_dir:
        logger.info(f"Saving results to {gradient_dir}")
        
        # Save average gradient to the main gradient directory
        if average and n_trans > 1:
            with h5py.File(gradient_dir / 'average_gradient.h5', 'w') as f:
                # Save average non-uniform gradient map
                f.create_dataset('gradient_average_nu', data=gradient_average_nu, 
                                compression="gzip", compression_opts=9)
                
                # Save average interpolated grid map
                f.create_dataset('gradient_average_grid', data=gradient_average_grid,
                                compression="gzip", compression_opts=9)
                
                # Save coordinates and grid information
                coords_group = f.create_group('coordinates')
                coords_group.create_dataset('x', data=x_coords)
                coords_group.create_dataset('y', data=y_coords)
                coords_group.create_dataset('z', data=z_coords)
                
                # Save k-space information
                kspace_group = f.create_group('k_space_info')
                kspace_group.attrs['max_k'] = k_max
                kspace_group.attrs['k_resolution'] = k_res
                kspace_group.attrs['min_spatial_distance'] = min_dist
                kspace_group.attrs['max_freq_required'] = max_freq_required
                
                # Save grid dimensions as attributes
                f.attrs['nx'] = nx
                f.attrs['ny'] = ny
                f.attrs['nz'] = nz
                f.attrs['n_points'] = n_points
                
                logger.info("Saved average gradient map")
            
            # Export average gradient to NIfTI if requested
            if export_nifti:
                try:
                    import nibabel as nib
                    
                    # If no affine provided, create a simple identity matrix
                    if affine_transform is None:
                        affine_transform = np.eye(4)
                    
                    # Create NIfTI image from the average gradient map
                    nifti_img = nib.Nifti1Image(gradient_average_grid, affine_transform)
                    
                    # Save the image
                    output_file = gradient_dir / "average_gradient.nii.gz"
                    nib.save(nifti_img, output_file)
                    logger.info(f"Saved average NIfTI image to {output_file}")
                    
                except ImportError:
                    logger.warning("nibabel package not found. NIfTI export skipped.")
        
        # Save all time points to the AllTimePoints subdirectory
        with h5py.File(all_timepoints_dir / 'all_gradients.h5', 'w') as f:
            # Save non-uniform gradient maps for all time points
            f.create_dataset('gradient_map_nu', data=gradient_magnitude_nu, 
                            compression="gzip", compression_opts=9)
            
            # Save interpolated grid maps
            f.create_dataset('gradient_map_grid', data=gradient_magnitude_grid,
                            compression="gzip", compression_opts=9)
            
            # Save k-space data
            f.create_dataset('fft_result', data=fft_result,
                            compression="gzip", compression_opts=9)
            
            # Save coordinates
            coords_group = f.create_group('coordinates')
            coords_group.create_dataset('x', data=x_coords)
            coords_group.create_dataset('y', data=y_coords)
            coords_group.create_dataset('z', data=z_coords)
            
            # Save k-space information
            kspace_group = f.create_group('k_space_info')
            kspace_group.attrs['max_k'] = k_max
            kspace_group.attrs['k_resolution'] = k_res
            kspace_group.attrs['min_spatial_distance'] = min_dist
            kspace_group.attrs['max_freq_required'] = max_freq_required
            
            # Save grid dimensions
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['nz'] = nz
            f.attrs['n_trans'] = n_trans
            f.attrs['n_points'] = n_points
            
            logger.info("Saved gradient maps for all time points")
        
        # Export each time point to NIfTI if requested
        if export_nifti:
            try:
                import nibabel as nib
                
                # If no affine provided, create a simple identity matrix
                if affine_transform is None:
                    affine_transform = np.eye(4)
                
                # Export each time point as a separate NIfTI file
                for t in range(n_trans):
                    # Create NIfTI image from the gradient map
                    nifti_img = nib.Nifti1Image(gradient_magnitude_grid[t], affine_transform)
                    
                    # Save the image
                    output_file = all_timepoints_dir / f"gradient_map_t{t}.nii.gz"
                    nib.save(nifti_img, output_file)
                    logger.info(f"Saved NIfTI image for time point {t} to {output_file}")
                    
            except ImportError:
                logger.warning("nibabel package not found. NIfTI export skipped.")
    
    logger.info("Analytical gradient calculation complete")
    return results 