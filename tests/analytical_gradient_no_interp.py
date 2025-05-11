import numpy as np
from pathlib import Path
import h5py
import nibabel as nib
import logging
import time
import psutil
import os
from QM_FFT_Analysis.utils import calculate_analytical_gradient
from QM_FFT_Analysis.utils.analytical_gradient import setup_logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nifti_data(file_path, downsample_factor=3, time_slice=None):
    """
    Load NIfTI data and return coordinates and data with options for downsampling.
    
    Args:
        file_path: Path to NIfTI file
        downsample_factor: Spatial downsampling factor (higher = fewer points)
        time_slice: Optional slice to select subset of time points (e.g., slice(0, 20, 2))
    """
    nii = nib.load(file_path)
    
    # Get image data - use mmap mode to reduce memory usage during loading
    data = nii.get_fdata(dtype=np.float32, caching='unchanged')
    affine = nii.affine
    
    # Get the dimensions
    nx, ny, nz, nt = data.shape
    logger.info(f"Original dimensions: {nx}x{ny}x{nz}x{nt}")
    
    # Downsample spatial dimensions if requested
    if downsample_factor > 1:
        # Take every nth point in each spatial dimension
        x_indices = np.arange(0, nx, downsample_factor)
        y_indices = np.arange(0, ny, downsample_factor)
        z_indices = np.arange(0, nz, downsample_factor)
        data = data[np.ix_(x_indices, y_indices, z_indices, np.arange(nt))]
        nx, ny, nz, _ = data.shape
        logger.info(f"Downsampled dimensions: {nx}x{ny}x{nz}x{nt}")
    
    # Create coordinate grids for downsampled data
    x, y, z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing='ij'
    )
    
    # Calculate scaling to compensate for downsampling
    scale_matrix = np.eye(4)
    if downsample_factor > 1:
        scale_matrix[0, 0] *= downsample_factor
        scale_matrix[1, 1] *= downsample_factor
        scale_matrix[2, 2] *= downsample_factor
    
    # Apply affine transformation (with scaling) to get real-world coordinates
    adjusted_affine = affine @ scale_matrix
    coords = np.stack([x.flatten(), y.flatten(), z.flatten(), np.ones_like(x.flatten())])
    real_coords = (adjusted_affine @ coords)[:3].T
    
    # Reshape data to (time, points)
    data_reshaped = data.reshape(-1, nt).T
    
    # Select time slice if specified
    if time_slice is not None:
        data_reshaped = data_reshaped[time_slice]
        logger.info(f"Selected time points: {data_reshaped.shape[0]}")
    
    return real_coords, data_reshaped, adjusted_affine, (nx, ny, nz)

def report_memory_usage():
    """Report current memory usage"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")

def calculate_analytical_gradient_no_interp(
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
    Modified version of calculate_analytical_gradient that skips the interpolation step.
    This function calculates the gradient directly on the non-uniform points and does not
    interpolate to a regular grid, which significantly improves performance.
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
    logger.info("Starting analytical gradient calculation (no interpolation)")
    
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
        # Use simple heuristic for grid size estimation
        approx_edge = int(np.ceil(n_points**(1/3))) 
        nx_est = approx_edge * upsampling_factor
        # Make dims even
        nx = nx_est + (nx_est % 2) 
        ny = nx
        nz = nx
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
    k_max = np.sqrt(max(np.max(np.abs(kx)), np.max(np.abs(ky)), np.max(np.abs(kz)))**2 * 3)
    k_res = min(
        np.diff(np.sort(kx))[0] if nx > 1 else 1,
        np.diff(np.sort(ky))[0] if ny > 1 else 1,
        np.diff(np.sort(kz))[0] if nz > 1 else 1
    )
    
    # Convert to true frequency (2π*normalized frequency)
    k_max = k_max * 2 * np.pi  
    k_res = k_res * 2 * np.pi
    
    # Check if k-space sampling is sufficient (basic check)
    logger.info(f"K-space parameters: max_k={k_max:.4f}, k_resolution={k_res:.4f}")
    
    # Step 1: Initialize FINUFFT plans
    logger.info("Initializing FINUFFT plans")
    import finufft
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
            'k_resolution': k_res
        }
    }
    
    # Compute average gradient if requested
    if average and n_trans > 1:
        logger.info("Computing average gradient magnitude across time points")
        gradient_average_nu = np.mean(gradient_magnitude_nu, axis=0)
        results['gradient_average_nu'] = gradient_average_nu
    
    # Save outputs if output_dir is provided
    if output_dir:
        logger.info(f"Saving results to {gradient_dir}")
        
        # Save average gradient to the main gradient directory
        if average and n_trans > 1:
            with h5py.File(gradient_dir / 'average_gradient.h5', 'w') as f:
                # Save average non-uniform gradient map
                f.create_dataset('gradient_average_nu', data=gradient_average_nu, 
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
                
                # Save grid dimensions as attributes
                f.attrs['nx'] = nx
                f.attrs['ny'] = ny
                f.attrs['nz'] = nz
                f.attrs['n_points'] = n_points
                
                logger.info("Saved average gradient map")
            
            # Export average gradient to NIfTI if requested
            if export_nifti:
                logger.warning("NIfTI export requires interpolation, which was skipped. NIfTI export disabled.")
        
        # Save all time points to the AllTimePoints subdirectory
        with h5py.File(all_timepoints_dir / 'all_gradients.h5', 'w') as f:
            # Save non-uniform gradient maps for all time points
            f.create_dataset('gradient_map_nu', data=gradient_magnitude_nu, 
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
            
            # Save grid dimensions
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['nz'] = nz
            f.attrs['n_trans'] = n_trans
            f.attrs['n_points'] = n_points
            
            logger.info("Saved gradient maps for all time points")
    
    logger.info("Analytical gradient calculation complete (no interpolation)")
    return results

def main():
    start_time = time.time()
    
    # Input path
    input_dir = Path("/media/brainlab-uwo/Data1/Results/Full_pipeline_test_5_new_module/derivatives/sub-17017/func")
    
    # Find the NIfTI file
    nifti_files = list(input_dir.glob("*.nii.gz"))
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    
    input_file = nifti_files[0]
    logger.info(f"Processing file: {input_file}")
    
    # Performance optimization parameters
    downsample_factor = 3  # Spatial downsampling factor
    time_slice = slice(0, None, 2)  # Use every other time point
    report_memory_usage()
    
    # Load the data with downsampling
    logger.info(f"Loading data with downsample_factor={downsample_factor}, time_slice={time_slice}")
    coords, data, affine, grid_dims = load_nifti_data(
        input_file, 
        downsample_factor=downsample_factor,
        time_slice=time_slice
    )
    
    # Memory usage after loading
    report_memory_usage()
    logger.info(f"Data loaded: {data.shape[0]} time points, {data.shape[1]} spatial points")
    
    # Convert data to complex (if it's not already)
    if not np.iscomplexobj(data):
        # IMPORTANT: FINUFFT requires complex128 dtype
        data = data.astype(np.complex128)  # Cannot use complex64 with FINUFFT
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run analytical gradient calculation with NO INTERPOLATION
    logger.info("Calculating analytical gradient without interpolation...")
    gradient_start = time.time()
    
    # Use our custom no-interpolation function
    results = calculate_analytical_gradient_no_interp(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        strengths=data,
        subject_id="sub-17017-no-interp",
        output_dir=output_dir,
        # Specify grid dimensions based on downsampled data size
        nx=grid_dims[0],
        ny=grid_dims[1],
        nz=grid_dims[2],
        estimate_grid=False,  # Use provided grid dimensions
        eps=1e-5,  # Slightly lower precision for better performance
        dtype='complex128',  # Must use complex128 with FINUFFT
        average=True,
        upsampling_factor=1.5,  # Lower upsampling factor for better performance
        export_nifti=False  # Skip NIfTI export (requires interpolation)
    )
    
    gradient_end = time.time()
    report_memory_usage()
    
    logger.info(f"Calculation complete! Took {gradient_end - gradient_start:.2f} seconds")
    logger.info(f"Results saved in: {output_dir / 'sub-17017-no-interp' / 'Analytical_FFT_Gradient_Maps'}")
    
    # Print result information
    logger.info(f"Gradient map shape: {results['gradient_map_nu'].shape}")
    if 'gradient_average_nu' in results:
        logger.info(f"Average gradient shape: {results['gradient_average_nu'].shape}")
    logger.info(f"K-space info: {results['k_space_info']}")
    
    # Total runtime
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 