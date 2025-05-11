import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import calculate_analytical_gradient
import nibabel as nib
import logging
import time
import psutil
import os

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
    
    # Run analytical gradient calculation with optimized parameters
    logger.info("Calculating analytical gradient...")
    gradient_start = time.time()
    
    results = calculate_analytical_gradient(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        strengths=data,
        subject_id="sub-17017-optimized",
        output_dir=output_dir,
        # Specify grid dimensions based on downsampled data size
        nx=grid_dims[0],
        ny=grid_dims[1],
        nz=grid_dims[2],
        estimate_grid=False,  # Use provided grid dimensions
        eps=1e-5,  # Slightly lower precision for better performance
        dtype='complex128',  # Must use complex128 with FINUFFT
        export_nifti=True,
        affine_transform=affine,
        average=True,
        upsampling_factor=1.5  # Lower upsampling factor for better performance
    )
    
    gradient_end = time.time()
    report_memory_usage()
    
    logger.info(f"Calculation complete! Took {gradient_end - gradient_start:.2f} seconds")
    logger.info(f"Results saved in: {output_dir / 'sub-17017-optimized' / 'Analytical_FFT_Gradient_Maps'}")
    
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