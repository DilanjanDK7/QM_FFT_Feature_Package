import numpy as np
from pathlib import Path
import logging
import time
import psutil
import os
import nibabel as nib
from QM_FFT_Analysis.utils import calculate_analytical_gradient

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

def run_performance_comparison():
    """Run performance comparison between with and without interpolation"""
    # Generate synthetic data
    n_points = 1000
    n_trans = 5
    
    # Create random coordinates
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Create complex data
    data = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run tests with different configurations
    tests = [
        {
            "name": "with_interpolation",
            "skip_interpolation": False,
            "subject_id": "synthetic-with-interp",
            "export_nifti": False
        },
        {
            "name": "without_interpolation",
            "skip_interpolation": True,
            "subject_id": "synthetic-no-interp",
            "export_nifti": False
        }
    ]
    
    results = {}
    
    for test in tests:
        logger.info(f"\n=== Running test: {test['name']} ===")
        report_memory_usage()
        start_time = time.time()
        
        # Run analytical gradient calculation
        logger.info(f"Calculating analytical gradient ({test['name']})...")
        calc_results = calculate_analytical_gradient(
            x=x,
            y=y,
            z=z,
            strengths=data,
            subject_id=test['subject_id'],
            output_dir=output_dir,
            eps=1e-5,
            dtype='complex128',
            export_nifti=test['export_nifti'],
            average=True,
            upsampling_factor=1.5,
            skip_interpolation=test['skip_interpolation']
        )
        
        end_time = time.time()
        report_memory_usage()
        
        # Record results
        execution_time = end_time - start_time
        results[test['name']] = {
            "execution_time": execution_time,
            "output_size": len(str(calc_results)),
            "gradient_shape": calc_results['gradient_map_nu'].shape,
            "has_grid": 'gradient_map_grid' in calc_results,
        }
        
        logger.info(f"Test {test['name']} complete! Took {execution_time:.2f} seconds")
        logger.info(f"Results saved in: {output_dir / test['subject_id'] / 'Analytical_FFT_Gradient_Maps'}")
    
    # Print comparison
    logger.info("\n=== Performance Comparison Results ===")
    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  Execution time: {result['execution_time']:.2f} seconds")
        logger.info(f"  Has interpolated grid: {result['has_grid']}")
        logger.info(f"  Gradient shape: {result['gradient_shape']}")
    
    # Calculate speedup
    if 'with_interpolation' in results and 'without_interpolation' in results:
        speedup = results['with_interpolation']['execution_time'] / results['without_interpolation']['execution_time']
        logger.info(f"\nSkipping interpolation provides a {speedup:.2f}x speedup!")

if __name__ == "__main__":
    run_performance_comparison() import numpy as np
from pathlib import Path
import logging
import time
import psutil
import os
import nibabel as nib
from QM_FFT_Analysis.utils import calculate_analytical_gradient

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

def run_performance_comparison():
    """Run performance comparison between with and without interpolation"""
    # Generate synthetic data
    n_points = 1000
    n_trans = 5
    
    # Create random coordinates
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)
    
    # Create complex data
    data = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run tests with different configurations
    tests = [
        {
            "name": "with_interpolation",
            "skip_interpolation": False,
            "subject_id": "synthetic-with-interp",
            "export_nifti": False
        },
        {
            "name": "without_interpolation",
            "skip_interpolation": True,
            "subject_id": "synthetic-no-interp",
            "export_nifti": False
        }
    ]
    
    results = {}
    
    for test in tests:
        logger.info(f"\n=== Running test: {test['name']} ===")
        report_memory_usage()
        start_time = time.time()
        
        # Run analytical gradient calculation
        logger.info(f"Calculating analytical gradient ({test['name']})...")
        calc_results = calculate_analytical_gradient(
            x=x,
            y=y,
            z=z,
            strengths=data,
            subject_id=test['subject_id'],
            output_dir=output_dir,
            eps=1e-5,
            dtype='complex128',
            export_nifti=test['export_nifti'],
            average=True,
            upsampling_factor=1.5,
            skip_interpolation=test['skip_interpolation']
        )
        
        end_time = time.time()
        report_memory_usage()
        
        # Record results
        execution_time = end_time - start_time
        results[test['name']] = {
            "execution_time": execution_time,
            "output_size": len(str(calc_results)),
            "gradient_shape": calc_results['gradient_map_nu'].shape,
            "has_grid": 'gradient_map_grid' in calc_results,
        }
        
        logger.info(f"Test {test['name']} complete! Took {execution_time:.2f} seconds")
        logger.info(f"Results saved in: {output_dir / test['subject_id'] / 'Analytical_FFT_Gradient_Maps'}")
    
    # Print comparison
    logger.info("\n=== Performance Comparison Results ===")
    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  Execution time: {result['execution_time']:.2f} seconds")
        logger.info(f"  Has interpolated grid: {result['has_grid']}")
        logger.info(f"  Gradient shape: {result['gradient_shape']}")
    
    # Calculate speedup
    if 'with_interpolation' in results and 'without_interpolation' in results:
        speedup = results['with_interpolation']['execution_time'] / results['without_interpolation']['execution_time']
        logger.info(f"\nSkipping interpolation provides a {speedup:.2f}x speedup!")

if __name__ == "__main__":
    run_performance_comparison() 