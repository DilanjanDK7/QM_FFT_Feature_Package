import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import calculate_analytical_gradient
import nibabel as nib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nifti_data(file_path):
    """Load NIfTI data and return coordinates and data."""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    
    # Get the dimensions
    nx, ny, nz, nt = data.shape
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing='ij'
    )
    
    # Apply affine transformation to get real-world coordinates
    coords = np.stack([x.flatten(), y.flatten(), z.flatten(), np.ones_like(x.flatten())])
    real_coords = (affine @ coords)[:3].T
    
    # Reshape data to (time, points)
    data_reshaped = data.reshape(-1, nt).T
    
    return real_coords, data_reshaped, affine

def main():
    # Input path
    input_dir = Path("/media/brainlab-uwo/Data1/Results/Full_pipeline_test_5_new_module/derivatives/sub-17017/func")
    
    # Find the NIfTI file (assuming it's the only .nii.gz file in the directory)
    nifti_files = list(input_dir.glob("*.nii.gz"))
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    
    input_file = nifti_files[0]
    logger.info(f"Processing file: {input_file}")
    
    # Load the data
    coords, data, affine = load_nifti_data(input_file)
    
    # Convert data to complex (if it's not already)
    if not np.iscomplexobj(data):
        data = data.astype(np.complex128)
    
    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run analytical gradient calculation
    logger.info("Calculating analytical gradient...")
    results = calculate_analytical_gradient(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        strengths=data,
        subject_id="sub-17017",
        output_dir=output_dir,
        export_nifti=True,
        affine_transform=affine,
        average=True
    )
    
    logger.info("Calculation complete!")
    logger.info(f"Results saved in: {output_dir / 'sub-17017' / 'Analytical_FFT_Gradient_Maps'}")
    
    # Print some information about the results
    logger.info(f"Gradient map shape: {results['gradient_map_nu'].shape}")
    if 'gradient_average_nu' in results:
        logger.info(f"Average gradient shape: {results['gradient_average_nu'].shape}")
    logger.info(f"K-space info: {results['k_space_info']}")

if __name__ == "__main__":
    main() 