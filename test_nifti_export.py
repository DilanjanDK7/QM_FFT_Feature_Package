import numpy as np
import tempfile
from pathlib import Path
from QM_FFT_Analysis.utils import calculate_analytical_gradient

def test_nifti_with_interpolation_flags():
    """Test how skip_interpolation interacts with NIfTI export."""
    # Create sample data
    n_points = 100
    n_trans = 3
    
    # Create coordinates in range [-1, 1]
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(-1, 1, n_points)
    
    # Create complex strengths with 3 time points
    strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        subject_id = "test_subject"
        
        print("Test 1: With skip_interpolation=True and export_nifti=True")
        try:
            # This should warn about interpolation being required for NIfTI export
            calculate_analytical_gradient(
                x=x, y=y, z=z, strengths=strengths,
                subject_id=subject_id,
                output_dir=temp_path,
                average=True,
                export_nifti=True,
                skip_interpolation=True
            )
            
            # Check if NIfTI files were created (should NOT exist with skip_interpolation=True)
            subject_dir = temp_path / subject_id
            gradient_dir = subject_dir / "Analytical_FFT_Gradient_Maps"
            average_nifti = gradient_dir / "average_gradient.nii.gz"
            
            print(f"NIfTI file exists: {average_nifti.exists()} (Should be False)")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\nTest 2: With skip_interpolation=False and export_nifti=True")
        try:
            # This should succeed and create NIfTI files
            calculate_analytical_gradient(
                x=x, y=y, z=z, strengths=strengths,
                subject_id=subject_id,
                output_dir=temp_path,
                average=True,
                export_nifti=True,
                skip_interpolation=False
            )
            
            # Check if NIfTI files were created (should exist with skip_interpolation=False)
            subject_dir = temp_path / subject_id
            gradient_dir = subject_dir / "Analytical_FFT_Gradient_Maps"
            average_nifti = gradient_dir / "average_gradient.nii.gz"
            
            print(f"NIfTI file exists: {average_nifti.exists()} (Should be True)")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_nifti_with_interpolation_flags() 