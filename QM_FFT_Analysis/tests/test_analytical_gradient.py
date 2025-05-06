import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
import h5py

from QM_FFT_Analysis.utils import calculate_analytical_gradient

class TestAnalyticalGradient:
    """Test suite for the analytical gradient calculation function."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample 3D data for testing."""
        # Create sample coordinates and strengths
        n_points = 100
        n_trans = 3
        
        # Create coordinates in range [-1, 1]
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.random.uniform(-1, 1, n_points)
        
        # Create complex strengths with 3 time points
        strengths = np.random.randn(n_trans, n_points) + 1j * np.random.randn(n_trans, n_points)
        
        return x, y, z, strengths
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        # Create temp directory
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Clean up after test
        shutil.rmtree(temp_path)
    
    def test_calculation_without_saving(self, sample_data):
        """Test basic calculation without saving to disk."""
        x, y, z, strengths = sample_data
        
        # Run calculation without saving
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            output_dir=None,
            average=True
        )
        
        # Check that the results dictionary contains expected keys
        assert 'gradient_map_nu' in results
        assert 'gradient_map_grid' in results
        assert 'gradient_average_nu' in results
        assert 'gradient_average_grid' in results
        assert 'fft_result' in results
        assert 'coordinates' in results
        
        # Check shapes
        n_trans, n_points = strengths.shape
        assert results['gradient_map_nu'].shape == (n_trans, n_points)
        assert results['gradient_average_nu'].shape == (n_points,)
        
        # Grid dimensions might be estimated
        nx = results['coordinates']['nx']
        ny = results['coordinates']['ny']
        nz = results['coordinates']['nz']
        
        assert results['gradient_map_grid'].shape == (n_trans, nx, ny, nz)
        assert results['gradient_average_grid'].shape == (nx, ny, nz)
        
        # Check that average is correctly computed
        expected_average = np.mean(results['gradient_map_nu'], axis=0)
        np.testing.assert_allclose(results['gradient_average_nu'], expected_average)
    
    def test_calculation_without_averaging(self, sample_data):
        """Test calculation with averaging disabled."""
        x, y, z, strengths = sample_data
        
        # Run calculation without averaging
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            output_dir=None,
            average=False
        )
        
        # Check that average values are not in results
        assert 'gradient_average_nu' not in results
        assert 'gradient_average_grid' not in results
        
        # Check that other results are present
        assert 'gradient_map_nu' in results
        assert 'gradient_map_grid' in results
    
    def test_single_timepoint(self, sample_data):
        """Test with a single time point."""
        x, y, z, strengths = sample_data
        
        # Use only the first time point
        single_strength = strengths[0]
        
        # Run calculation
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=single_strength,
            output_dir=None,
            average=True  # Even with average=True, no averaging happens with single time point
        )
        
        # Check that results have correct shapes
        assert results['gradient_map_nu'].shape == (1, len(x))
        assert results['gradient_map_grid'].shape[0] == 1
        
        # No average should be computed for a single time point
        assert 'gradient_average_nu' not in results
        assert 'gradient_average_grid' not in results
    
    def test_directory_structure(self, sample_data, temp_dir):
        """Test that output files are created with the correct directory structure."""
        x, y, z, strengths = sample_data
        subject_id = "test_subject"
        
        # Run calculation with saving enabled
        calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            subject_id=subject_id,
            output_dir=temp_dir,
            average=True
        )
        
        # Check that directories were created
        subject_dir = temp_dir / subject_id
        gradient_dir = subject_dir / "Analytical_FFT_Gradient_Maps"
        all_timepoints_dir = gradient_dir / "AllTimePoints"
        
        assert subject_dir.exists()
        assert gradient_dir.exists()
        assert all_timepoints_dir.exists()
        
        # Check that files were created
        average_file = gradient_dir / "average_gradient.h5"
        all_timepoints_file = all_timepoints_dir / "all_gradients.h5"
        
        assert average_file.exists()
        assert all_timepoints_file.exists()
        
        # Check file contents
        with h5py.File(average_file, 'r') as f:
            assert 'gradient_average_nu' in f
            assert 'gradient_average_grid' in f
            assert 'coordinates' in f
            assert 'k_space_info' in f  # Check for k-space info
        
        with h5py.File(all_timepoints_file, 'r') as f:
            assert 'gradient_map_nu' in f
            assert 'gradient_map_grid' in f
            assert 'fft_result' in f
            assert 'coordinates' in f
            assert 'k_space_info' in f  # Check for k-space info
    
    def test_nifti_export(self, sample_data, temp_dir):
        """Test NIfTI export if nibabel is available."""
        try:
            import nibabel as nib
            has_nibabel = True
        except ImportError:
            has_nibabel = False
            pytest.skip("nibabel not installed, skipping NIfTI test")
        
        if has_nibabel:
            x, y, z, strengths = sample_data
            subject_id = "test_subject"
            
            # Run calculation with NIfTI export enabled
            calculate_analytical_gradient(
                x=x, y=y, z=z, strengths=strengths,
                subject_id=subject_id,
                output_dir=temp_dir,
                average=True,
                export_nifti=True
            )
            
            # Check that NIfTI files were created
            subject_dir = temp_dir / subject_id
            gradient_dir = subject_dir / "Analytical_FFT_Gradient_Maps"
            all_timepoints_dir = gradient_dir / "AllTimePoints"
            
            average_nifti = gradient_dir / "average_gradient.nii.gz"
            timepoint_niftis = list(all_timepoints_dir.glob("gradient_map_t*.nii.gz"))
            
            assert average_nifti.exists()
            assert len(timepoint_niftis) == strengths.shape[0]
    
    def test_grid_dimensions(self, sample_data, temp_dir):
        """Test specifying grid dimensions."""
        x, y, z, strengths = sample_data
        
        # Specify grid dimensions
        nx, ny, nz = 32, 32, 32
        
        # Run calculation with specified grid dimensions
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            nx=nx, ny=ny, nz=nz,
            estimate_grid=False,
            output_dir=None
        )
        
        # Check that grid dimensions match what we specified
        assert results['coordinates']['nx'] == nx
        assert results['coordinates']['ny'] == ny
        assert results['coordinates']['nz'] == nz
        assert results['gradient_map_grid'].shape[1:] == (nx, ny, nz)
    
    def test_analytical_property(self, sample_data):
        """Test that the analytical gradient behaves as expected.
        
        For a simple spherical harmonic function, we can verify that
        the gradient magnitude follows expected patterns.
        """
        # Replace strengths with a simple spherical harmonic
        n_points = 100
        n_trans = 2
        
        # Create coordinates in range [-1, 1]
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.random.uniform(-1, 1, n_points)
        
        # Calculate radius
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Create simple spherical function: e^(-r²)
        f = np.exp(-r**2)
        
        # Analytical gradient magnitude of e^(-r²) is 2r*e^(-r²)
        expected_gradient_mag = 2 * r * np.exp(-r**2)
        
        # Create 2 time points with the same function
        strengths = np.array([f, f])
        
        # Run calculation
        results = calculate_analytical_gradient(
            x=x, y=y, z=z, strengths=strengths,
            output_dir=None
        )
        
        # Check that gradient has the expected form (approximately)
        # Note: The numerical calculation won't be exact, but should be similar
        computed_gradient = results['gradient_map_nu'][0]  # First time point
        
        # Check correlation coefficient (should be high)
        correlation = np.corrcoef(computed_gradient, expected_gradient_mag)[0, 1]
        assert correlation > 0.5, f"Should be reasonably correlated with expected result, got {correlation:.4f}"
        
    def test_optimal_kspace_calculation(self):
        """Test that automatic k-space calculation produces numerically accurate results."""
        # Create a test case with known analytical gradient solution
        # Use a gaussian function with finer spatial structure
        n_points = 500  # Use more points for better accuracy
        
        # Create non-uniform grid spanning [-2, 2] in each dimension
        # Make points denser in center to capture finer structure
        np.random.seed(42)  # For reproducibility
        r_raw = np.sqrt(np.random.uniform(0, 1, n_points))  # Non-uniform radial distribution
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        
        # Convert to Cartesian coordinates with denser points near center
        x = 2 * r_raw * np.sin(phi) * np.cos(theta)
        y = 2 * r_raw * np.sin(phi) * np.sin(theta)
        z = 2 * r_raw * np.cos(phi)
        
        # Create a function with multiple spatial frequencies
        # f(r) = exp(-r²) + 0.3*exp(-4r²)
        r = np.sqrt(x**2 + y**2 + z**2)
        f_slow = np.exp(-r**2)  # Slower variation (lower frequency)
        f_fast = 0.3 * np.exp(-4*r**2)  # Faster variation (higher frequency)
        f = f_slow + f_fast
        
        # Analytical gradient magnitude
        # |∇f| = |2r*exp(-r²) + 2.4r*exp(-4r²)|
        expected_gradient_mag = np.abs(2*r*np.exp(-r**2) + 2.4*r*np.exp(-4*r**2))
        
        # Run computation with increasing upsampling factors
        upsampling_factors = [1.0, 2.0, 4.0]
        correlations = []
        mse_errors = []
        
        for factor in upsampling_factors:
            results = calculate_analytical_gradient(
                x=x, y=y, z=z, strengths=f,
                output_dir=None,
                average=False,
                upsampling_factor=factor
            )
            
            # Extract computed gradient and grid dimensions
            computed_gradient = results['gradient_map_nu'][0]  # Single time point
            grid_dims = (
                results['coordinates']['nx'],
                results['coordinates']['ny'],
                results['coordinates']['nz']
            )
            
            # Normalize both gradients to [0,1] range for fair comparison
            # This eliminates scale differences which can cause huge relative errors
            norm_expected = (expected_gradient_mag - expected_gradient_mag.min()) / (expected_gradient_mag.max() - expected_gradient_mag.min())
            norm_computed = (computed_gradient - computed_gradient.min()) / (computed_gradient.max() - computed_gradient.min())
            
            # Calculate correlation and mean squared error
            correlation = np.corrcoef(norm_computed, norm_expected)[0, 1]
            mse = np.mean((norm_computed - norm_expected)**2)
            
            correlations.append(correlation)
            mse_errors.append(mse)
            
            # Print info for debugging
            print(f"Upsampling factor {factor}: Grid dims {grid_dims}, " 
                  f"Correlation {correlation:.4f}, MSE {mse:.4f}")
        
        # Verify that accuracy improves with higher upsampling factors
        assert correlations[-1] > 0.85, "Correlation should be high with highest upsampling"
        
        # Verify that error decreases with higher upsampling
        assert mse_errors[-1] < 0.1, "Mean squared error should be low with highest upsampling"
        
        # Check that the estimated k-space parameters are included in results
        assert 'k_space_info' in results
        assert 'max_k' in results['k_space_info']
        assert 'k_resolution' in results['k_space_info']
        assert 'min_spatial_distance' in results['k_space_info']
        assert 'max_freq_required' in results['k_space_info']
        
        # Verify that max_k is sufficient for the calculation
        # The highest frequency in our test function comes from exp(-4r²) term
        # with characteristic frequency roughly 2/√4 = 1
        assert results['k_space_info']['max_k'] > 2.0, f"K-space extent ({results['k_space_info']['max_k']:.4f}) should cover highest frequencies" 