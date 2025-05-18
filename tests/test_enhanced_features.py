import numpy as np
import pytest
from pathlib import Path
import sys
import shutil
import time

# Ensure the package directory is in the Python path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

# Import the enhanced features module
try:
    from QM_FFT_Analysis.utils.enhanced_features import (
        compute_radial_gradient, 
        calculate_spectral_slope,
        calculate_spectral_entropy,
        calculate_kspace_anisotropy,
        calculate_higher_order_moments,
        calculate_excitation_map,
        generate_canonical_hrf
    )
    from QM_FFT_Analysis.utils.map_builder import MapBuilder
except ImportError:
    pytest.skip("Enhanced features not available", allow_module_level=True)

@pytest.fixture
def setup_teardown():
    """Set up test data and clean up after tests."""
    # Create test output directory
    output_dir = Path("tests/data/enhanced_features_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    n_sources = 20
    n_time = 5
    
    # Generate coordinates
    rng = np.random.RandomState(seed=42)  # Fixed seed for reproducibility
    x = rng.uniform(-np.pi, np.pi, n_sources)
    y = rng.uniform(-np.pi, np.pi, n_sources)
    z = rng.uniform(-np.pi, np.pi, n_sources)
    
    # Generate complex strengths with a known pattern
    # Create a mixture of frequencies to test spectral features
    k_values = np.linspace(0.5, 2.0, n_time)  # Different frequencies for each timepoint
    strengths = np.zeros((n_time, n_sources), dtype=np.complex128)
    
    for t in range(n_time):
        r = np.sqrt(x**2 + y**2 + z**2)
        # Generate data with known spectral properties for testing
        strengths[t] = np.sin(k_values[t] * r) + 1j * np.cos(k_values[t] * r)
    
    # Store test data
    data = {
        "output_dir": output_dir,
        "x": x,
        "y": y,
        "z": z,
        "strengths": strengths,
        "n_sources": n_sources,
        "n_time": n_time,
        "k_values": k_values
    }
    
    yield data
    
    # Clean up
    if output_dir.exists():
        shutil.rmtree(output_dir)

def test_map_builder_with_enhanced_features(setup_teardown):
    """Test initializing MapBuilder with enhanced features enabled."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "enhanced_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        assert map_builder.enable_enhanced_features is True
        # Check HDF5 file attribute
        assert hasattr(map_builder, 'enhanced_file'), "MapBuilder should have 'enhanced_file' attribute"
        assert map_builder.enhanced_file is not None, "'enhanced_file' attribute should not be None"
        # Check file exists by trying to access its name (will fail if closed/invalid)
        assert Path(map_builder.enhanced_file.filename).exists()
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_analytical_gradient(setup_teardown):
    """Test the analytical gradient computation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "gradient_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        map_builder.compute_forward_fft()
        map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
        map_builder.compute_inverse_maps()
        map_builder.compute_gradient_maps(use_analytical_method=True)

        # Verify results stored correctly
        assert hasattr(map_builder, 'analytical_gradient_maps') # Check attribute exists
        assert len(map_builder.analytical_gradient_maps) > 0
        assert hasattr(map_builder, 'gradient_maps') # Check legacy attribute exists
        assert len(map_builder.gradient_maps) > 0

        # Check HDF5 datasets exist
        assert 'analytical_gradient_map_0' in map_builder.enhanced_file, "Analytical gradient dataset missing in enhanced file"
        # When using analytical method, we don't expect inverse_map_nu_0 in data file
        assert 'analytical_gradient_map_0' in map_builder.enhanced_file, "Analytical gradient dataset missing in enhanced file"
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_spectral_slope(setup_teardown):
    """Test the spectral slope calculation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "spectral_slope_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        map_builder.compute_forward_fft()
        enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['spectral_slope'])

        # Verify results in dictionary and HDF5
        assert 'spectral_slope' in enhanced_metrics
        assert enhanced_metrics['spectral_slope'].shape == (data["n_time"],)
        assert 'spectral_slope' in map_builder.enhanced_file, "Spectral slope dataset missing"
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_spectral_entropy(setup_teardown):
    """Test the spectral entropy calculation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "spectral_entropy_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        map_builder.compute_forward_fft()
        enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['spectral_entropy'])

        # Verify results
        assert 'spectral_entropy' in enhanced_metrics
        assert enhanced_metrics['spectral_entropy'].shape == (data["n_time"],)
        assert 'spectral_entropy' in map_builder.enhanced_file, "Spectral entropy dataset missing"

        nbins = 64 # Default value
        max_entropy = np.log2(nbins)
        assert np.all(enhanced_metrics['spectral_entropy'] >= 0)
        assert np.all(enhanced_metrics['spectral_entropy'] <= max_entropy)
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_kspace_anisotropy(setup_teardown):
    """Test the k-space anisotropy calculation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "anisotropy_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        map_builder.compute_forward_fft()
        enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['anisotropy'])

        # Verify results
        assert 'anisotropy' in enhanced_metrics
        assert enhanced_metrics['anisotropy'].shape == (data["n_time"],)
        assert 'anisotropy' in map_builder.enhanced_file, "Anisotropy dataset missing"

        assert np.all(enhanced_metrics['anisotropy'] >= 0)
        assert np.all(enhanced_metrics['anisotropy'] <= 1)
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_higher_order_moments(setup_teardown):
    """Test the higher-order moments calculation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "moments_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        map_builder.compute_forward_fft()
        map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
        map_builder.compute_inverse_maps()
        enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['higher_moments'])

        # Verify results in dictionary
        assert 'map_0' in enhanced_metrics # Check if map specific dict exists
        assert 'skewness' in enhanced_metrics['map_0']
        assert 'kurtosis' in enhanced_metrics['map_0']
        assert enhanced_metrics['map_0']['skewness'].shape == (data["n_time"],)
        assert enhanced_metrics['map_0']['kurtosis'].shape == (data["n_time"],)

        # Verify HDF5 datasets exist (saved individually now)
        assert 'map_0_skewness' in map_builder.enhanced_file, "Skewness dataset missing"
        assert 'map_0_kurtosis' in map_builder.enhanced_file, "Kurtosis dataset missing"
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_excitation_map(setup_teardown):
    """Test the excitation map calculation."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "excitation_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        
        # Run the forward FFT
        map_builder.compute_forward_fft()
        
        # Compute excitation map
        enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['excitation'])
        
        # Verify results
        assert 'excitation_map' in enhanced_metrics
        assert enhanced_metrics['excitation_map'].shape == map_builder.fft_result.shape
        assert 'excitation_map' in map_builder.enhanced_file, "Excitation map dataset missing"
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_canonical_hrf():
    """Test the HRF generation."""
    hrf = generate_canonical_hrf(duration=20, tr=1.0)
    assert hrf.shape == (20,)
    assert np.max(hrf) == 1.0

def test_process_map_with_enhanced_features(setup_teardown):
    """Test the full process_map pipeline with enhanced features."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "process_enhanced_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=True
        )
        
        # Define analyses to run, including enhanced ones
        analyses = ['magnitude', 'phase', 'local_variance', 
                    'spectral_slope', 'spectral_entropy', 'anisotropy', 
                    'higher_moments', 'excitation']
        
        # Run the full pipeline
        map_builder.process_map(n_centers=1, radius=0.5, analyses_to_run=analyses)
        
        # Verify standard outputs exist
        assert 'forward_fft' in map_builder.data_file
        assert 'kspace_mask_0' in map_builder.data_file
        assert 'inverse_map_0' in map_builder.data_file
        
        # Check for either analytical gradient or non-uniform gradient
        if hasattr(map_builder, 'analytical_gradient_maps') and len(map_builder.analytical_gradient_maps) > 0:
            assert 'analytical_gradient_map_0' in map_builder.enhanced_file
        else:
            assert 'inverse_map_nu_0' in map_builder.data_file
            
        assert 'analysis_summary' in map_builder.analysis_file
        assert 'map_0' in map_builder.analysis_results # Check summary dict
        assert 'magnitude' in map_builder.analysis_results['map_0']
        
        # Verify enhanced outputs exist in HDF5 file
        assert 'spectral_slope' in map_builder.enhanced_file
        assert 'spectral_entropy' in map_builder.enhanced_file
        assert 'anisotropy' in map_builder.enhanced_file
        assert 'map_0_skewness' in map_builder.enhanced_file
        assert 'map_0_kurtosis' in map_builder.enhanced_file
        assert 'excitation_map' in map_builder.enhanced_file
        # Verify enhanced results are also in the returned/stored dictionary
        assert 'enhanced' in map_builder.analysis_results
        assert 'spectral_slope' in map_builder.analysis_results['enhanced']
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_enhanced_features_backward_compatibility(setup_teardown):
    """Test that disabling enhanced features works and doesn't create enhanced files/results."""
    data = setup_teardown
    map_builder = None
    try:
        subject_id = "backward_compat_test"
        map_builder = MapBuilder(
            subject_id=subject_id,
            output_dir=data["output_dir"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            enable_enhanced_features=False # Explicitly disable
        )
        
        # Verify enhanced features are disabled
        assert map_builder.enable_enhanced_features is False
        assert not hasattr(map_builder, 'enhanced_results'), "'enhanced_results' attribute should not exist"
        assert not hasattr(map_builder, 'enhanced_file'), "'enhanced_file' attribute should not exist"
        
        # Run standard processing
        analyses = ['magnitude', 'phase']
        map_builder.process_map(n_centers=1, radius=0.5, analyses_to_run=analyses)
        
        # Verify standard outputs exist
        assert 'forward_fft' in map_builder.data_file
        assert 'kspace_mask_0' in map_builder.data_file
        assert 'inverse_map_0' in map_builder.data_file
        assert 'inverse_map_nu_0' in map_builder.data_file
        
        # Verify enhanced file does NOT exist
        assert not (map_builder.subject_dir / 'enhanced.h5').exists()
        
    finally:
        if map_builder:
             if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
             if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
             # Don't try to close enhanced_file as it shouldn't exist
             # if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close() 