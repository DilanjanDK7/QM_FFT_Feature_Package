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
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Verify enhanced features are enabled
    assert map_builder.enable_enhanced_features is True
    assert hasattr(map_builder, 'enhanced_dir')
    assert map_builder.enhanced_dir.exists()

def test_analytical_gradient(setup_teardown):
    """Test the analytical gradient computation."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Run the processing with analytical gradient
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
    map_builder.compute_inverse_maps()
    
    # Compute gradients using the analytical method
    map_builder.compute_gradient_maps(use_analytical_method=True)
    
    # Verify results
    assert len(map_builder.analytical_gradient_maps) > 0
    assert len(map_builder.gradient_maps) > 0
    
    # Check that analytical gradient map files were created
    assert (map_builder.enhanced_dir / "analytical_gradient_map_0.npy").exists()
    
    # Check that compatible gradient map files were created
    assert (map_builder.data_dir / "gradient_map_0.npy").exists()

def test_spectral_slope(setup_teardown):
    """Test the spectral slope calculation."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Run the forward FFT
    map_builder.compute_forward_fft()
    
    # Compute spectral slope
    enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['spectral_slope'])
    
    # Verify results
    assert 'spectral_slope' in enhanced_metrics
    assert enhanced_metrics['spectral_slope'].shape == (data["n_time"],)
    assert (map_builder.enhanced_dir / "spectral_slope.npy").exists()

def test_spectral_entropy(setup_teardown):
    """Test the spectral entropy calculation."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Run the forward FFT
    map_builder.compute_forward_fft()
    
    # Compute spectral entropy
    enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['spectral_entropy'])
    
    # Verify results
    assert 'spectral_entropy' in enhanced_metrics
    assert enhanced_metrics['spectral_entropy'].shape == (data["n_time"],)
    assert (map_builder.enhanced_dir / "spectral_entropy.npy").exists()
    
    # Spectral entropy should be between 0 and log2(nbins)
    nbins = 64  # Default value
    max_entropy = np.log2(nbins)
    assert np.all(enhanced_metrics['spectral_entropy'] >= 0)
    assert np.all(enhanced_metrics['spectral_entropy'] <= max_entropy)

def test_kspace_anisotropy(setup_teardown):
    """Test the k-space anisotropy calculation."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Run the forward FFT
    map_builder.compute_forward_fft()
    
    # Compute anisotropy
    enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['anisotropy'])
    
    # Verify results
    assert 'anisotropy' in enhanced_metrics
    assert enhanced_metrics['anisotropy'].shape == (data["n_time"],)
    assert (map_builder.enhanced_dir / "anisotropy.npy").exists()
    
    # Anisotropy should be between 0 (isotropic) and 1 (fully anisotropic)
    assert np.all(enhanced_metrics['anisotropy'] >= 0)
    assert np.all(enhanced_metrics['anisotropy'] <= 1)

def test_higher_order_moments(setup_teardown):
    """Test the higher-order moments calculation."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
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
    
    # Run the forward FFT and inverse map computation
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
    map_builder.compute_inverse_maps()
    
    # Compute higher-order moments
    enhanced_metrics = map_builder.compute_enhanced_metrics(metrics_to_run=['higher_moments'])
    
    # Verify results
    assert 'map_0' in enhanced_metrics
    assert 'skewness' in enhanced_metrics['map_0']
    assert 'kurtosis' in enhanced_metrics['map_0']
    
    assert enhanced_metrics['map_0']['skewness'].shape == (data["n_time"],)
    assert enhanced_metrics['map_0']['kurtosis'].shape == (data["n_time"],)
    
    assert (map_builder.enhanced_dir / "map_0_skewness.npy").exists()
    assert (map_builder.enhanced_dir / "map_0_kurtosis.npy").exists()

def test_excitation_map(setup_teardown):
    """Test the excitation map calculation."""
    data = setup_teardown
    
    # Skip test if n_time is not sufficient
    if data["n_time"] < 3:
        pytest.skip("Not enough time points for HRF deconvolution")
    
    # Initialize MapBuilder with enhanced features enabled
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
    assert (map_builder.enhanced_dir / "excitation_map.npy").exists()

def test_canonical_hrf():
    """Test the canonical HRF generation."""
    # Generate HRF with different parameters
    hrf1 = generate_canonical_hrf(duration=20, tr=0.5)
    hrf2 = generate_canonical_hrf(duration=30, tr=1.0)
    
    # Check shapes
    assert len(hrf1) == 40  # duration/tr = 20/0.5 = 40
    assert len(hrf2) == 30  # duration/tr = 30/1.0 = 30
    
    # Check normalization
    assert np.isclose(np.max(hrf1), 1.0)
    assert np.isclose(np.max(hrf2), 1.0)
    
    # HRF should start at 0, rise, and then fall back below zero
    assert np.isclose(hrf1[0], 0, atol=1e-2)
    assert np.any(hrf1 > 0.5)  # Should have a distinct peak
    assert np.any(hrf1 < 0)     # Should go negative at some point

def test_process_map_with_enhanced_features(setup_teardown):
    """Test the full processing pipeline with enhanced features."""
    data = setup_teardown
    
    # Initialize MapBuilder with enhanced features enabled
    subject_id = "full_pipeline_test"
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=data["output_dir"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
        strengths=data["strengths"],
        enable_enhanced_features=True
    )
    
    # Run the full processing pipeline with enhanced analyses
    analyses = [
        'magnitude', 'phase',  # Standard analyses
        'spectral_slope', 'spectral_entropy', 'anisotropy'  # Enhanced analyses
    ]
    
    map_builder.process_map(
        n_centers=1,
        radius=0.5,
        analyses_to_run=analyses,
        use_analytical_gradient=True
    )
    
    # Verify standard results
    assert len(map_builder.inverse_maps) > 0
    assert len(map_builder.gradient_maps) > 0
    assert len(map_builder.analysis_results) > 0
    
    # Verify enhanced results
    assert len(map_builder.enhanced_results) > 0
    assert 'spectral_slope' in map_builder.enhanced_results
    assert 'spectral_entropy' in map_builder.enhanced_results
    assert 'anisotropy' in map_builder.enhanced_results
    
    # Verify output files
    assert (map_builder.enhanced_dir / "spectral_slope.npy").exists()
    assert (map_builder.enhanced_dir / "spectral_entropy.npy").exists()
    assert (map_builder.enhanced_dir / "anisotropy.npy").exists()
    assert (map_builder.enhanced_dir / "analytical_gradient_map_0.npy").exists()
    
    # Verify legacy output format is maintained
    assert (map_builder.analysis_dir / "map_0_magnitude.npy").exists()
    assert (map_builder.analysis_dir / "map_0_phase.npy").exists()
    assert (map_builder.data_dir / "gradient_map_0.npy").exists()

def test_enhanced_features_backward_compatibility(setup_teardown):
    """Test that existing code continues to work without enabling enhanced features."""
    data = setup_teardown
    
    # Initialize MapBuilder WITHOUT enhanced features enabled (default)
    subject_id = "backward_compat_test"
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=data["output_dir"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
        strengths=data["strengths"],
        # enable_enhanced_features=False  # Default, not needed
    )
    
    # Verify enhanced features are disabled
    assert map_builder.enable_enhanced_features is False
    
    # Run the standard processing pipeline
    map_builder.process_map(
        n_centers=1,
        radius=0.5,
        analyses_to_run=['magnitude', 'phase']
    )
    
    # Verify results still work as expected
    assert len(map_builder.inverse_maps) > 0
    assert len(map_builder.gradient_maps) > 0
    assert len(map_builder.analysis_results) > 0
    
    # Verify output files in standard locations
    assert (map_builder.analysis_dir / "map_0_magnitude.npy").exists()
    assert (map_builder.analysis_dir / "map_0_phase.npy").exists()
    assert (map_builder.data_dir / "gradient_map_0.npy").exists()
    
    # Enhanced directory should not exist
    assert not hasattr(map_builder, 'enhanced_results')
    # Test if backwards compatibility maintained with analytical_gradient
    with pytest.raises(Exception):
        map_builder.compute_gradient_maps(use_analytical_method=True) 