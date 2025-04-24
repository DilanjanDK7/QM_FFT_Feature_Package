import numpy as np
import pytest
from pathlib import Path
import sys
import shutil
import time
import h5py

# Ensure the package directory is in the Python path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

from QM_FFT_Analysis.utils.preprocessing import get_normalized_wavefunctions_at_times
from QM_FFT_Analysis.utils.map_builder import MapBuilder
from QM_FFT_Analysis.utils.map_analysis import (
    calculate_magnitude, calculate_phase, 
    calculate_local_variance, calculate_temporal_difference
)

@pytest.fixture
def setup_teardown():
    """Set up test data and clean up after tests."""
    # Create test output directory
    output_dir = Path("tests/data/integration_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    n_sources = 20
    n_time = 100
    n_selected_times = 5
    
    # Generate time series data
    time_series_data = np.random.randn(n_sources, n_time)
    
    # Select time points
    time_indices = np.linspace(0, n_time-1, n_selected_times, dtype=int)
    
    # Generate coordinates
    x = np.random.uniform(-np.pi, np.pi, n_sources)
    y = np.random.uniform(-np.pi, np.pi, n_sources)
    z = np.random.uniform(-np.pi, np.pi, n_sources)
    
    # Store test data
    data = {
        "output_dir": output_dir,
        "time_series_data": time_series_data,
        "time_indices": time_indices,
        "x": x,
        "y": y,
        "z": z,
        "n_sources": n_sources,
        "n_time": n_time,
        "n_selected_times": n_selected_times
    }
    
    yield data
    
    # Clean up
    if output_dir.exists():
        shutil.rmtree(output_dir)

def test_preprocessing_to_mapbuilder(setup_teardown):
    """Test the integration between preprocessing and MapBuilder."""
    data = setup_teardown
    map_builder = None # Initialize to None
    try:
    # Step 1: Preprocess the time series data
    normalized_wavefunctions = get_normalized_wavefunctions_at_times(
        data["time_series_data"],
        data["time_indices"],
        time_axis=1,
        source_axis=0
    )
    
    # Verify preprocessing output
    assert normalized_wavefunctions.shape == (data["n_selected_times"], data["n_sources"])
    assert np.iscomplexobj(normalized_wavefunctions)
    
    # Check normalization for each time point
    for i in range(data["n_selected_times"]):
        total_prob = np.sum(np.abs(normalized_wavefunctions[i])**2)
        assert np.isclose(total_prob, 1.0)
    
    # Step 2: Initialize MapBuilder with the normalized wavefunctions
    subject_id = "integration_test"
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=data["output_dir"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
        strengths=normalized_wavefunctions,
        eps=1e-6,
        dtype='complex128',
        estimate_grid=True,
        normalize_fft_result=True
    )
    
    # Step 3: Run the pipeline steps individually
        # Compute forward FFT and run mask generation and analysis
        analysis_types = ['magnitude', 'phase', 'local_variance']
        k_var = 5
        # Note that calculate_local_variance=True ensures local variance is actually computed
        calculate_local_var = True
        map_builder.process_map(n_centers=2, radius=0.3, analyses_to_run=analysis_types, 
                             k_neighbors_local_var=k_var, calculate_local_variance=calculate_local_var)
        
        # Check results exist
    assert map_builder.fft_result is not None
    assert map_builder.fft_result.shape == (data["n_selected_times"], map_builder.nx, map_builder.ny, map_builder.nz)
        assert 'forward_fft' in map_builder.data_file # Check HDF5 dataset

        # Check masks were created
    assert len(map_builder.kspace_masks) == 2
        assert 'kspace_mask_0' in map_builder.data_file # Check HDF5 dataset

        # Check inverse maps were computed
    assert len(map_builder.inverse_maps) == 2
    assert map_builder.inverse_maps[0].shape == (data["n_selected_times"], data["n_sources"])
        assert 'inverse_map_0' in map_builder.data_file # Check HDF5 dataset

        # Check gradients were computed
    assert len(map_builder.gradient_maps) == 2
    assert map_builder.gradient_maps[0].shape == (data["n_selected_times"], map_builder.nx, map_builder.ny, map_builder.nz)
        assert 'gradient_map_0' in map_builder.data_file # Check HDF5 dataset

    # Verify analysis output (basic checks)
    assert 'map_0' in map_builder.analysis_results
    assert 'magnitude' in map_builder.analysis_results['map_0']
    assert map_builder.analysis_results['map_0']['magnitude'].shape == (data["n_selected_times"], data["n_sources"])
    assert not np.iscomplexobj(map_builder.analysis_results['map_0']['magnitude'])
    assert 'phase' in map_builder.analysis_results['map_0']
            
        # Only check for local_variance if we requested to calculate it
        if calculate_local_var:
            assert f'local_variance_k{k_var}' in map_builder.analysis_results['map_0']

        # Check HDF5 analysis file for summary group
        assert 'analysis_summary' in map_builder.analysis_file, "analysis_summary group missing in analysis HDF5 file"
        assert 'map_0' in map_builder.analysis_file['analysis_summary'], "map_0 group missing in analysis_summary"
        assert 'magnitude' in map_builder.analysis_file['analysis_summary']['map_0'], "magnitude dataset missing in map_0 group"
        assert f'map_0_local_variance_k{k_var}' in map_builder.analysis_file # Check dataset in HDF5 root
        
    finally:
        # Ensure files are closed even if tests fail
        if map_builder:
            if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
            if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
            if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_mapbuilder_to_analysis(setup_teardown):
    """Test the integration between MapBuilder and analysis functions."""
    data = setup_teardown
    map_builder = None # Initialize to None
    try:
    # Initialize MapBuilder with random strengths
    subject_id = "integration_test_analysis"
        # Use n_selected_times for this test, as it's the dimension passed to MapBuilder
    strengths = np.random.randn(data["n_selected_times"], data["n_sources"]) + 1j * np.random.randn(data["n_selected_times"], data["n_sources"])
    
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=data["output_dir"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
        strengths=strengths,
        eps=1e-6,
        dtype='complex128',
        estimate_grid=True,
        normalize_fft_result=True
    )
    
        # Run the full pipeline instead of individual steps
        analysis_types = ['magnitude', 'phase', 'local_variance', 'temporal_diff_magnitude', 'temporal_diff_phase']
        k_var = 3
        # Note that calculate_local_variance=True ensures local variance is actually computed
        calculate_local_var = True
        map_builder.process_map(n_centers=3, radius=0.4, analyses_to_run=analysis_types, 
                             k_neighbors_local_var=k_var, calculate_local_variance=calculate_local_var)
            
        # Generate additional masks (we already have 3 from process_map)
    map_builder.generate_cubic_mask(kx_min=-0.2, kx_max=0.2, ky_min=-0.2, ky_max=0.2, kz_min=-0.2, kz_max=0.2) # Specific cubic mask 1
    map_builder.generate_cubic_mask(kx_min=-0.4, kx_max=0.4, ky_min=-0.4, ky_max=0.4, kz_min=-0.4, kz_max=0.4) # Specific cubic mask 2
    assert len(map_builder.kspace_masks) == 5 # 3 spherical + 2 cubic

        # Compute inverse maps for the additional masks
        # Note: We need to preserve the existing inverse maps from process_map
        existing_inverse_maps = map_builder.inverse_maps.copy()
        map_builder.inverse_maps = []  # Clear to compute just for new masks
    map_builder.compute_inverse_maps()
        # Combine old and new inverse maps
        map_builder.inverse_maps = existing_inverse_maps + map_builder.inverse_maps
        
        # Check that we have the expected number of masks in total:
        # - 3 spherical masks from process_map
        # - 2 cubic masks added manually
        # - Each mask generates an inverse map
        assert len(map_builder.kspace_masks) == 5 # 3 spherical + 2 cubic
        assert len(map_builder.inverse_maps) == 8  # 3 from process_map + 5 from second compute_inverse_maps call

        # Analyze just the additional maps (last two)
        for i in range(3, 5):
            inv_map_nu = map_builder.inverse_maps[i]
            map_name_base = f"map_{i}"
            
            # Create analysis entries for these maps
            analysis_set = {}
            
            # Standard analyses
            magnitude = calculate_magnitude(inv_map_nu)
            analysis_set['magnitude'] = magnitude
            map_builder._save_to_hdf5(map_builder.analysis_file, f"{map_name_base}_magnitude", magnitude)
            
            phase = calculate_phase(inv_map_nu)
            analysis_set['phase'] = phase
            map_builder._save_to_hdf5(map_builder.analysis_file, f"{map_name_base}_phase", phase)
            
            # Store in analysis_results
            map_builder.analysis_results[map_name_base] = analysis_set
    finally:
        # Ensure files are closed even if tests fail
        if map_builder:
            if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
            if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
            if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close()

def test_full_pipeline_performance(setup_teardown):
    """Test the performance of the full processing pipeline."""
    data = setup_teardown
    map_builder = None # Initialize to None
    try:
    # Initialize MapBuilder
        subject_id = "performance_test"
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=data["output_dir"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
            strengths=np.random.randn(data["n_selected_times"], data["n_sources"]) + 1j * np.random.randn(data["n_selected_times"], data["n_sources"]) # Use selected times dim
    )
    
        # Run the full pipeline and measure time
    start_time = time.time()
        map_builder.process_map(n_centers=2, radius=0.4, analyses_to_run=['magnitude'])
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nFull pipeline duration: {duration:.2f} seconds")
    
        # Basic check that results were generated
        assert len(map_builder.inverse_maps) > 0
        assert len(map_builder.gradient_maps) > 0
        assert len(map_builder.analysis_results) > 0
        assert 'analysis_summary' in map_builder.analysis_file # Check summary group exists

        # Define an acceptable time limit (e.g., 10 seconds for this small test)
        # This will depend heavily on the machine and test data size
        assert duration < 30.0, "Full pipeline took longer than expected."
    finally:
        # Ensure files are closed even if tests fail
        if map_builder:
            if hasattr(map_builder, 'data_file') and map_builder.data_file: map_builder.data_file.close()
            if hasattr(map_builder, 'analysis_file') and map_builder.analysis_file: map_builder.analysis_file.close()
            if hasattr(map_builder, 'enhanced_file') and map_builder.enhanced_file: map_builder.enhanced_file.close() 