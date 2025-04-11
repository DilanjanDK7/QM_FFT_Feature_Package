import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure the package directory is in the Python path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

from QM_FFT_Analysis.utils.preprocessing import get_normalized_wavefunction_at_time, hilbert_transform_axis, get_normalized_wavefunctions_at_times

@pytest.fixture
def sample_data():
    """Provides sample time series data for testing."""
    n_sources = 5
    n_time = 50
    data = np.random.randn(n_sources, n_time)
    return data

def test_hilbert_transform_axis_shape_and_type(sample_data):
    """Test Hilbert transform output shape and type."""
    data = sample_data
    # Test on time axis (last axis)
    transformed = hilbert_transform_axis(data, axis=-1)
    assert transformed.shape == data.shape
    assert np.iscomplexobj(transformed)
    # Test on source axis (first axis)
    transformed_alt = hilbert_transform_axis(data, axis=0)
    assert transformed_alt.shape == data.shape
    assert np.iscomplexobj(transformed_alt)

def test_hilbert_transform_axis_complex_input():
    """Test Hilbert transform with complex input data."""
    n_sources = 5
    n_time = 50
    # Create complex data
    data = np.random.randn(n_sources, n_time) + 1j * np.random.randn(n_sources, n_time)
    
    # Apply Hilbert transform
    transformed = hilbert_transform_axis(data, axis=-1)
    
    # Check shape and type
    assert transformed.shape == data.shape
    assert np.iscomplexobj(transformed)
    
    # Check that real part was used for transform
    # The imaginary part should be different from the original
    assert not np.allclose(transformed.imag, data.imag)

def test_get_normalized_wavefunction_basic(sample_data):
    """Test basic functionality and normalization."""
    data = sample_data
    time_index = 10
    time_axis = 1
    source_axis = 0

    normalized_wf = get_normalized_wavefunction_at_time(
        data, time_index, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape - should be shape of sources dimension
    expected_shape = (data.shape[source_axis],)
    assert normalized_wf.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wf)
    # Check normalization
    total_prob = np.sum(np.abs(normalized_wf)**2)
    assert np.isclose(total_prob, 1.0)

def test_get_normalized_wavefunction_alt_axes(sample_data):
    """Test with alternative time and source axes."""
    data = sample_data.T # Shape (time, sources)
    time_index = 10
    time_axis = 0
    source_axis = 1

    normalized_wf = get_normalized_wavefunction_at_time(
        data, time_index, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape - should be shape of sources dimension
    expected_shape = (data.shape[source_axis],)
    assert normalized_wf.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wf)
    # Check normalization
    total_prob = np.sum(np.abs(normalized_wf)**2)
    assert np.isclose(total_prob, 1.0)

def test_get_normalized_wavefunction_zero_input():
    """Test with input data that is all zeros."""
    n_sources = 5
    n_time = 50
    data = np.zeros((n_sources, n_time))
    time_index = 10
    time_axis = 1
    source_axis = 0

    normalized_wf = get_normalized_wavefunction_at_time(
        data, time_index, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape
    expected_shape = (data.shape[source_axis],)
    assert normalized_wf.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wf)
    # Check values (should be zero, as norm factor becomes 1)
    assert np.allclose(normalized_wf, 0)
    # Check normalization (sum should be 0)
    total_prob = np.sum(np.abs(normalized_wf)**2)
    assert np.isclose(total_prob, 0.0)

def test_get_normalized_wavefunction_invalid_time_index(sample_data):
    """Test with an out-of-bounds time index."""
    data = sample_data
    time_index = data.shape[1] # Index equals size, so it's out of bounds
    with pytest.raises(IndexError):
        get_normalized_wavefunction_at_time(data, time_index, time_axis=1, source_axis=0)

def test_get_normalized_wavefunction_invalid_dims():
    """Test with invalid input dimensions (less than 2D)."""
    data_1d = np.random.randn(50)
    with pytest.raises(ValueError, match="Input data must be at least 2D."):
        get_normalized_wavefunction_at_time(data_1d, 10, time_axis=0, source_axis=0) # Axis doesn't matter here

def test_get_normalized_wavefunction_same_axes():
    """Test when time_axis and source_axis are the same."""
    data = np.random.randn(5, 50)
    with pytest.raises(ValueError, match="time_axis and source_axis cannot be the same."):
        get_normalized_wavefunction_at_time(data, 10, time_axis=1, source_axis=1)

# New tests for get_normalized_wavefunctions_at_times
def test_get_normalized_wavefunctions_at_times_basic(sample_data):
    """Test basic functionality of get_normalized_wavefunctions_at_times."""
    data = sample_data
    time_indices = [10, 20, 30]
    time_axis = 1
    source_axis = 0

    normalized_wfs = get_normalized_wavefunctions_at_times(
        data, time_indices, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape - should be (n_time_indices, n_sources)
    expected_shape = (len(time_indices), data.shape[source_axis])
    assert normalized_wfs.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wfs)
    # Check normalization for each time point
    for i in range(len(time_indices)):
        total_prob = np.sum(np.abs(normalized_wfs[i])**2)
        assert np.isclose(total_prob, 1.0)

def test_get_normalized_wavefunctions_at_times_alt_axes():
    """Test get_normalized_wavefunctions_at_times with alternative axis ordering."""
    n_sources = 5
    n_time = 50
    data = np.random.randn(n_time, n_sources)  # Shape (time, sources)
    time_indices = [10, 20, 30]
    time_axis = 0
    source_axis = 1

    normalized_wfs = get_normalized_wavefunctions_at_times(
        data, time_indices, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape - should be (n_time_indices, n_sources)
    expected_shape = (len(time_indices), data.shape[source_axis])
    assert normalized_wfs.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wfs)
    # Check normalization for each time point
    for i in range(len(time_indices)):
        total_prob = np.sum(np.abs(normalized_wfs[i])**2)
        assert np.isclose(total_prob, 1.0)

def test_get_normalized_wavefunctions_at_times_3d():
    """Test get_normalized_wavefunctions_at_times with 3D input data."""
    n_sources = 5
    n_time = 50
    n_extra = 3
    data = np.random.randn(n_sources, n_time, n_extra)
    time_indices = [10, 20, 30]
    time_axis = 1
    source_axis = 0

    normalized_wfs = get_normalized_wavefunctions_at_times(
        data, time_indices, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape - should be (n_time_indices, n_sources, n_extra)
    expected_shape = (len(time_indices), data.shape[source_axis], data.shape[2])
    assert normalized_wfs.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wfs)
    # Check normalization for each time point (sum over source axis)
    for i in range(len(time_indices)):
        total_prob = np.sum(np.abs(normalized_wfs[i])**2, axis=0)
        assert np.allclose(total_prob, 1.0)

def test_get_normalized_wavefunctions_at_times_invalid_indices(sample_data):
    """Test get_normalized_wavefunctions_at_times with invalid time indices."""
    data = sample_data
    time_indices = [10, data.shape[1]]  # Second index is out of bounds
    with pytest.raises(IndexError):
        get_normalized_wavefunctions_at_times(data, time_indices, time_axis=1, source_axis=0)

def test_get_normalized_wavefunctions_at_times_invalid_dims():
    """Test get_normalized_wavefunctions_at_times with invalid input dimensions."""
    data_1d = np.random.randn(50)
    with pytest.raises(ValueError, match="Input data must be at least 2D."):
        get_normalized_wavefunctions_at_times(data_1d, [10, 20], time_axis=0, source_axis=0)

def test_get_normalized_wavefunctions_at_times_same_axes():
    """Test get_normalized_wavefunctions_at_times when time_axis and source_axis are the same."""
    data = np.random.randn(5, 50)
    with pytest.raises(ValueError, match="time_axis and source_axis cannot be the same."):
        get_normalized_wavefunctions_at_times(data, [10, 20], time_axis=1, source_axis=1)

def test_get_normalized_wavefunctions_at_times_zero_input():
    """Test get_normalized_wavefunctions_at_times with input data that is all zeros."""
    n_sources = 5
    n_time = 50
    data = np.zeros((n_sources, n_time))
    time_indices = [10, 20, 30]
    time_axis = 1
    source_axis = 0

    normalized_wfs = get_normalized_wavefunctions_at_times(
        data, time_indices, time_axis=time_axis, source_axis=source_axis
    )

    # Check shape
    expected_shape = (len(time_indices), data.shape[source_axis])
    assert normalized_wfs.shape == expected_shape
    # Check type
    assert np.iscomplexobj(normalized_wfs)
    # Check values (should be zero, as norm factor becomes 1)
    assert np.allclose(normalized_wfs, 0)
    # Check normalization (sum should be 0)
    for i in range(len(time_indices)):
        total_prob = np.sum(np.abs(normalized_wfs[i])**2)
        assert np.isclose(total_prob, 0.0) 