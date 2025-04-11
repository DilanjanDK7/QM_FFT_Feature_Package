import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure the package directory is in the Python path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

from QM_FFT_Analysis.utils.map_analysis import (
    calculate_magnitude, calculate_phase, 
    calculate_local_variance, calculate_temporal_difference
)

@pytest.fixture
def sample_inverse_map():
    """Provides sample inverse map data for testing."""
    n_trans = 3
    n_points = 50
    # Create complex data with varying magnitudes and phases
    data = np.zeros((n_trans, n_points), dtype=np.complex128)
    for i in range(n_trans):
        # Vary the magnitude and phase for each transform
        magnitude = np.exp(-np.arange(n_points) / 10) * (1 + i * 0.2)
        phase = np.arange(n_points) * 0.1 + i * np.pi / 4
        data[i] = magnitude * np.exp(1j * phase)
    return data

@pytest.fixture
def sample_points():
    """Provides sample 3D point coordinates for testing."""
    n_points = 50
    # Generate points in a 3D grid-like pattern
    x = np.linspace(-1, 1, int(np.ceil(n_points**(1/3))))
    y = np.linspace(-1, 1, int(np.ceil(n_points**(1/3))))
    z = np.linspace(-1, 1, int(np.ceil(n_points**(1/3))))
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    # Trim to exact n_points if needed
    return points[:n_points]

def test_calculate_magnitude(sample_inverse_map):
    """Test magnitude calculation."""
    magnitude = calculate_magnitude(sample_inverse_map)
    
    # Check shape
    assert magnitude.shape == sample_inverse_map.shape
    # Check type (should be real)
    assert not np.iscomplexobj(magnitude)
    # Check values (should be non-negative)
    assert np.all(magnitude >= 0)
    # Check calculation (should match np.abs)
    expected = np.abs(sample_inverse_map)
    assert np.allclose(magnitude, expected)

def test_calculate_phase(sample_inverse_map):
    """Test phase calculation."""
    phase = calculate_phase(sample_inverse_map)
    
    # Check shape
    assert phase.shape == sample_inverse_map.shape
    # Check type (should be real)
    assert not np.iscomplexobj(phase)
    # Check values (should be in [-pi, pi])
    assert np.all(phase >= -np.pi)
    assert np.all(phase <= np.pi)
    # Check calculation (should match np.angle)
    expected = np.angle(sample_inverse_map)
    assert np.allclose(phase, expected)

def test_calculate_local_variance(sample_inverse_map, sample_points):
    """Test local variance calculation."""
    k = 5  # Number of nearest neighbors
    variance = calculate_local_variance(sample_inverse_map, sample_points, k=k)
    
    # Check shape
    assert variance.shape == sample_inverse_map.shape
    # Check type (should be real)
    assert not np.iscomplexobj(variance)
    # Check values (should be non-negative)
    assert np.all(variance >= 0)
    
    # Test with different k values
    k_values = [3, 7, 10]
    for k_val in k_values:
        if k_val < sample_points.shape[0]:  # Only test if k is less than n_points
            variance_k = calculate_local_variance(sample_inverse_map, sample_points, k=k_val)
            assert variance_k.shape == sample_inverse_map.shape
            assert not np.iscomplexobj(variance_k)
            assert np.all(variance_k >= 0)

def test_calculate_local_variance_edge_cases(sample_inverse_map, sample_points):
    """Test local variance calculation with edge cases."""
    # Test with k >= n_points (should use all points)
    k = sample_points.shape[0]
    variance = calculate_local_variance(sample_inverse_map, sample_points, k=k)
    assert variance.shape == sample_inverse_map.shape
    
    # Test with k = 1 (minimum meaningful value)
    k = 1
    variance = calculate_local_variance(sample_inverse_map, sample_points, k=k)
    assert variance.shape == sample_inverse_map.shape
    
    # Test with very small dataset
    small_map = sample_inverse_map[:, :3]
    small_points = sample_points[:3]
    k = 2
    variance = calculate_local_variance(small_map, small_points, k=k)
    assert variance.shape == small_map.shape

def test_calculate_temporal_difference(sample_inverse_map):
    """Test temporal difference calculation."""
    diff = calculate_temporal_difference(sample_inverse_map)
    
    # Check shape (should be (n_trans-1, n_points))
    expected_shape = (sample_inverse_map.shape[0] - 1, sample_inverse_map.shape[1])
    assert diff.shape == expected_shape
    # Check type (should be complex)
    assert np.iscomplexobj(diff)
    # Check calculation
    expected = sample_inverse_map[1:] - sample_inverse_map[:-1]
    assert np.allclose(diff, expected)

def test_calculate_temporal_difference_edge_cases():
    """Test temporal difference calculation with edge cases."""
    # Test with single transform (should return None)
    single_transform = np.random.randn(1, 10) + 1j * np.random.randn(1, 10)
    diff = calculate_temporal_difference(single_transform)
    assert diff is None
    
    # Test with empty array
    empty_array = np.array([]).reshape(0, 10)
    diff = calculate_temporal_difference(empty_array)
    assert diff is None
    
    # Test with 2D array (should work fine)
    two_transforms = np.random.randn(2, 10) + 1j * np.random.randn(2, 10)
    diff = calculate_temporal_difference(two_transforms)
    assert diff.shape == (1, 10)
    assert np.iscomplexobj(diff) 