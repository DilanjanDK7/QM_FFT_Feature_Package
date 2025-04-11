import numpy as np
import pytest
from pathlib import Path
import shutil
import time
from QM_FFT_Analysis.utils.map_builder import MapBuilder

@pytest.fixture(params=[1, 3]) # Test with n_trans=1 and n_trans=3
def setup_teardown(request):
    """Set up test data and clean up after tests."""
    n_trans = request.param
    print(f"\nSetting up test for n_trans = {n_trans}")

    # Create test output directory
    output_dir = Path("QM_FFT_Analysis/tests/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test data
    n_points = 50 # Use fewer points for faster tests
    # Use 1D coordinates as expected by modified MapBuilder
    x = np.random.uniform(-np.pi, np.pi, n_points)
    y = np.random.uniform(-np.pi, np.pi, n_points)
    z = np.random.uniform(-np.pi, np.pi, n_points)

    # Generate strength data: shape (n_trans, n_points)
    # Make strengths vary across transforms
    strengths_list = []
    for i in range(n_trans):
        # Example: shift center slightly for each transform
        center_shift = i * 0.2
        str_i = np.exp(-((x - center_shift)**2 + y**2 + z**2) / 0.5) + np.random.randn(n_points) * 0.01 # Add tiny noise
        strengths_list.append(str_i)
    strengths = np.stack(strengths_list)
    if n_trans == 1:
        strengths = strengths.squeeze(0) # Keep 1D for n_trans=1 case input

    # Store test data
    data = {
        "n_trans": n_trans,
        "output_dir": output_dir,
        "x": x,
        "y": y,
        "z": z,
        "strengths": strengths
    }

    yield data

    # Clean up
    if output_dir.exists():
        # Use try-except for robustness in cleanup
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Error cleaning up {output_dir}: {e}")

def test_initialization(setup_teardown):
    """Test MapBuilder initialization."""
    data = setup_teardown

    # Test basic initialization
    map_builder = MapBuilder(
        subject_id=f"test_subject_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Verify initialization
    assert map_builder.subject_id == f"test_subject_nt{data['n_trans']}"
    assert map_builder.output_dir == data["output_dir"]
    assert map_builder.n_trans == data['n_trans']
    assert map_builder.n_points == data['x'].size
    assert map_builder.x_coords_1d.shape == (data['x'].size,)
    assert map_builder.y_coords_1d.shape == (data['y'].size,)
    assert map_builder.z_coords_1d.shape == (data['z'].size,)
    assert map_builder.strengths.shape == (data['n_trans'], data['x'].size)
    assert map_builder.padding == 0
    assert map_builder.stride == 1
    assert map_builder.eps == 1e-6
    assert map_builder.dtype == 'complex128'

    # Test custom initialization
    # Ensure custom strengths match n_trans if provided
    custom_strengths = data["strengths"] if data['n_trans'] == 1 else np.random.rand(data['n_trans'], data['x'].size).astype(np.complex64)
    map_builder_custom = MapBuilder(
        subject_id=f"test_subject_custom_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=custom_strengths,
        padding=1, # Change padding
        stride=2,
        eps=1e-8,
        dtype='complex64'
    )

    assert map_builder_custom.n_trans == data['n_trans']
    assert map_builder_custom.dtype == 'complex64'
    assert map_builder_custom.eps == 1e-8
    assert map_builder_custom.stride == 2
    # Grid size depends on heuristic and padding
    approx_edge = int(np.ceil(map_builder_custom.n_points**(1/3)))
    expected_grid_size = approx_edge * (1 + 1) # padding = 1
    assert map_builder_custom.nx == expected_grid_size

def test_forward_fft(setup_teardown):
    """Test forward FFT computation."""
    data = setup_teardown

    map_builder = MapBuilder(
        subject_id=f"test_forward_fft_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Compute forward FFT
    map_builder.compute_forward_fft()

    # Verify output
    assert map_builder.forward_fft is not None
    # Shape is now (n_trans, nx, ny, nz)
    expected_shape = (data['n_trans'], map_builder.nx, map_builder.ny, map_builder.nz)
    assert map_builder.forward_fft.shape == expected_shape
    assert map_builder.fft_result.shape == expected_shape
    assert (map_builder.data_dir / "forward_fft.npy").exists()

def test_kspace_masks(setup_teardown):
    """Test k-space mask generation."""
    data = setup_teardown
    n_masks_to_test = 2

    map_builder = MapBuilder(
        subject_id=f"test_kspace_masks_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Generate k-space masks
    map_builder.generate_kspace_masks(n_centers=n_masks_to_test, radius=0.5)

    # Verify output
    assert len(map_builder.kspace_masks) == n_masks_to_test
    for i, mask in enumerate(map_builder.kspace_masks):
        # Mask shape is 3D (nx, ny, nz)
        assert mask.shape == (map_builder.nx, map_builder.ny, map_builder.nz)
        assert (map_builder.data_dir / f"kspace_mask_{i}.npy").exists()

def test_inverse_maps(setup_teardown):
    """Test inverse map computation."""
    data = setup_teardown
    n_masks_to_test = 2

    map_builder = MapBuilder(
        subject_id=f"test_inverse_maps_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Compute forward FFT and generate masks
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=n_masks_to_test, radius=0.5)

    # Compute inverse maps
    map_builder.compute_inverse_maps()

    # Verify output
    assert len(map_builder.inverse_maps) == n_masks_to_test
    for i, inverse_map in enumerate(map_builder.inverse_maps):
        # Shape is (n_trans, n_points)
        expected_shape = (data['n_trans'], map_builder.n_points)
        assert inverse_map.shape == expected_shape
        assert (map_builder.data_dir / f"inverse_map_{i}.npy").exists()
        # Plotting is now handled outside the method
        # assert (map_builder.plots_dir / f"inverse_volume_{i}.html").exists()

def test_gradient_maps(setup_teardown):
    """Test gradient map computation."""
    data = setup_teardown
    n_masks_to_test = 2

    map_builder = MapBuilder(
        subject_id=f"test_gradient_maps_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Compute forward FFT and generate masks
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=n_masks_to_test, radius=0.5)
    map_builder.compute_inverse_maps()

    # Compute gradient maps
    map_builder.compute_gradient_maps()

    # Verify output
    assert len(map_builder.gradient_maps) == n_masks_to_test
    for i, gradient_map in enumerate(map_builder.gradient_maps):
        # Shape is (n_trans, nx, ny, nz)
        expected_shape = (data['n_trans'], map_builder.nx, map_builder.ny, map_builder.nz)
        assert gradient_map.shape == expected_shape
        assert (map_builder.data_dir / f"gradient_map_{i}.npy").exists()
        # Plotting is handled outside
        # assert (map_builder.plots_dir / f"gradient_volume_{i}.html").exists()

def test_visualization(setup_teardown):
    """Test visualization function for 3D data."""
    data = setup_teardown

    map_builder = MapBuilder(
        subject_id=f"test_visualization_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Generate 3D test data matching the grid size
    test_3d_data = np.random.rand(map_builder.nx, map_builder.ny, map_builder.nz)

    # Test volume plot generation
    map_builder.generate_volume_plot(
        test_3d_data,
        "test_volume.html",
        opacity=0.7,
        surface_count=20,
        colormap="viridis"
    )

    # Verify output
    assert (map_builder.plots_dir / "test_volume.html").exists()

def test_full_process(setup_teardown):
    """Test the full processing pipeline."""
    data = setup_teardown

    # Test with default settings
    map_builder = MapBuilder(
        subject_id=f"test_full_process_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"]
    )

    # Process map
    map_builder.process_map(n_centers=2, radius=0.4)

    # Verify output
    assert map_builder.forward_fft is not None
    assert len(map_builder.kspace_masks) == 2
    assert len(map_builder.inverse_maps) == 2
    assert len(map_builder.gradient_maps) == 2

def test_error_handling(): # Removed setup_teardown fixture dependency
    """Test error handling for invalid inputs."""
    # Create minimal valid data for error testing
    n_points = 10
    output_dir = Path("QM_FFT_Analysis/tests/data/output_errors")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = y = z = np.random.rand(n_points)
    strengths = np.random.rand(n_points) + 1j * np.random.rand(n_points)

    # Test empty subject_id
    with pytest.raises(ValueError, match="subject_id cannot be empty"):
        MapBuilder(
            subject_id="",
            output_dir=output_dir,
            x=x, y=y, z=z,
            strengths=strengths
        )

    # Test empty output_dir
    with pytest.raises(ValueError, match="output_dir cannot be empty"):
        MapBuilder(
            subject_id="test_error",
            output_dir="",
            x=x, y=y, z=z,
            strengths=strengths
        )

    # Test invalid eps
    with pytest.raises(ValueError, match="eps must be positive"):
        MapBuilder(
            subject_id="test_error",
            output_dir=output_dir,
            x=x, y=y, z=z,
            strengths=strengths,
            eps=0
        )

    # Test invalid dtype
    with pytest.raises(ValueError, match="Invalid dtype: invalid"):
        MapBuilder(
            subject_id="test_error",
            output_dir=output_dir,
            x=x, y=y, z=z,
            strengths=strengths,
            dtype='invalid'
        )
    # Test mismatched strengths shape
    with pytest.raises(ValueError, match="Strengths last dimension .* must match number of points"):
        MapBuilder(
            subject_id="test_error",
            output_dir=output_dir,
            x=x, y=y, z=z,
            strengths=np.random.rand(2, n_points + 1) # Wrong last dim
        )
    # Test wrong strengths ndim
    with pytest.raises(ValueError, match="strengths must be a 1D or 2D array"):
        MapBuilder(
            subject_id="test_error",
            output_dir=output_dir,
            x=x, y=y, z=z,
            strengths=np.random.rand(2, 3, n_points) # 3D is invalid
        )

    # Clean up error dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

# --- New Tests for Verification ---

def test_mask_content(setup_teardown, monkeypatch):
    """Verify that applying a mask modifies the FFT result meaningfully."""
    data = setup_teardown
    map_builder = MapBuilder(
        subject_id=f"test_mask_content_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"],
        dtype='complex64' # Use float32 for quicker test
    )
    map_builder.compute_forward_fft()
    # Ensure FFT result is not all zeros initially (given the Gaussian input)
    assert not np.allclose(map_builder.fft_result, 0)
    
    # --- Monkeypatch np.random.uniform to return a fixed center (0,0,0) --- 
    fixed_center_coords = [0.0, 0.0, 0.0] # Use list to pop values
    def mock_uniform(low, high):
        # Return 0.0 the first 3 times it's called in generate_kspace_masks
        assert low == -1 and high == 1 # Sanity check
        try:
            return fixed_center_coords.pop(0)
        except IndexError:
            # Should not happen for n_centers=1
            return np.random.uniform(low, high) 
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    # --- End Monkeypatch ---
    
    # Generate a mask expected to select some central region
    # Increase radius slightly to ensure grid points are captured near center
    map_builder.generate_kspace_masks(n_centers=1, radius=0.3)
    mask = map_builder.kspace_masks[0]
    
    # Debugging: Print mask info
    # print(f"\n--- Debug Info: test_mask_content ---")
    # # The centers are not explicitly stored, but we know n_centers=1
    # # Let's check how many points the mask selected:
    # num_mask_true = np.count_nonzero(mask)
    # print(f"Number of True values in generated mask: {num_mask_true}")
    # print(f"Mask shape: {mask.shape}")
    # print(f"--- End Debug Info ---")
    
    # Apply the mask
    masked_fft = map_builder.fft_result * mask # Broadcasts mask across n_trans
    
    # Verify the mask did something
    # 1. The masked result should not be identical to the original
    assert not np.array_equal(masked_fft, map_builder.fft_result)
    # 2. The masked result should not be all zeros (unless the original FFT was zero, which we checked)
    assert not np.allclose(masked_fft, 0)
    # 3. Check that the number of non-zero elements is plausible
    # Use the mask directly to count expected non-zeros per transform
    num_mask_true = np.count_nonzero(mask)
    masked_non_zeros = np.count_nonzero(masked_fft)
    # Total non-zeros should be approx n_trans * num_mask_true (within tolerance for numerical zeros)
    assert masked_non_zeros <= data['n_trans'] * num_mask_true 
    assert masked_non_zeros > 0 # Mask should select something

def test_inverse_map_zero_input(setup_teardown):
    """Test that a zero k-space input results in a zero inverse map."""
    data = setup_teardown
    map_builder = MapBuilder(
        subject_id=f"test_inverse_zero_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"],
        dtype='complex64' # Use float32 for quicker test
    )
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
    
    # Manually set the FFT result to zero before inverse transform
    map_builder.fft_result = np.zeros_like(map_builder.fft_result)
    
    map_builder.compute_inverse_maps()
    
    assert len(map_builder.inverse_maps) == 1
    # Check shape (n_trans, n_points)
    assert map_builder.inverse_maps[0].shape == (data['n_trans'], map_builder.n_points)
    # The inverse map should be all zeros (or very close due to precision)
    assert np.allclose(map_builder.inverse_maps[0], 0, atol=1e-7)

def test_gradient_map_constant_input(setup_teardown):
    """Test that a constant inverse map results in a zero gradient map."""
    data = setup_teardown
    map_builder = MapBuilder(
        subject_id=f"test_gradient_constant_nt{data['n_trans']}",
        output_dir=data["output_dir"],
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"],
        dtype='complex64' # Use float32 for quicker test
    )
    map_builder.compute_forward_fft()
    map_builder.generate_kspace_masks(n_centers=1, radius=0.5)
    map_builder.compute_inverse_maps()
    
    assert len(map_builder.inverse_maps) == 1
    
    # Create a dummy inverse map (n_trans, n_points) of ones
    # We bypass compute_inverse_maps to avoid dependency
    constant_inverse_map = np.ones((data['n_trans'], map_builder.n_points), dtype=map_builder.dtype)
    map_builder.inverse_maps = [constant_inverse_map] # Use a list containing this map
    
    # Compute gradients
    map_builder.compute_gradient_maps()
    
    assert len(map_builder.gradient_maps) == 1
    # Check shape (n_trans, nx, ny, nz)
    expected_shape = (data['n_trans'], map_builder.nx, map_builder.ny, map_builder.nz)
    assert map_builder.gradient_maps[0].shape == expected_shape

    # The gradient map corresponding to the constant input should be zero
    assert np.allclose(map_builder.gradient_maps[0], 0, atol=1e-7)

def test_custom_mask_shapes(setup_teardown):
    """Test generation and use of cubic, slice, and slab k-space masks."""
    data = setup_teardown
    subject_id = f"test_custom_masks_nt{data['n_trans']}"
    output_dir = data["output_dir"]
    
    # Use a fixed grid size for predictable k-values if needed, otherwise estimate
    map_builder = MapBuilder(
        subject_id=subject_id,
        output_dir=output_dir,
        x=data["x"], y=data["y"], z=data["z"],
        strengths=data["strengths"],
        dtype='complex64' # Use float32 for quicker test
    )

    # Need k-space coordinates, so compute forward FFT first
    map_builder.compute_forward_fft()
    assert map_builder.fft_result is not None

    # --- Generate Custom Masks ---
    initial_mask_count = len(map_builder.kspace_masks) # Should be 0
    assert initial_mask_count == 0

    # 1. Cubic Mask (small region around origin)
    # k-values depend on grid size and stride, fftfreq typical range is [-0.5, 0.5]
    k_bound = 0.1 
    map_builder.generate_cubic_mask(
        kx_min=-k_bound, kx_max=k_bound,
        ky_min=-k_bound, ky_max=k_bound,
        kz_min=-k_bound, kz_max=k_bound
    )
    
    # 2. Slice Mask (DC component plane in z)
    map_builder.generate_slice_mask(axis='z', k_value=0) 
    
    # 3. Slab Mask (A band along x-axis)
    map_builder.generate_slab_mask(axis='x', k_min=0.05, k_max=0.15)

    num_masks_generated = 3
    assert len(map_builder.kspace_masks) == initial_mask_count + num_masks_generated

    # --- Basic checks on mask content and files ---
    mask_dir = map_builder.data_dir
    for i in range(initial_mask_count, initial_mask_count + num_masks_generated):
        mask = map_builder.kspace_masks[i]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (map_builder.nx, map_builder.ny, map_builder.nz)
        assert mask.dtype == bool
        # Check corresponding file exists
        mask_file = mask_dir / f"kspace_mask_{i}.npy"
        assert mask_file.exists(), f"Mask file not found: {mask_file}"
        # Optional: Load and verify shape
        loaded_mask = np.load(mask_file)
        assert loaded_mask.shape == mask.shape

    # --- Compute inverse maps using these masks ---
    map_builder.compute_inverse_maps()
    assert len(map_builder.inverse_maps) == initial_mask_count + num_masks_generated

    # --- Check inverse map files exist ---
    for i in range(initial_mask_count, initial_mask_count + num_masks_generated):
         inv_map_file = mask_dir / f"inverse_map_{i}.npy"
         assert inv_map_file.exists(), f"Inverse map file not found: {inv_map_file}"
         # Optional: check shape of loaded inverse map
         inv_map = np.load(inv_map_file)
         assert inv_map.shape == (data['n_trans'], map_builder.n_points)

    # --- Check analysis (optional basic check) ---
    # map_builder.analyze_inverse_maps(analyses_to_run=['magnitude'])
    # analysis_file = map_builder.analysis_dir / f"map_{initial_mask_count}_magnitude.npy"
    # assert analysis_file.exists()

# --- End of New Tests --- 