import numpy as np
import finufft
import logging
from pathlib import Path
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import h5py
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import the new analysis functions
from .map_analysis import (
    calculate_magnitude, calculate_phase, 
    calculate_local_variance, calculate_temporal_difference,
    calculate_local_variance_vectorized, calculate_temporal_difference_vectorized,
    calculate_local_variance_fully_vectorized  # Add the new fully optimized function
)

# Import the enhanced features module conditionally
try:
    from .enhanced_features import (
        load_config, compute_radial_gradient, calculate_spectral_slope,
        calculate_spectral_entropy, calculate_kspace_anisotropy,
        calculate_higher_order_moments, calculate_excitation_map
    )
    _ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    _ENHANCED_FEATURES_AVAILABLE = False

class MapBuilder:
    """Class for building and analyzing 3D maps using FINUFFT, supporting multiple transforms (n_trans)."""

    def __init__(self, subject_id, output_dir, x, y, z, strengths, 
                 nx=None, ny=None, nz=None, eps=1e-6, upsampling_factor=2, dtype='complex128', estimate_grid=True,
                 normalize_fft_result=False, padding=0, stride=1, enable_enhanced_features=True, 
                 config_path=None):
        """Initialize MapBuilder.

        Args:
            subject_id (str): Unique identifier for the subject
            output_dir (str or Path): Directory to save outputs
            x (ndarray): 1D X coordinates of non-uniform points.
            y (ndarray): 1D Y coordinates of non-uniform points.
            z (ndarray): 1D Z coordinates of non-uniform points.
            strengths (ndarray): Strength values at each coordinate. 
                                 Shape can be (N,) for single transform or (n_trans, N) for multiple.
                                 N must be equal to len(x).
            nx (int, optional): Number of grid points in the X direction.
            ny (int, optional): Number of grid points in the Y direction.
            nz (int, optional): Number of grid points in the Z direction.
            eps (float, optional): FINUFFT precision. Defaults to 1e-6.
            upsampling_factor (int, optional): Upsampling factor for grid estimation. Defaults to 2.
            dtype (str, optional): Data type for complex values. Must be 'complex128' for FINUFFT.
            estimate_grid (bool, optional): If True, estimate grid dimensions based on point density. Defaults to True.
            normalize_fft_result (bool, optional): Normalize FFT power spectrum. Defaults to False.
            padding (int, optional): [Deprecated/Test Compatibility] Padding factor. Defaults to 0.
            stride (int, optional): Stride for k-space sampling. Defaults to 1.
            enable_enhanced_features (bool, optional): Enable enhanced features. Defaults to True.
            config_path (str or Path, optional): Path to custom configuration file for enhanced features.
        """
        # Input validation
        if not subject_id:
            raise ValueError("subject_id cannot be empty")
        if not output_dir:
            raise ValueError("output_dir cannot be empty")
        if eps <= 0:
            raise ValueError("eps must be positive")
        
        # Validate dtype - must be complex128 for FINUFFT
        if dtype != 'complex128':
            raise ValueError("dtype must be 'complex128' for FINUFFT compatibility")

        self.subject_id = subject_id
        self.output_dir = Path(output_dir)
        self._setup_logging(subject_id)

        # Create output directories
        self.subject_dir = self.output_dir / subject_id
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 files for different data types
        self.data_file = h5py.File(self.subject_dir / 'data.h5', 'w')
        self.analysis_file = h5py.File(self.subject_dir / 'analysis.h5', 'w')
        if enable_enhanced_features:
            self.enhanced_file = h5py.File(self.subject_dir / 'enhanced.h5', 'w')

        # --- Assign parameters early --- 
        self.normalize_fft_result = normalize_fft_result 
        self.padding = padding
        self.stride = stride
        self.eps = eps
        self.upsampling_factor = upsampling_factor

        # Enhanced features settings
        self.enable_enhanced_features = enable_enhanced_features
        if enable_enhanced_features:
            if not _ENHANCED_FEATURES_AVAILABLE:
                self.logger.warning("Enhanced features requested but module not available. Disabling enhanced features.")
                self.enable_enhanced_features = False
            else:
                self.config = load_config(config_path)
                self.logger.info("Enhanced features enabled with configuration")

        # Convert coordinates to float32 for memory efficiency (except when used with FINUFFT)
        x_in = np.asarray(x, dtype=np.float32).ravel()
        y_in = np.asarray(y, dtype=np.float32).ravel()
        z_in = np.asarray(z, dtype=np.float32).ravel()
        self.n_points = x_in.size

        if not (x_in.shape == y_in.shape == z_in.shape):
            raise ValueError("x, y, and z must have the same number of points after flattening")
         
        # Store coordinates
        self.x_coords_1d = x_in
        self.y_coords_1d = y_in
        self.z_coords_1d = z_in

        # Handle strengths and convert to complex128 for FINUFFT
        strengths_in = np.asarray(strengths, dtype=np.complex128)
        if strengths_in.ndim == 1:
            if strengths_in.size != self.n_points:
                raise ValueError(f"1D strengths size ({strengths_in.size}) must match number of points ({self.n_points})")
            self.strengths = strengths_in.reshape(1, self.n_points)
            self.n_trans = 1
        elif strengths_in.ndim == 2:
            if strengths_in.shape[1] != self.n_points:
                raise ValueError(f"Strengths last dimension ({strengths_in.shape[1]}) must match number of points ({self.n_points})")
            self.strengths = strengths_in
            self.n_trans = strengths_in.shape[0]
        else:
            raise ValueError("strengths must be a 1D or 2D array")

        self.logger.info(f"Initialized with n_trans = {self.n_trans}")
        
        # Store coordinates for FINUFFT (must be float64)
        self.x = x_in.astype(np.float64)  # Required by FINUFFT
        self.y = y_in.astype(np.float64)  # Required by FINUFFT
        self.z = z_in.astype(np.float64)  # Required by FINUFFT
        self.dtype = np.dtype(dtype)
        self.real_dtype = np.float32  # Use float32 for real numbers

        if self.strengths.ndim == 1:
            self.strengths = self.strengths[np.newaxis, :]
        self.n_trans, self.n_points = self.strengths.shape

        if not (self.x.size == self.y.size == self.z.size == self.n_points):
             raise ValueError("Coordinate array sizes must match the number of strength points.")

        self.logger.info(f"Initialized with n_trans = {self.n_trans}, n_points = {self.n_points}")

        # Grid Dimensions
        if estimate_grid:
             self.nx, self.ny, self.nz = self._estimate_grid_dims(self.n_points)
             self.logger.info(f"Estimated grid dimensions (nx, ny, nz): ({self.nx}, {self.ny}, {self.nz}) based on {self.n_points} points")
        else:
             if not all([nx, ny, nz]):
                 raise ValueError("If estimate_grid is False, nx, ny, and nz must be provided.")
             self.nx, self.ny, self.nz = nx, ny, nz
             self.logger.info(f"Using provided grid dimensions (nx, ny, nz): ({self.nx}, {self.ny}, {self.nz})")
        
        self.n_modes = (self.nx, self.ny, self.nz)

        # Initialize k-space grids (1D)
        self.kx = np.fft.fftfreq(self.nx, d=self.stride).astype(np.float32)
        self.ky = np.fft.fftfreq(self.ny, d=self.stride).astype(np.float32)
        self.kz = np.fft.fftfreq(self.nz, d=self.stride).astype(np.float32)
        
        # Create 3D k-space coordinate grids for masking
        self.Kx, self.Ky, self.Kz = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')

        # Initialize FINUFFT plans
        self.forward_plan = None
        self.inverse_plan = None
        self._initialize_plans()

        # Results storage
        self.fft_result = None
        self.fft_prob_density = None
        self.kspace_masks = []
        self.inverse_maps = []
        self.gradient_maps = []
        self.analysis_results = {}
        
        if self.enable_enhanced_features:
            self.enhanced_results = {}

        self.logger.info("MapBuilder initialized successfully")
        if self.enable_enhanced_features:
            self.logger.info("Enhanced features are enabled")

    def __del__(self):
        """Cleanup method to ensure HDF5 files are properly closed."""
        try:
            self.data_file.close()
            self.analysis_file.close()
            if hasattr(self, 'enhanced_file'):
                self.enhanced_file.close()
        except Exception as e:
            self.logger.warning(f"Error closing HDF5 files: {e}")

    def _save_to_hdf5(self, file_or_group, name, data, compression="gzip", compression_opts=9):
        """Helper method to save data to HDF5 with compression.
        Handles basic types, numpy arrays, and nested dictionaries.
        """
        # Check if the target name already exists in the current group/file
        if name in file_or_group:
            del file_or_group[name]
            
        if isinstance(data, dict):
            # Create a group for the dictionary
            group = file_or_group.create_group(name)
            # Recursively save items within the dictionary to the new group
            for key, value in data.items():
                self._save_to_hdf5(group, key, value, compression, compression_opts)
        elif isinstance(data, np.ndarray):
            # Convert data to float16 if it's a real-valued array
            if np.isrealobj(data):
                data = data.astype(np.float16)
            # Save NumPy array directly
            file_or_group.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)
        else:
            # Attempt to save other types (int, float, bool, string, list, tuple)
            try:
                # Convert basic python types (or lists/tuples of them) to numpy arrays for saving
                serializable_data = np.array(data) 
                # Convert to float16 if it's a real-valued array
                if np.isrealobj(serializable_data):
                    serializable_data = serializable_data.astype(np.float16)
                # Check if conversion resulted in object dtype (often indicates mixed types or unhandled objects)
                if serializable_data.dtype == object:
                    # Fallback to string if it's an object array
                    self.logger.warning(f"Could not serialize '{name}' directly (dtype={serializable_data.dtype}). Saving as string.")
                    file_or_group.create_dataset(name, data=str(data), compression=compression, compression_opts=compression_opts)
                else:
                    file_or_group.create_dataset(name, data=serializable_data, compression=compression, compression_opts=compression_opts)
            except (TypeError, ValueError) as e:
                # If conversion or direct saving fails, convert to string as fallback
                self.logger.warning(f"Could not serialize '{name}' (type: {type(data)}, error: {e}). Saving as string.")
                try:
                    file_or_group.create_dataset(name, data=str(data), compression=compression, compression_opts=compression_opts)
                except Exception as e_str:
                    self.logger.error(f"Failed to save '{name}' even as string: {e_str}")

    def _setup_logging(self, subject_id):
        """Set up logging configuration."""
        self.logger = logging.getLogger(f"MapBuilder_{subject_id}")
        self.logger.setLevel(logging.INFO)

        # Create file handler
        fh = logging.FileHandler(self.output_dir / "map_builder.log")
        fh.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _estimate_grid_dims(self, n_points):
        """Estimate grid dimensions based on number of points and upsampling factor."""
        # Simple heuristic: cube root of points times upsampling factor
        # Ensure dims are even, as often preferred by FFT algorithms
        approx_edge = int(np.ceil(n_points**(1/3))) 
        n_est = approx_edge * self.upsampling_factor
        # Make dims even
        nx = n_est + (n_est % 2) 
        ny = n_est + (n_est % 2)
        nz = n_est + (n_est % 2)
        return nx, ny, nz

    def _save_params(self):
        pass # Placeholder - saving not implemented in this version

    def _initialize_plans(self):
        """Initialize FINUFFT plans for forward and inverse transforms."""
        # Determine the real data type corresponding to the complex dtype
        real_dtype = np.float32 if np.dtype(self.dtype) == np.complex64 else np.float64

        # Convert stored 1D coordinates to the appropriate real dtype for the plan
        x_plan = np.asarray(self.x_coords_1d, dtype=real_dtype)
        y_plan = np.asarray(self.y_coords_1d, dtype=real_dtype)
        z_plan = np.asarray(self.z_coords_1d, dtype=real_dtype)

        # Initialize forward transform plan (type 1)
        self.forward_plan = finufft.Plan(1, (self.nx, self.ny, self.nz), n_trans=self.n_trans, eps=self.eps, dtype=self.dtype)
        self.forward_plan.setpts(x_plan, y_plan, z_plan)

        # Initialize inverse transform plan (type 2)
        self.inverse_plan = finufft.Plan(2, (self.nx, self.ny, self.nz), n_trans=self.n_trans, eps=self.eps, dtype=self.dtype)
        self.inverse_plan.setpts(x_plan, y_plan, z_plan)

        self.logger.info("FINUFFT plans initialized")

    def compute_forward_fft(self):
        """Compute forward FFT and save the result. Optionally normalizes the result."""
        # Input strengths shape: (n_trans, N)
        # Execute forward transform. Output is flattened C-ordered: (n_trans * nx * ny * nz,)
        fft_result_flat = self.forward_plan.execute(self.strengths)

        # Reshape the result to (n_trans, nx, ny, nz)
        self.fft_result = fft_result_flat.reshape(self.n_trans, self.nx, self.ny, self.nz)

        # Store the reshaped 3D result in forward_fft for test compatibility
        self.forward_fft = self.fft_result

        # Save raw FFT result to file
        self._save_to_hdf5(self.data_file, 'forward_fft', self.fft_result)
        self.logger.info("Forward FFT computed and saved")

        # Normalize if requested
        if self.normalize_fft_result:
            self.logger.info("Normalizing FFT results to probability densities.")
            prob_density = np.abs(self.fft_result)**2 # Shape (n_trans, nx, ny, nz)
            norm_factor = np.sum(prob_density, axis=(1, 2, 3), keepdims=True)
            # Handle potential division by zero if a transform's power is all zero
            zero_power_mask = (norm_factor == 0)
            if np.any(zero_power_mask):
                 self.logger.warning(f"Detected zero total power for {np.sum(zero_power_mask)} transform(s). Setting density to zero.")
                 norm_factor[zero_power_mask] = 1.0 # Avoid division by zero, result will be 0
            
            self.fft_prob_density = prob_density / norm_factor # Shape (n_trans, nx, ny, nz)
            self._save_to_hdf5(self.data_file, 'fft_prob_density', self.fft_prob_density)
            self.logger.info("FFT probability densities computed and saved.")
        else:
             self.fft_prob_density = None # Ensure it's None if not calculated

    def generate_kspace_masks(self, n_centers=2, radius=0.5):
        """Generates spherical masks centered randomly in k-space.
        
        This method appends masks to self.kspace_masks.

        Args:
            n_centers (int): Number of random spherical masks to generate.
            radius (float): Radius of the spheres in k-space units.
        """
        self.logger.info(f"Generating {n_centers} spherical k-space masks with radius {radius}")
        
        # Use pre-calculated 3D grids Kx, Ky, Kz from __init__
        # kx_max = np.max(np.abs(self.kx)) # Can be used for relative radius if needed

        for _ in range(n_centers):
            # Choose a random center *index* on the grid
            center_idx = (
                np.random.randint(0, self.nx),
                np.random.randint(0, self.ny),
                np.random.randint(0, self.nz)
            )
            center_k = (self.Kx[center_idx], self.Ky[center_idx], self.Kz[center_idx])
            self.logger.debug(f"Mask center (kx, ky, kz): {center_k}")

            # Calculate squared Euclidean distance from the center to all grid points
            dist_sq = (self.Kx - center_k[0])**2 + (self.Ky - center_k[1])**2 + (self.Kz - center_k[2])**2

            # Create mask based on radius
            mask = dist_sq <= radius**2
            
            # Append and save
            mask_index = len(self.kspace_masks)
            self.kspace_masks.append(mask)
            mask_filename = self.data_file.create_dataset(f"kspace_mask_{mask_index}", data=mask)
            self.logger.info(f"Generated spherical mask {mask_index} centered near {center_k} and saved to {mask_filename}")
        
        self.logger.info(f"Total k-space masks generated: {len(self.kspace_masks)}")

    # --- NEW MASK GENERATION METHODS --- 

    def generate_cubic_mask(self, kx_min, kx_max, ky_min, ky_max, kz_min, kz_max):
        """Generates a cubic (box) mask in k-space based on coordinate boundaries.
        
        This method appends the mask to self.kspace_masks.

        Args:
            kx_min, kx_max (float): Min/max boundaries for kx dimension.
            ky_min, ky_max (float): Min/max boundaries for ky dimension.
            kz_min, kz_max (float): Min/max boundaries for kz dimension.
        """
        self.logger.info(f"Generating cubic k-space mask: kx=[{kx_min},{kx_max}), ky=[{ky_min},{ky_max}), kz=[{kz_min},{kz_max})")
        mask_x = (self.Kx >= kx_min) & (self.Kx < kx_max)
        mask_y = (self.Ky >= ky_min) & (self.Ky < ky_max)
        mask_z = (self.Kz >= kz_min) & (self.Kz < kz_max)
        mask = mask_x & mask_y & mask_z
        
        # Append and save
        mask_index = len(self.kspace_masks)
        self.kspace_masks.append(mask)
        mask_filename = self.data_file.create_dataset(f"kspace_mask_{mask_index}", data=mask)
        self.logger.info(f"Generated cubic mask {mask_index} and saved to {mask_filename}")

    def generate_slice_mask(self, axis, k_value):
        """Generates a mask selecting a single slice in k-space closest to a given k-value.
        
        This method appends the mask to self.kspace_masks.

        Args:
            axis (str): The axis for the slice ('x', 'y', or 'z').
            k_value (float): The target k-coordinate value for the slice.
        """
        self.logger.info(f"Generating k-space slice mask: axis={axis}, k_value={k_value}")
        mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        
        try:
            if axis == 'x':
                idx = np.argmin(np.abs(self.kx - k_value))
                mask[idx, :, :] = True
                actual_k = self.kx[idx]
            elif axis == 'y':
                idx = np.argmin(np.abs(self.ky - k_value))
                mask[:, idx, :] = True
                actual_k = self.ky[idx]
            elif axis == 'z':
                idx = np.argmin(np.abs(self.kz - k_value))
                mask[:, :, idx] = True
                actual_k = self.kz[idx]
            else:
                raise ValueError("Axis must be 'x', 'y', or 'z'")
            
            # Append and save
            mask_index = len(self.kspace_masks)
            self.kspace_masks.append(mask)
            mask_filename = self.data_file.create_dataset(f"kspace_mask_{mask_index}", data=mask)
            self.logger.info(f"Generated slice mask {mask_index} for axis {axis} at index {idx} (k~{actual_k:.4f}) and saved to {mask_filename}")
        
        except ValueError as e:
            self.logger.error(f"Failed to generate slice mask: {e}")

    def generate_slab_mask(self, axis, k_min, k_max):
        """Generates a mask selecting a slab (range of slices) in k-space.
        
        This method appends the mask to self.kspace_masks.

        Args:
            axis (str): The axis for the slab ('x', 'y', or 'z').
            k_min (float): The minimum k-coordinate value for the slab (inclusive).
            k_max (float): The maximum k-coordinate value for the slab (inclusive).
        """
        self.logger.info(f"Generating k-space slab mask: axis={axis}, range=[{k_min}, {k_max}]")
        mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        
        try:
            if axis == 'x':
                indices = np.where((self.kx >= k_min) & (self.kx <= k_max))[0]
                if indices.size > 0:
                    mask[indices, :, :] = True
                    k_range_actual = (self.kx[indices.min()], self.kx[indices.max()])
                else: k_range_actual = (np.nan, np.nan)
            elif axis == 'y':
                indices = np.where((self.ky >= k_min) & (self.ky <= k_max))[0]
                if indices.size > 0:
                    mask[:, indices, :] = True
                    k_range_actual = (self.ky[indices.min()], self.ky[indices.max()])
                else: k_range_actual = (np.nan, np.nan)
            elif axis == 'z':
                indices = np.where((self.kz >= k_min) & (self.kz <= k_max))[0]
                if indices.size > 0:
                    mask[:, :, indices] = True
                    k_range_actual = (self.kz[indices.min()], self.kz[indices.max()])
                else: k_range_actual = (np.nan, np.nan)
            else:
                raise ValueError("Axis must be 'x', 'y', or 'z'")

            if indices.size == 0:
                 self.logger.warning(f"Generated slab mask for axis {axis} produced an empty mask for range [{k_min}, {k_max}].")
                 # Still append and save the empty mask

            # Append and save
            mask_index = len(self.kspace_masks)
            self.kspace_masks.append(mask)
            mask_filename = self.data_file.create_dataset(f"kspace_mask_{mask_index}", data=mask)
            self.logger.info(f"Generated slab mask {mask_index} for axis {axis} covering k range ~[{k_range_actual[0]:.4f}, {k_range_actual[1]:.4f}] and saved to {mask_filename}")

        except ValueError as e:
            self.logger.error(f"Failed to generate slab mask: {e}")

    # --- END NEW MASK METHODS ---

    def compute_inverse_maps(self):
        """Compute inverse maps for each k-space mask across all transforms."""
        if not hasattr(self, 'kspace_masks'):
            self.generate_kspace_masks()

        if self.fft_result is None:
            raise RuntimeError("Forward FFT must be computed before inverse maps.")

        self.inverse_maps = []
        for i, mask in enumerate(self.kspace_masks):
            # Apply 3D mask to 4D FFT result using broadcasting
            # mask shape (nx, ny, nz), fft_result shape (n_trans, nx, ny, nz)
            masked_fft = self.fft_result * mask # Result shape (n_trans, nx, ny, nz)

            # Execute inverse transform. Input needs to match plan (n_trans, nx, ny, nz)
            # Output is flattened C-ordered: (n_trans * n_points,)
            inverse_map_flat = self.inverse_plan.execute(masked_fft)

            # Reshape result back to (n_trans, n_points)
            inverse_map = inverse_map_flat.reshape(self.n_trans, self.n_points)

            # Store the result
            self.inverse_maps.append(inverse_map)
            
            # Save data to file
            # Save the (n_trans, n_points) array for this mask
            self._save_to_hdf5(self.data_file, f"inverse_map_{i}", inverse_map)

            # Generate visualization for the inverse map
            # Plotting needs to be handled outside, as input is now 4D/batched
            # self.generate_volume_plot(inverse_map[0], f"inverse_volume_{i}_t0.html") # Example: Plot first transform

        self.logger.info(f"Computed inverse maps for {len(self.inverse_maps)} masks, each with {self.n_trans} transforms.")

    def _interpolate_transform(self, points_nu, points_u, inverse_map_t, fill_val, nx, ny, nz):
        """Helper function for parallel interpolation of a single transform."""
        # Use KDTree for faster nearest-neighbor interpolation
        tree = KDTree(points_nu)
        distances, indices = tree.query(points_u, k=1)
        
        # Get interpolated values
        interpolated = inverse_map_t[indices]
        
        # Handle any points that couldn't be interpolated
        mask = distances > np.mean(distances) * 2  # Points too far from any data point
        interpolated[mask] = fill_val
        
        return interpolated.reshape(nx, ny, nz)

    def compute_gradient_maps(self, use_analytical_method=None, skip_interpolation=True):
        """
        Compute gradient maps by interpolating inverse maps onto a grid.
        
        If enhanced features are enabled and use_analytical_method is True or None,
        the analytical gradient method will be used, which is faster and more accurate.
        
        Args:
            use_analytical_method (bool, optional): Whether to use the analytical gradient method.
                If None, uses the value from config if enhanced features are enabled.
            skip_interpolation (bool, optional): Whether to skip interpolation to regular grid.
                When True, only non-uniform data is stored, which significantly improves performance.
                Default is True.
        """
        # Determine if we should use the analytical method
        if use_analytical_method is None and self.enable_enhanced_features:
            use_analytical_method = self.config.get("gradient_weighting", False)
        elif use_analytical_method and not self.enable_enhanced_features:
            self.logger.warning("Analytical gradient method requested but enhanced features are not enabled. Using interpolation method.")
            raise RuntimeError("Analytical gradient method requires enhanced features to be enabled")
        
        if use_analytical_method and self.enable_enhanced_features and _ENHANCED_FEATURES_AVAILABLE:
            self.logger.info("Computing gradient maps using analytical method (enhanced feature).")
            return self._compute_analytical_gradient_maps(skip_interpolation=skip_interpolation)
        
        # Legacy interpolation-based method
        if not self.inverse_maps:
            self.logger.error("No inverse maps computed yet. Cannot compute gradient maps.")
            return

        self.gradient_maps = []
        
        # If skipping interpolation, just store non-uniform data
        if skip_interpolation:
            self.logger.info(f"Computing gradient maps for {len(self.inverse_maps)} inverse maps (skipping interpolation)")
            for i, inverse_map_nu in enumerate(self.inverse_maps):
                # Convert to float32 for memory efficiency
                inverse_map_nu = inverse_map_nu.astype(np.float32)
                
                # Store the non-uniform data directly
                self._save_to_hdf5(self.data_file, f"inverse_map_nu_{i}", inverse_map_nu)
                self.logger.info(f"Inverse map {i} stored without interpolation.")
            
            self.logger.info(f"Stored {len(self.inverse_maps)} inverse maps without interpolation.")
            return
            
        # Regular interpolation-based method
        self.logger.info(f"Computing gradient maps for {len(self.inverse_maps)} inverse maps using optimized interpolation.")

        # Define source points (non-uniform) - convert to float32 for memory efficiency
        points_nu = np.stack((
            self.x_coords_1d.astype(np.float32),
            self.y_coords_1d.astype(np.float32),
            self.z_coords_1d.astype(np.float32)
        ), axis=-1)

        # Define target grid points (uniform) - convert to float32
        grid_x = np.linspace(self.x_coords_1d.min(), self.x_coords_1d.max(), self.nx, dtype=np.float32)
        grid_y = np.linspace(self.y_coords_1d.min(), self.y_coords_1d.max(), self.ny, dtype=np.float32)
        grid_z = np.linspace(self.z_coords_1d.min(), self.z_coords_1d.max(), self.nz, dtype=np.float32)
        
        grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        points_u = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=-1)
        
        # Create a partial function with fixed arguments for parallel processing
        interpolate_func = partial(
            self._interpolate_transform,
            points_nu=points_nu,
            points_u=points_u,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz
        )
        
        for i, inverse_map_nu in enumerate(self.inverse_maps):
            # Convert to float32 for memory efficiency
            inverse_map_nu = inverse_map_nu.astype(np.float32)
            
            # Calculate fill value once per map
            fill_val = np.mean(inverse_map_nu)
            
            # Use ThreadPoolExecutor for parallel processing of transforms
            with ThreadPoolExecutor() as executor:
                # Create tasks for each transform
                futures = [
                    executor.submit(interpolate_func, inverse_map_t=inverse_map_nu[t], fill_val=fill_val)
                    for t in range(self.n_trans)
                ]
                
                # Collect results
                interpolated_maps = [future.result() for future in futures]
            
            # Stack results into a single array
            interpolated_maps_grid = np.stack(interpolated_maps, axis=0)

            # Calculate gradients using float32
            try:
                dz, dy, dx = np.gradient(interpolated_maps_grid, axis=(1, 2, 3))
                gradient_magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2)
                
                # Convert back to original dtype for storage
                gradient_magnitude = gradient_magnitude.astype(self.dtype)

                self.gradient_maps.append(gradient_magnitude)
                self._save_to_hdf5(self.data_file, f"gradient_map_{i}", gradient_magnitude)
                self.logger.info(f"Gradient map {i} computed and saved after interpolation.")

            except Exception as e:
                self.logger.error(f"Error calculating gradient for map {i}: {e}")

        self.logger.info(f"Computed gradient maps for {len(self.gradient_maps)} maps, each with {self.n_trans} transforms, using optimized interpolation.")

    def _compute_analytical_gradient_maps(self, skip_interpolation=True):
        """
        Compute gradient maps analytically in k-space using enhanced features.
        
        This is more efficient than the interpolation-based method as it requires only
        one inverse transform per mask instead of first computing inverse maps and then gradients.
        
        Args:
            skip_interpolation (bool, optional): Whether to skip interpolation to regular grid.
                When True, only non-uniform data is stored, which significantly improves performance.
                Default is True.
        """
        if self.fft_result is None:
            self.logger.error("Forward FFT must be computed before analytical gradient maps.")
            raise RuntimeError("Forward FFT must be computed before analytical gradient maps.")
            
        if not hasattr(self, 'kspace_masks') or not self.kspace_masks:
            self.logger.warning("No k-space masks generated yet. Generating default masks.")
            self.generate_kspace_masks()
            
        self.analytical_gradient_maps = []
        self.logger.info(f"Computing analytical gradient maps for {len(self.kspace_masks)} masks")
        
        for i, mask in enumerate(self.kspace_masks):
            # Apply 3D mask to 4D FFT result using broadcasting
            masked_fft = self.fft_result * mask
            
            # Compute gradient analytically in k-space
            gradient_fft = compute_radial_gradient(
                masked_fft, self.kx, self.ky, self.kz, 
                eps=self.eps, dtype=str(self.dtype)
            )
            
            # Execute inverse transform for the gradient
            gradient_map_flat = self.inverse_plan.execute(gradient_fft)
            
            # Reshape result to (n_trans, n_points)
            gradient_map_nu = gradient_map_flat.reshape(self.n_trans, self.n_points)
            gradient_magnitude_nu = np.abs(gradient_map_nu)
            
            # Save the analytical gradient results
            self._save_to_hdf5(self.enhanced_file, f"analytical_gradient_map_{i}", gradient_magnitude_nu)
            
            # Add to collection
            self.analytical_gradient_maps.append(gradient_magnitude_nu)
            
            # Skip interpolation if requested, otherwise do interpolation for backward compatibility
            if not skip_interpolation:
                # This requires interpolation onto a uniform grid
                self.logger.info(f"Interpolating analytical gradient map {i} onto regular grid")
                points_nu = np.stack((self.x_coords_1d, self.y_coords_1d, self.z_coords_1d), axis=-1)
                grid_x = np.linspace(self.x_coords_1d.min(), self.x_coords_1d.max(), self.nx)
                grid_y = np.linspace(self.y_coords_1d.min(), self.y_coords_1d.max(), self.ny)
                grid_z = np.linspace(self.z_coords_1d.min(), self.z_coords_1d.max(), self.nz)
                
                X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
                points_u = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
                
                gradient_magnitude_grid = np.zeros((self.n_trans, self.nx, self.ny, self.nz), dtype=self.real_dtype)
                
                for t in range(self.n_trans):
                    fill_val = np.mean(gradient_magnitude_nu[t])
                    interpolated_data_flat = griddata(
                        points_nu, 
                        gradient_magnitude_nu[t],
                        points_u, 
                        method='linear',
                        fill_value=fill_val
                    )
                    gradient_magnitude_grid[t] = interpolated_data_flat.reshape(self.nx, self.ny, self.nz)
                
                # Add to gradient_maps collection for backward compatibility
                self.gradient_maps.append(gradient_magnitude_grid)
    
                # Save interpolated version too
                self._save_to_hdf5(self.data_file, f"gradient_map_{i}", gradient_magnitude_grid)
                self.logger.info(f"Interpolated analytical gradient map {i} saved")
            else:
                # Just store the non-uniform data directly in gradient_maps for simplicity
                self.gradient_maps.append(gradient_magnitude_nu)
                self.logger.info(f"Analytical gradient map {i} computed and saved (no interpolation)")
            
        if skip_interpolation:
            self.logger.info(f"Computed analytical gradient maps for {len(self.kspace_masks)} masks (no interpolation)")
        else:
            self.logger.info(f"Computed analytical gradient maps for {len(self.kspace_masks)} masks with interpolation")
        return True

    def compute_enhanced_metrics(self, metrics_to_run=None):
        """
        Compute enhanced spectral metrics from the FFT result.

        Args:
            metrics_to_run (list, optional): List of metrics to compute. If None, computes all available metrics.
                Options: 'spectral_slope', 'spectral_entropy', 'anisotropy', 'higher_moments', 'excitation'.
                
        Returns:
            dict: Dictionary of computed metrics.
        """
        if not self.enable_enhanced_features:
            self.logger.warning("Enhanced features are not enabled. Call with enable_enhanced_features=True during initialization.")
            return {}
            
        if not _ENHANCED_FEATURES_AVAILABLE:
            self.logger.error("Enhanced features module not available.")
            return {}
            
        if self.fft_result is None:
            self.logger.error("Forward FFT must be computed before enhanced metrics.")
            raise RuntimeError("Forward FFT must be computed before enhanced metrics.")
            
        # Default metrics to run
        if metrics_to_run is None:
            metrics_to_run = ['spectral_slope', 'spectral_entropy', 'anisotropy', 'higher_moments']
            # Only add excitation if we have enough time points
            if self.n_trans >= 3:
                metrics_to_run.append('excitation')
                
        self.logger.info(f"Computing enhanced metrics: {metrics_to_run}")
        enhanced_results = {}
        
        # Spectral slope
        if 'spectral_slope' in metrics_to_run:
            k_min = self.config.get('slope_fit_range', [0.1, 0.8])[0]
            k_max = self.config.get('slope_fit_range', [0.1, 0.8])[1]
            
            self.logger.info(f"Computing spectral slope with k_range: [{k_min}, {k_max}]")
            spectral_slope = calculate_spectral_slope(
                self.fft_result, self.kx, self.ky, self.kz, 
                k_min=k_min, k_max=k_max
            )
            enhanced_results['spectral_slope'] = spectral_slope
            self._save_to_hdf5(self.enhanced_file, "spectral_slope", spectral_slope)
            self.logger.info(f"Spectral slope computed: {spectral_slope}")
            
        # Spectral entropy
        if 'spectral_entropy' in metrics_to_run:
            nbins = self.config.get('entropy_bin_count', 64)
            
            self.logger.info(f"Computing spectral entropy with {nbins} bins")
            spectral_entropy = calculate_spectral_entropy(
                self.fft_result, self.kx, self.ky, self.kz, 
                nbins=nbins
            )
            enhanced_results['spectral_entropy'] = spectral_entropy
            self._save_to_hdf5(self.enhanced_file, "spectral_entropy", spectral_entropy)
            self.logger.info(f"Spectral entropy computed: {spectral_entropy}")
            
        # Anisotropy
        if 'anisotropy' in metrics_to_run:
            moment = self.config.get('anisotropy_moment', 2)
            
            self.logger.info(f"Computing k-space anisotropy with moment={moment}")
            anisotropy = calculate_kspace_anisotropy(
                self.fft_result, self.kx, self.ky, self.kz, 
                moment=moment
            )
            enhanced_results['anisotropy'] = anisotropy
            self._save_to_hdf5(self.enhanced_file, "anisotropy", anisotropy)
            self.logger.info(f"K-space anisotropy computed: {anisotropy}")
            
        # Higher-order moments
        if 'higher_moments' in metrics_to_run and self.inverse_maps:
            for i, inv_map_nu in enumerate(self.inverse_maps):
                self.logger.info(f"Computing higher-order moments for inverse map {i}")
                skewness, kurtosis = calculate_higher_order_moments(inv_map_nu)
                
                # Store in map-specific results
                map_results = enhanced_results.get(f"map_{i}", {})
                map_results['skewness'] = skewness
                map_results['kurtosis'] = kurtosis
                enhanced_results[f"map_{i}"] = map_results
                
                # Save to files
                self._save_to_hdf5(self.enhanced_file, f"map_{i}_skewness", skewness)
                self._save_to_hdf5(self.enhanced_file, f"map_{i}_kurtosis", kurtosis)
                
                self.logger.info(f"Higher-order moments computed for map {i}")
                
        # Excitation map (for time series data)
        if 'excitation' in metrics_to_run and self.n_trans >= 3:
            hrf_type = self.config.get('hrf_kernel', 'canonical')
            
            self.logger.info(f"Computing excitation map with HRF type: {hrf_type}")
            excitation_map = calculate_excitation_map(
                self.fft_result, time_axis=0, 
                hrf_type=hrf_type
            )
            
            if excitation_map is not None:
                enhanced_results['excitation_map'] = excitation_map
                self._save_to_hdf5(self.enhanced_file, "excitation_map", excitation_map)
                self.logger.info(f"Excitation map computed with shape: {excitation_map.shape}")
            else:
                self.logger.warning("Excitation map computation failed")
        
        # Store the enhanced results
        self.enhanced_results = enhanced_results

        # Save results to the HDF5 file
        try:
            metrics_group = self.enhanced_file.require_group("enhanced_metrics")
            for metric_name, metric_data in enhanced_results.items():
                if metric_name.startswith("map_"): # Handle nested results for higher_moments
                    map_group = metrics_group.require_group(metric_name)
                    for sub_metric_name, sub_metric_data in metric_data.items():
                         # Check if dataset exists, delete if it does to overwrite
                        if f"{sub_metric_name}" in map_group:
                            del map_group[f"{sub_metric_name}"]
                        map_group.create_dataset(f"{sub_metric_name}", data=sub_metric_data)
                        self.logger.debug(f"Saved enhanced metric {metric_name}/{sub_metric_name} to HDF5")
                else:
                    # Check if dataset exists, delete if it does to overwrite
                    if metric_name in metrics_group:
                        del metrics_group[metric_name]
                    metrics_group.create_dataset(metric_name, data=metric_data)
                    self.logger.debug(f"Saved enhanced metric {metric_name} to HDF5")
            self.logger.info(f"Successfully saved enhanced metrics to {self.enhanced_file.filename}")
        except Exception as e:
            self.logger.error(f"Failed to save enhanced metrics to HDF5: {e}", exc_info=True)
            # Optional: re-raise or handle error appropriately
            raise

        return enhanced_results

    def analyze_inverse_maps(self, analyses_to_run=['magnitude', 'phase'], k_neighbors=5, save_format='hdf5',
                            compute_enhanced=None, enable_local_variance=False):
        """Analyze inverse maps using various metrics.

        Args:
            analyses_to_run (list): List of analyses to run (magnitude, phase, temporal_diff_magnitude, etc.)
            k_neighbors (int): Number of neighbors for local variance calculation
            save_format (str): Format for saving results ('hdf5' or 'npz')
            compute_enhanced (bool): Whether to compute enhanced metrics
            enable_local_variance (bool): Whether to calculate local variance (computationally expensive)
        """
        self.logger.info(f"Starting analysis of {len(self.inverse_maps)} inverse maps. Analyses requested: {analyses_to_run}")
        
        # Determine which analyses to run
        standard_analyses = [a for a in analyses_to_run if a not in ['spectral_slope', 'spectral_entropy', 'anisotropy', 
                                                                    'higher_moments', 'excitation']]
        enhanced_analyses = [a for a in analyses_to_run if a in ['spectral_slope', 'spectral_entropy', 'anisotropy', 
                                                               'higher_moments', 'excitation']]
        
        # Initialize results dictionary for this run
        current_analysis_results = {}
        
        # Process each inverse map for standard analyses
        for i, inv_map_nu in enumerate(self.inverse_maps):
            map_name_base = f"map_{i}"
            analysis_set = {}

            # Standard analyses
            if 'magnitude' in standard_analyses:
                magnitude = calculate_magnitude(inv_map_nu)
                analysis_set['magnitude'] = magnitude
                self._save_to_hdf5(self.analysis_file, f"{map_name_base}_magnitude", magnitude)
                self.logger.debug(f"Computed and saved magnitude for {map_name_base}")
            
            if 'phase' in standard_analyses:
                phase = calculate_phase(inv_map_nu)
                analysis_set['phase'] = phase
                self._save_to_hdf5(self.analysis_file, f"{map_name_base}_phase", phase)
                self.logger.debug(f"Computed and saved phase for {map_name_base}")

            # Only calculate local variance if explicitly requested via parameter
            if 'local_variance' in standard_analyses and enable_local_variance:
                points_nu = np.stack((self.x_coords_1d, self.y_coords_1d, self.z_coords_1d), axis=-1)
                # Use the new optimized vectorized function for better performance
                local_var = calculate_local_variance_fully_vectorized(inv_map_nu, points_nu, k=k_neighbors)
                analysis_set[f'local_variance_k{k_neighbors}'] = local_var
                self._save_to_hdf5(self.analysis_file, f"{map_name_base}_local_variance_k{k_neighbors}", local_var)
                self.logger.debug(f"Computed and saved local variance (k={k_neighbors}) for {map_name_base}")
            elif 'local_variance' in standard_analyses and not enable_local_variance:
                self.logger.info(f"Skipping local variance calculation for {map_name_base} (disabled by default)")

            # --- Corrected Temporal Difference Logic --- 
            run_td_mag = 'temporal_diff_magnitude' in standard_analyses
            run_td_phase = 'temporal_diff_phase' in standard_analyses

            if (run_td_mag or run_td_phase) and self.n_trans > 1:
                # Ensure magnitude is available if needed for td_mag
                if run_td_mag:
                    if 'magnitude' not in analysis_set:
                        # Calculate magnitude if not already done
                        magnitude = calculate_magnitude(inv_map_nu)
                        # Storing it temporarily, won't save unless explicitly requested
                        analysis_set['magnitude_temp'] = magnitude 
                        self.logger.debug(f"Calculated temporary magnitude for TD calculation on {map_name_base}")
                    else:
                        magnitude = analysis_set['magnitude']
                
                    # Use NumPy's optimized diff function for temporal difference
                    td_mag = calculate_temporal_difference(magnitude)
                    if td_mag is not None:
                        analysis_set['temporal_diff_magnitude'] = td_mag
                        self._save_to_hdf5(self.analysis_file, f"{map_name_base}_temporal_diff_magnitude", td_mag)
                        self.logger.debug(f"Computed and saved temporal difference (magnitude) for {map_name_base}")
                    else:
                        self.logger.warning(f"Temporal difference magnitude calculation returned None for {map_name_base}")
                    # Clean up temp key if it exists
                    if 'magnitude_temp' in analysis_set: del analysis_set['magnitude_temp']

                # Ensure phase is available if needed for td_phase
                if run_td_phase:
                    if 'phase' not in analysis_set:
                         # Calculate phase if not already done
                        phase = calculate_phase(inv_map_nu)
                        # Storing it temporarily, won't save unless explicitly requested
                        analysis_set['phase_temp'] = phase 
                        self.logger.debug(f"Calculated temporary phase for TD calculation on {map_name_base}")
                    else:
                        phase = analysis_set['phase']

                    # Use NumPy's optimized diff function for temporal difference
                    td_phase = calculate_temporal_difference(phase)
                    if td_phase is not None:
                        analysis_set['temporal_diff_phase'] = td_phase
                        self._save_to_hdf5(self.analysis_file, f"{map_name_base}_temporal_diff_phase", td_phase)
                        self.logger.debug(f"Computed and saved temporal difference (phase) for {map_name_base}")
                    else:
                        self.logger.warning(f"Temporal difference phase calculation returned None for {map_name_base}")
                    # Clean up temp key if it exists
                    if 'phase_temp' in analysis_set: del analysis_set['phase_temp']

            elif (run_td_mag or run_td_phase) and self.n_trans <= 1:
                self.logger.warning(f"Temporal difference requested but n_trans={self.n_trans} < 2 for {map_name_base}. Skipping.")
            # --- End Corrected Temporal Difference Logic ---
            
            # Store the analysis set for this map
            current_analysis_results[map_name_base] = analysis_set
            
        # Enhanced analyses if requested and available
        # Note: compute_enhanced_metrics saves its results directly to enhanced_file
        if enhanced_analyses and (compute_enhanced or (compute_enhanced is None and self.enable_enhanced_features)):
            enhanced_results = self.compute_enhanced_metrics(metrics_to_run=enhanced_analyses)
            if enhanced_results:
                # Add enhanced results to the summary dictionary as well
                current_analysis_results['enhanced'] = enhanced_results

        self.logger.info("Completed analysis calculation for {} maps.".format(len(self.inverse_maps)))
        
        # Update the main analysis_results attribute
        self.analysis_results.update(current_analysis_results)

        # Save the summary analysis_results dictionary based on save_format
        if save_format == 'hdf5':
            # Pass the file object and the desired top-level name for the summary group
            self._save_to_hdf5(self.analysis_file, "analysis_summary", self.analysis_results)
            self.logger.info("Successfully saved analysis results summary to HDF5")
        elif save_format == 'npz':
            # This part remains unchanged, saving individual files is fine.
            self.logger.info("Individual analysis results saved as .npy files or in HDF5. Caller is responsible for creating a summary .npz file if needed.")
        else:
            self.logger.warning(f"Unsupported save format: {save_format}")
            
        return self.analysis_results

    def process_map(self, n_centers=1, radius=0.5, analyses_to_run=['magnitude'], k_neighbors_local_var=5,
                  use_analytical_gradient=None, calculate_local_variance=False, skip_interpolation=True):
        """Run the main processing steps: FFT, masks, inverse, gradients, and analysis.
        
        Args:
            n_centers (int, optional): Number of spherical mask centers. Defaults to 2.
            radius (float, optional): Radius of spherical masks. Defaults to 0.5.
            analyses_to_run (list, optional): Analyses to run. Defaults to ['magnitude'].
            k_neighbors_local_var (int, optional): k for local variance. Defaults to 5.
            use_analytical_gradient (bool, optional): Whether to use analytical gradient.
                If None, uses the value from config if enhanced features enabled.
            calculate_local_variance (bool, optional): Whether to calculate local variance.
                Defaults to False as it is computationally expensive.
                This parameter is passed as enable_local_variance to analyze_inverse_maps.
            skip_interpolation (bool, optional): Whether to skip interpolation to regular grid.
                When True, only non-uniform data is stored, which significantly improves performance.
                Default is True.
        """
        self.compute_forward_fft()
        self.generate_kspace_masks(n_centers=n_centers, radius=radius)
        self.compute_inverse_maps()
        self.compute_gradient_maps(use_analytical_method=use_analytical_gradient, skip_interpolation=skip_interpolation)
        
        # Determine if any enhanced metrics are requested
        enhanced_requested = any(a in ['spectral_slope', 'spectral_entropy', 'anisotropy', 
                                       'higher_moments', 'excitation'] for a in analyses_to_run)
                                       
        self.analyze_inverse_maps(analyses_to_run=analyses_to_run, k_neighbors=k_neighbors_local_var,
                                 enable_local_variance=calculate_local_variance)

        self.logger.info("Map processing pipeline complete.") 

    def generate_volume_plot(self, data, filename, opacity=0.5, surface_count=10, colormap="viridis"):
        """Generate interactive 3D volume plot."""
        # Check if data is 3D
        if data.ndim != 3:
            self.logger.error(f"generate_volume_plot requires 3D data, but got shape {data.shape}. Plotting skipped.")
            return

        # Ensure coordinates used for plotting match the grid dimensions
        if self.x_coords_1d.size != data.size:
             self.logger.warning(f"Number of plot coordinates ({self.x_coords_1d.size}) does not match data size ({data.size}). Using estimated grid coordinates for plot.")
             plot_x = np.linspace(self.x_coords_1d.min(), self.x_coords_1d.max(), self.nx)
             plot_y = np.linspace(self.y_coords_1d.min(), self.y_coords_1d.max(), self.ny)
             plot_z = np.linspace(self.z_coords_1d.min(), self.z_coords_1d.max(), self.nz)
             X, Y, Z = np.meshgrid(plot_x, plot_y, plot_z, indexing='ij')
        else:
             # Assuming data corresponds to the grid derived from original points
             X = self.x_coords_1d.reshape(self.nx, self.ny, self.nz)
             Y = self.y_coords_1d.reshape(self.nx, self.ny, self.nz)
             Z = self.z_coords_1d.reshape(self.nx, self.ny, self.nz)

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=np.abs(data.flatten()), # Plot magnitude
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colormap
        ))

        fig.update_layout(title=filename,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # Save plot to HTML file
        plot_path = self.subject_dir / filename
        fig.write_html(str(plot_path))
        self.logger.info(f"Saved volume plot: {plot_path}")