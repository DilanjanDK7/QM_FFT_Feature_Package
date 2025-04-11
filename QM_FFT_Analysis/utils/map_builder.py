import numpy as np
import finufft
import logging
from pathlib import Path
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import h5py

# Import the new analysis functions
from .map_analysis import (
    calculate_magnitude, calculate_phase, 
    calculate_local_variance, calculate_temporal_difference
)

class MapBuilder:
    """Class for building and analyzing 3D maps using FINUFFT, supporting multiple transforms (n_trans)."""

    def __init__(self, subject_id, output_dir, x, y, z, strengths, 
                 nx=None, ny=None, nz=None, eps=1e-6, upsampling_factor=2, dtype='complex128', estimate_grid=True,
                 normalize_fft_result=False, padding=0, stride=1):
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
            dtype (str, optional): Data type for complex values. Defaults to 'complex128'.
            estimate_grid (bool, optional): If True, estimate grid dimensions based on point density. Defaults to True.
            normalize_fft_result (bool, optional): Normalize FFT power spectrum. Defaults to False.
            padding (int, optional): [Deprecated/Test Compatibility] Padding factor. Defaults to 0.
            stride (int, optional): Stride for k-space sampling. Defaults to 1.
        """
        # Input validation
        if not subject_id:
            raise ValueError("subject_id cannot be empty")
        if not output_dir:
            raise ValueError("output_dir cannot be empty")
        if eps <= 0:
            raise ValueError("eps must be positive")
        
        # Validate dtype
        try:
            np.dtype(dtype)
        except TypeError:
            raise ValueError(f"Invalid dtype: {dtype}")

        self.subject_id = subject_id
        self.output_dir = Path(output_dir)
        self._setup_logging(subject_id)

        # --- Assign parameters early --- 
        self.normalize_fft_result = normalize_fft_result 
        self.padding = padding # For test compatibility
        self.stride = stride
        self.eps = eps # Assign eps early too
        self.upsampling_factor = upsampling_factor # Assign upsampling factor early
        # --- End early assignments ---

        # Create output directories
        self.subject_dir = self.output_dir / subject_id
        self.plots_dir = self.subject_dir / "plots"
        self.data_dir = self.subject_dir / "data"
        self.analysis_dir = self.output_dir / self.subject_id / 'analysis'

        for directory in [self.subject_dir, self.plots_dir, self.data_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directories created/verified for subject: {subject_id}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Plots directory: {self.plots_dir}")
        self.logger.info(f"Analysis directory: {self.analysis_dir}")

        # Convert coordinates to float64 and ensure they are 1D
        x_in = np.asarray(x, dtype=np.float64).ravel()
        y_in = np.asarray(y, dtype=np.float64).ravel()
        z_in = np.asarray(z, dtype=np.float64).ravel()
        self.n_points = x_in.size

        if not (x_in.shape == y_in.shape == z_in.shape):
            raise ValueError("x, y, and z must have the same number of points after flattening")
         
        # Store coordinates (used by plotting if needed later)
        self.x_coords_1d = x_in
        self.y_coords_1d = y_in
        self.z_coords_1d = z_in

        # Handle strengths: Convert and determine n_trans
        strengths_in = np.asarray(strengths, dtype=dtype)
        if strengths_in.ndim == 1:
            if strengths_in.size != self.n_points:
                raise ValueError(f"1D strengths size ({strengths_in.size}) must match number of points ({self.n_points})")
            self.strengths = strengths_in.reshape(1, self.n_points) # Reshape to (1, N)
            self.n_trans = 1
        elif strengths_in.ndim == 2:
            if strengths_in.shape[1] != self.n_points:
                raise ValueError(f"Strengths last dimension ({strengths_in.shape[1]}) must match number of points ({self.n_points})")
            self.strengths = strengths_in
            self.n_trans = strengths_in.shape[0]
        else:
            raise ValueError("strengths must be a 1D or 2D array")

        self.logger.info(f"Initialized with n_trans = {self.n_trans}")
        
        self.x = x_in
        self.y = y_in
        self.z = z_in
        self.dtype = np.dtype(dtype)
        self.real_dtype = np.float64 if self.dtype == np.complex128 else np.float32

        if self.strengths.ndim == 1:
            self.strengths = self.strengths[np.newaxis, :] # Add batch dim if needed
        self.n_trans, self.n_points = self.strengths.shape

        if not (self.x.size == self.y.size == self.z.size == self.n_points):
             raise ValueError("Coordinate array sizes must match the number of strength points.")

        self.logger.info(f"Initialized with n_trans = {self.n_trans}, n_points = {self.n_points}")

        # Flatten coordinates for FINUFFT
        self.x_coords_1d = self.x.flatten().astype(self.real_dtype)
        self.y_coords_1d = self.y.flatten().astype(self.real_dtype)
        self.z_coords_1d = self.z.flatten().astype(self.real_dtype)
        self.points_nu = np.stack((self.x_coords_1d, self.y_coords_1d, self.z_coords_1d), axis=-1) # Store non-uniform points

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
        self.kx = np.fft.fftfreq(self.nx, d=self.stride)
        self.ky = np.fft.fftfreq(self.ny, d=self.stride)
        self.kz = np.fft.fftfreq(self.nz, d=self.stride)
        
        # Create 3D k-space coordinate grids for masking
        self.Kx, self.Ky, self.Kz = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')

        # Initialize FINUFFT plans (Now self.eps is defined)
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

        self.logger.info("MapBuilder initialized successfully")

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
        np.save(self.data_dir / "forward_fft.npy", self.fft_result)
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
            np.save(self.data_dir / "fft_prob_density.npy", self.fft_prob_density)
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
            mask_filename = self.data_dir / f"kspace_mask_{mask_index}.npy"
            np.save(mask_filename, mask)
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
        mask_filename = self.data_dir / f"kspace_mask_{mask_index}.npy"
        np.save(mask_filename, mask)
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
            mask_filename = self.data_dir / f"kspace_mask_{mask_index}.npy"
            np.save(mask_filename, mask)
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
            mask_filename = self.data_dir / f"kspace_mask_{mask_index}.npy"
            np.save(mask_filename, mask)
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
            np.save(
                self.data_dir / f"inverse_map_{i}.npy",
                inverse_map
            )

            # Generate visualization for the inverse map
            # Plotting needs to be handled outside, as input is now 4D/batched
            # self.generate_volume_plot(inverse_map[0], f"inverse_volume_{i}_t0.html") # Example: Plot first transform

        self.logger.info(f"Computed inverse maps for {len(self.inverse_maps)} masks, each with {self.n_trans} transforms.")

    def compute_gradient_maps(self):
        """Compute gradient maps by interpolating inverse maps onto a grid."""
        if not self.inverse_maps:
            self.logger.error("No inverse maps computed yet")
            raise ValueError("Inverse maps must be computed before gradient maps")

        self.gradient_maps = []

        # Define source points (non-uniform)
        points_nu = np.stack((self.x_coords_1d, self.y_coords_1d, self.z_coords_1d), axis=-1)

        # Define target grid points (uniform)
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(self.x_coords_1d.min(), self.x_coords_1d.max(), self.nx),
            np.linspace(self.y_coords_1d.min(), self.y_coords_1d.max(), self.ny),
            np.linspace(self.z_coords_1d.min(), self.z_coords_1d.max(), self.nz),
            indexing='ij'
        )
        points_u = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=-1)
        
        for i, inverse_map_nu in enumerate(self.inverse_maps):
            # inverse_map_nu shape is (n_trans, n_points)
            interpolated_maps_grid = np.zeros((self.n_trans, self.nx, self.ny, self.nz), dtype=self.dtype)

            for t in range(self.n_trans):
                # Interpolate data for this transform onto the uniform grid
                # Using linear interpolation. Fill points outside convex hull with the mean value.
                fill_val = np.mean(inverse_map_nu[t]) # Calculate mean for fill value
                interpolated_data_flat = griddata(
                    points_nu, 
                    inverse_map_nu[t], # Values at non-uniform points for transform t
                    points_u, 
                    method='linear',
                    fill_value=fill_val # Use mean as fill value
                )
                interpolated_maps_grid[t] = interpolated_data_flat.reshape(self.nx, self.ny, self.nz)
                self.logger.debug(f"Interpolated inverse map {i}, transform {t} onto grid.")

            # Gradients calculated independently for each transform along spatial axes
            # using the interpolated grid data
            try:
                dz, dy, dx = np.gradient(interpolated_maps_grid, axis=(1, 2, 3))
                gradient_magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2) # Shape (n_trans, nx, ny, nz)

                self.gradient_maps.append(gradient_magnitude)
                np.save(self.data_dir / f"gradient_map_{i}.npy", gradient_magnitude)
                self.logger.info(f"Gradient map {i} computed and saved after interpolation.")

                # Optional: Generate visualization for the first transform's gradient
                # self.generate_volume_plot(gradient_magnitude[0], f"gradient_volume_{i}_t0.html")
            except Exception as e:
                self.logger.error(f"Error calculating gradient for map {i}: {e}")

        self.logger.info(f"Computed gradient maps for {len(self.gradient_maps)} maps, each with {self.n_trans} transforms, using interpolation.")

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
        plot_path = self.plots_dir / filename
        fig.write_html(str(plot_path))
        self.logger.info(f"Saved volume plot: {plot_path}")

    def _save_dict_to_hdf5(self, hdf5_file, path, dictionary):
        """Recursively saves dictionary contents to an HDF5 file.

        Args:
            hdf5_file (h5py.File): The open HDF5 file object.
            path (str): The current group path within the HDF5 file.
            dictionary (dict): The dictionary to save.
        """
        for key, item in dictionary.items():
            current_path = f"{path}/{key}" if path else key # Build path
            if isinstance(item, dict):
                # Create group for sub-dictionary and recurse
                try:
                    group = hdf5_file.create_group(current_path)
                    self._save_dict_to_hdf5(hdf5_file, current_path, item)
                except ValueError: # Handle case where group might already exist if called multiple times
                    self.logger.warning(f"Group '{current_path}' already exists in HDF5 file. Overwriting or skipping entries.")
                    self._save_dict_to_hdf5(hdf5_file, current_path, item) 
            elif isinstance(item, np.ndarray):
                # Save NumPy array as dataset
                try:
                    hdf5_file.create_dataset(current_path, data=item)
                except ValueError: 
                     self.logger.warning(f"Dataset '{current_path}' already exists in HDF5 file. Overwriting.")
                     del hdf5_file[current_path] # Delete existing dataset before creating new one
                     hdf5_file.create_dataset(current_path, data=item)
            elif item is not None: # Handle basic types like numbers, strings
                 try:
                     # Attempt to save as dataset; might need refinement for complex types
                     hdf5_file.create_dataset(current_path, data=item)
                 except TypeError:
                      self.logger.warning(f"Could not save item of type {type(item)} directly to HDF5 dataset at '{current_path}'. Converting to string.")
                      try:
                           hdf5_file.create_dataset(current_path, data=str(item))
                      except Exception as e:
                           self.logger.error(f"Failed to save item {key} as string to HDF5: {e}")
                 except ValueError:
                     self.logger.warning(f"Dataset '{current_path}' already exists in HDF5 file. Overwriting.")
                     del hdf5_file[current_path]
                     hdf5_file.create_dataset(current_path, data=item)
            # else: skip None values

    def analyze_inverse_maps(self, analyses_to_run=['magnitude', 'phase'], k_neighbors=5, save_format='hdf5'):
        """Performs selected analyses on the computed non-uniform inverse maps.

        Args:
            analyses_to_run (list): A list of strings specifying which analyses to run.
                                    Options: 'magnitude', 'phase', 'local_variance', 
                                             'temporal_diff_magnitude', 'temporal_diff_phase'.
            k_neighbors (int): Number of neighbors for local variance calculation.
            save_format (str): Format to save the summary analysis results. 
                               Options: 'hdf5' (default) saves a single HDF5 file,
                                        'npz' logs that individual .npy files are saved and caller handles summary.
        """
        if not self.inverse_maps:
            self.logger.error("No inverse maps computed yet. Cannot run analysis.")
            return

        self.logger.info(f"Starting analysis of {len(self.inverse_maps)} inverse maps. Analyses requested: {analyses_to_run}")
        self.analysis_results = {} # Clear previous results

        for i, inv_map_nu in enumerate(self.inverse_maps):
            analysis_set = {} # Store results for this specific inverse map
            map_name_base = f"map_{i}"

            # --- Simple Analyses (Magnitude, Phase) --- 
            if 'magnitude' in analyses_to_run:
                magnitude = calculate_magnitude(inv_map_nu)
                analysis_set['magnitude'] = magnitude
                np.save(self.analysis_dir / f"{map_name_base}_magnitude.npy", magnitude)
                self.logger.debug(f"Computed and saved magnitude for {map_name_base}")
            
            if 'phase' in analyses_to_run:
                phase = calculate_phase(inv_map_nu)
                analysis_set['phase'] = phase
                np.save(self.analysis_dir / f"{map_name_base}_phase.npy", phase)
                self.logger.debug(f"Computed and saved phase for {map_name_base}")

            # --- Local Variance --- 
            if 'local_variance' in analyses_to_run:
                local_var = calculate_local_variance(inv_map_nu, self.points_nu, k=k_neighbors)
                analysis_set['local_variance'] = local_var
                np.save(self.analysis_dir / f"{map_name_base}_local_variance_k{k_neighbors}.npy", local_var)
                self.logger.debug(f"Computed and saved local variance (k={k_neighbors}) for {map_name_base}")

            # --- Temporal Differences --- 
            # Temporal diff requires magnitude/phase to be calculated first if not already done
            temp_magnitude = analysis_set.get('magnitude')
            if 'temporal_diff_magnitude' in analyses_to_run:
                if temp_magnitude is None:
                     temp_magnitude = calculate_magnitude(inv_map_nu) # Calculate if needed
                
                td_mag = calculate_temporal_difference(temp_magnitude)
                if td_mag is not None:
                    analysis_set['temporal_diff_magnitude'] = td_mag
                    np.save(self.analysis_dir / f"{map_name_base}_temporal_diff_magnitude.npy", td_mag)
                    self.logger.debug(f"Computed and saved temporal difference (magnitude) for {map_name_base}")
                else:
                    self.logger.warning(f"Skipping temporal difference (magnitude) for {map_name_base} - requires n_trans >= 2.")

            temp_phase = analysis_set.get('phase')
            if 'temporal_diff_phase' in analyses_to_run:
                if temp_phase is None:
                     temp_phase = calculate_phase(inv_map_nu) # Calculate if needed

                td_phase = calculate_temporal_difference(temp_phase)
                if td_phase is not None:
                    analysis_set['temporal_diff_phase'] = td_phase
                    np.save(self.analysis_dir / f"{map_name_base}_temporal_diff_phase.npy", td_phase)
                    self.logger.debug(f"Computed and saved temporal difference (phase) for {map_name_base}")
                else:
                    self.logger.warning(f"Skipping temporal difference (phase) for {map_name_base} - requires n_trans >= 2.")
            
            # Store results for this map
            self.analysis_results[map_name_base] = analysis_set
            self.logger.debug(f"Completed analysis for map {i}") # Now inside loop

        self.logger.info(f"Completed analysis calculation for {len(self.analysis_results)} maps.")

        # Save the summary analysis_results dictionary based on save_format
        if save_format == 'hdf5':
            summary_file_path = self.analysis_dir / "analysis_summary.h5"
            try:
                with h5py.File(summary_file_path, 'w') as f:
                    self.logger.info(f"Saving analysis results dictionary to HDF5 file: {summary_file_path}")
                    self._save_dict_to_hdf5(f, '', self.analysis_results)
                    self.logger.info("Successfully saved analysis results to HDF5.")
            except Exception as e:
                self.logger.error(f"Failed to save analysis results to HDF5 file {summary_file_path}: {e}")
        elif save_format == 'npz':
            self.logger.info("Individual analysis results saved as .npy files. Caller is responsible for creating a summary .npz file if needed.")
        else:
            self.logger.warning(f"Unsupported save_format '{save_format}'. No summary file generated. Individual .npy files were still saved.")

    def process_map(self, n_centers=2, radius=0.5, analyses_to_run=['magnitude'], k_neighbors_local_var=5):
        """Run the main processing steps: FFT, masks, inverse, gradients, and analysis."""
        self.compute_forward_fft()
        self.generate_kspace_masks(n_centers=n_centers, radius=radius)
        self.compute_inverse_maps()
        self.compute_gradient_maps() # Gradient maps (interpolated)
        self.analyze_inverse_maps(analyses_to_run=analyses_to_run, k_neighbors=k_neighbors_local_var) # Run specified analyses

        self.logger.info("Map processing pipeline complete.") 