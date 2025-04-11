"""
Provides a wrapper around MapBuilder to add progress bars for long-running operations.
"""

import numpy as np
from tqdm.auto import tqdm  # Use tqdm.auto for flexible environment support (notebook, console)
from .map_builder import MapBuilder
from .map_analysis import calculate_magnitude, calculate_phase, calculate_local_variance, calculate_temporal_difference # Needed for analyze_inverse_maps override

class ProgressMapBuilder(MapBuilder):
    """
    A wrapper around MapBuilder that adds tqdm progress bars to potentially
    long-running operations like computing inverse maps, gradients, and analysis.

    Use this class instead of MapBuilder when progress visualization is desired.
    The core MapBuilder functionality remains unchanged.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ProgressMapBuilder, passing arguments to MapBuilder."""
        # Initialize the parent MapBuilder class
        super().__init__(*args, **kwargs)
        self.logger.info("ProgressMapBuilder initialized, progress bars enabled for relevant steps.")

    def compute_inverse_maps(self):
        """
        Compute inverse maps for each k-space mask, showing progress.
        Overrides MapBuilder.compute_inverse_maps.
        """
        if not hasattr(self, 'kspace_masks') or not self.kspace_masks:
             self.logger.warning("No k-space masks generated yet. Cannot compute inverse maps.")
             # Optionally, call a default mask generation or raise error, matching parent behavior if desired.
             # For now, just return to avoid errors if masks are expected to be generated manually.
             return
             # If default generation is desired:
             # self.logger.info("No masks found, generating default masks first.")
             # self.generate_kspace_masks() # Assuming default parameters are acceptable

        if self.fft_result is None:
            self.logger.error("Forward FFT must be computed before inverse maps.")
            raise RuntimeError("Forward FFT must be computed before inverse maps.")

        self.inverse_maps = []
        self.logger.info(f"Computing inverse maps for {len(self.kspace_masks)} masks...")

        # Wrap the loop with tqdm
        for i, mask in enumerate(tqdm(self.kspace_masks, desc="Computing Inverse Maps", unit="mask")):
            # Apply 3D mask to 4D FFT result using broadcasting
            masked_fft = self.fft_result * mask

            # Execute inverse transform
            inverse_map_flat = self.inverse_plan.execute(masked_fft)

            # Reshape result back to (n_trans, n_points)
            inverse_map = inverse_map_flat.reshape(self.n_trans, self.n_points)

            # Store the result
            self.inverse_maps.append(inverse_map)
            
            # Save data to file (matching parent behavior)
            np.save(
                self.data_dir / f"inverse_map_{i}.npy",
                inverse_map
            )
            # self.logger.debug(f"Computed and saved inverse map {i}") # tqdm provides progress

        self.logger.info(f"Completed inverse map computation for {len(self.inverse_maps)} masks.")

    def compute_gradient_maps(self):
        """
        Compute gradient maps by interpolating inverse maps onto a grid, showing progress.
        Overrides MapBuilder.compute_gradient_maps.
        """
        if not self.inverse_maps:
            self.logger.error("No inverse maps computed yet. Cannot compute gradient maps.")
            # raise ValueError("Inverse maps must be computed before gradient maps") # Or just return
            return

        self.gradient_maps = []
        self.logger.info(f"Computing gradient maps for {len(self.inverse_maps)} inverse maps...")

        # Define source points (non-uniform) - Reuse from parent or re-define if needed
        points_nu = self.points_nu # Assuming points_nu is available from parent __init__

        # Define target grid points (uniform) - Reuse from parent or re-define
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(self.x_coords_1d.min(), self.x_coords_1d.max(), self.nx),
            np.linspace(self.y_coords_1d.min(), self.y_coords_1d.max(), self.ny),
            np.linspace(self.z_coords_1d.min(), self.z_coords_1d.max(), self.nz),
            indexing='ij'
        )
        points_u = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=-1)
        
        # Wrap the loop with tqdm
        for i, inverse_map_nu in enumerate(tqdm(self.inverse_maps, desc="Computing Gradient Maps", unit="map")):
            interpolated_maps_grid = np.zeros((self.n_trans, self.nx, self.ny, self.nz), dtype=self.dtype)

            # Inner loop for interpolation (usually fast, might not need progress bar)
            for t in range(self.n_trans):
                fill_val = np.mean(inverse_map_nu[t])
                # Import griddata if not imported globally
                from scipy.interpolate import griddata 
                interpolated_data_flat = griddata(
                    points_nu, 
                    inverse_map_nu[t],
                    points_u, 
                    method='linear',
                    fill_value=fill_val
                )
                interpolated_maps_grid[t] = interpolated_data_flat.reshape(self.nx, self.ny, self.nz)

            # Calculate gradients
            try:
                dz, dy, dx = np.gradient(interpolated_maps_grid, axis=(1, 2, 3))
                gradient_magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2)

                self.gradient_maps.append(gradient_magnitude)
                np.save(self.data_dir / f"gradient_map_{i}.npy", gradient_magnitude)
                # self.logger.debug(f"Computed and saved gradient map {i}")
            except Exception as e:
                self.logger.error(f"Error calculating gradient for map {i}: {e}")
                # Decide how to handle errors, maybe append None or re-raise

        self.logger.info(f"Completed gradient map computation for {len(self.gradient_maps)} maps.")

    def analyze_inverse_maps(self, analyses_to_run=['magnitude', 'phase'], k_neighbors=5):
        """
        Performs selected analyses on the computed non-uniform inverse maps, showing progress.
        Overrides MapBuilder.analyze_inverse_maps.
        """
        if not self.inverse_maps:
            self.logger.error("No inverse maps computed yet. Cannot run analysis.")
            return

        self.logger.info(f"Starting analysis of {len(self.inverse_maps)} inverse maps. Analyses requested: {analyses_to_run}")
        self.analysis_results = {} # Clear previous results

        # Wrap the main loop with tqdm
        for i, inv_map_nu in enumerate(tqdm(self.inverse_maps, desc="Analyzing Inverse Maps", unit="map")):
            analysis_set = {} # Store results for this specific inverse map
            map_name_base = f"map_{i}"

            # --- Simple Analyses (Magnitude, Phase) --- 
            if 'magnitude' in analyses_to_run:
                magnitude = calculate_magnitude(inv_map_nu)
                analysis_set['magnitude'] = magnitude
                np.save(self.analysis_dir / f"{map_name_base}_magnitude.npy", magnitude)
            
            if 'phase' in analyses_to_run:
                phase = calculate_phase(inv_map_nu)
                analysis_set['phase'] = phase
                np.save(self.analysis_dir / f"{map_name_base}_phase.npy", phase)

            # --- Local Variance --- 
            if 'local_variance' in analyses_to_run:
                # Need points_nu from parent class
                local_var = calculate_local_variance(inv_map_nu, self.points_nu, k=k_neighbors)
                analysis_set['local_variance'] = local_var
                np.save(self.analysis_dir / f"{map_name_base}_local_variance_k{k_neighbors}.npy", local_var)

            # --- Temporal Differences --- 
            temp_magnitude = analysis_set.get('magnitude')
            temp_phase = analysis_set.get('phase')
            
            if 'temporal_diff_magnitude' in analyses_to_run:
                if temp_magnitude is None:
                     temp_magnitude = calculate_magnitude(inv_map_nu) # Calculate if needed only for this
                
                td_mag = calculate_temporal_difference(temp_magnitude)
                if td_mag is not None:
                    analysis_set['temporal_diff_magnitude'] = td_mag
                    np.save(self.analysis_dir / f"{map_name_base}_temporal_diff_magnitude.npy", td_mag)

            if 'temporal_diff_phase' in analyses_to_run:
                if temp_phase is None:
                    temp_phase = calculate_phase(inv_map_nu) # Calculate if needed only for this

                td_phase = calculate_temporal_difference(temp_phase)
                if td_phase is not None:
                    analysis_set['temporal_diff_phase'] = td_phase
                    np.save(self.analysis_dir / f"{map_name_base}_temporal_diff_phase.npy", td_phase)
            
            # Store results for this map
            self.analysis_results[map_name_base] = analysis_set
            # self.logger.debug(f"Completed analysis for map {i}")

        self.logger.info(f"Completed analysis for {len(self.analysis_results)} maps.")

    # --- Other MapBuilder methods like compute_forward_fft, mask generation etc. ---
    # --- are inherited directly and will NOT show progress bars unless overridden ---

    # Example: Override compute_forward_fft if simple start/end notification is desired
    # def compute_forward_fft(self):
    #     self.logger.info("Starting forward FFT computation...")
    #     # Use tqdm for unknown duration if desired
    #     # with tqdm(total=1, desc="Computing Forward FFT") as pbar:
    #     super().compute_forward_fft() # Call the original method
    #         # pbar.update(1)
    #     self.logger.info("Forward FFT computation finished.") 