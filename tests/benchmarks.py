import numpy as np
import time
from pathlib import Path
import sys
import json
from datetime import datetime

# Ensure the package directory is in the Python path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))

from QM_FFT_Analysis.utils.map_builder import MapBuilder
from QM_FFT_Analysis.utils.preprocessing import get_normalized_wavefunctions_at_times

class Benchmark:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def generate_test_data(self, n_sources, n_time, n_selected_times):
        """Generate test data for benchmarking."""
        # Generate time series data
        time_series_data = np.random.randn(n_sources, n_time)
        
        # Select time points
        time_indices = np.linspace(0, n_time-1, n_selected_times, dtype=int)
        
        # Generate coordinates
        x = np.random.uniform(-np.pi, np.pi, n_sources)
        y = np.random.uniform(-np.pi, np.pi, n_sources)
        z = np.random.uniform(-np.pi, np.pi, n_sources)
        
        # Normalize wavefunctions
        strengths = get_normalized_wavefunctions_at_times(
            time_series_data,
            time_indices,
            time_axis=1,
            source_axis=0
        )
        
        return {
            "x": x,
            "y": y,
            "z": z,
            "strengths": strengths,
            "n_sources": n_sources,
            "n_time": n_time,
            "n_selected_times": n_selected_times
        }
    
    def benchmark_preprocessing(self, data):
        """Benchmark preprocessing operations."""
        start_time = time.time()
        
        normalized_wavefunctions = get_normalized_wavefunctions_at_times(
            data["time_series_data"],
            data["time_indices"],
            time_axis=1,
            source_axis=0
        )
        
        preprocessing_time = time.time() - start_time
        return {
            "preprocessing_time": preprocessing_time,
            "n_sources": data["n_sources"],
            "n_time": data["n_time"],
            "n_selected_times": data["n_selected_times"]
        }
    
    def benchmark_mapbuilder(self, data, grid_size=None):
        """Benchmark MapBuilder operations."""
        results = {}
        
        # Initialize MapBuilder
        map_builder = MapBuilder(
            subject_id="benchmark",
            output_dir=self.output_dir / "temp",
            x=data["x"],
            y=data["y"],
            z=data["z"],
            strengths=data["strengths"],
            eps=1e-6,
            dtype='complex128',
            estimate_grid=True if grid_size is None else False,
            normalize_fft_result=True
        )
        
        if grid_size is not None:
            map_builder.grid_size = grid_size
        
        # Benchmark forward FFT
        start_time = time.time()
        map_builder.compute_forward_fft()
        results["forward_fft_time"] = time.time() - start_time
        
        # Benchmark mask generation
        start_time = time.time()
        map_builder.generate_kspace_masks(
            mask_type='spherical',
            radius_range=(0.1, 0.4),
            num_masks=3
        )
        results["mask_generation_time"] = time.time() - start_time
        
        # Benchmark inverse map computation
        start_time = time.time()
        map_builder.compute_inverse_maps()
        results["inverse_map_time"] = time.time() - start_time
        
        # Benchmark gradient map computation
        start_time = time.time()
        map_builder.compute_gradient_maps()
        results["gradient_map_time"] = time.time() - start_time
        
        # Benchmark analysis
        start_time = time.time()
        map_builder.analyze_inverse_maps(
            analysis_types=['magnitude', 'phase', 'local_variance'],
            k_variance=5
        )
        results["analysis_time"] = time.time() - start_time
        
        # Add metadata
        results["n_sources"] = data["n_sources"]
        results["n_time"] = data["n_time"]
        results["n_selected_times"] = data["n_selected_times"]
        results["grid_size"] = map_builder.grid_size
        
        return results
    
    def run_benchmarks(self, test_cases):
        """Run benchmarks for different test cases."""
        for case_name, params in test_cases.items():
            print(f"Running benchmark for {case_name}...")
            
            # Generate test data
            data = self.generate_test_data(**params)
            
            # Run benchmarks
            preprocessing_results = self.benchmark_preprocessing(data)
            mapbuilder_results = self.benchmark_mapbuilder(data)
            
            # Store results
            self.results[case_name] = {
                "preprocessing": preprocessing_results,
                "mapbuilder": mapbuilder_results,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_results(self):
        """Save benchmark results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")

def main():
    # Define test cases
    test_cases = {
        "small": {
            "n_sources": 20,
            "n_time": 100,
            "n_selected_times": 5
        },
        "medium": {
            "n_sources": 50,
            "n_time": 200,
            "n_selected_times": 10
        },
        "large": {
            "n_sources": 100,
            "n_time": 500,
            "n_selected_times": 20
        }
    }
    
    # Run benchmarks
    benchmark = Benchmark()
    benchmark.run_benchmarks(test_cases)
    benchmark.save_results()

if __name__ == "__main__":
    main() 