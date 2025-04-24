# QM FFT Analysis Package

A high-performance package for analyzing 3D non-uniform data using Non-Uniform Fast Fourier Transforms (NUFFT), with advanced spectral metrics and optimization features.

## Overview

This package provides tools for transforming and analyzing scattered 3D point data with complex strengths. It's designed for quantum mechanical analysis, neuroimaging, and other scientific applications requiring frequency-domain processing of non-uniform data.

## Key Features

* **Non-Uniform FFT Processing**: Transform between non-uniform points and regular k-space grid
* **K-Space Masking**: Isolate specific frequency components with spherical masks
* **Gradient Calculation**: Compute spatial gradients using standard or accelerated analytical methods
* **Advanced Spectral Metrics**: Spectral slope, entropy, anisotropy, and neural activity estimation
* **Performance Optimizations**: Analytical gradient method (up to 2.3x faster) and enhanced-only pipeline (up to 9x faster)
* **Flexible Storage**: Efficient HDF5-based organization with compression support
* **Comprehensive Logging**: Detailed progress tracking and error handling

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd QM_FFT_Feature_Package

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

See [HOW-TO.md](./HOW-TO.md) for detailed installation instructions.

## Quick Start

```python
import numpy as np
from pathlib import Path
from QM_FFT_Analysis.utils import MapBuilder

# Prepare data (coordinates and complex strengths)
n_points, n_times = 1000, 5
x = np.random.uniform(-np.pi, np.pi, n_points)
y = np.random.uniform(-np.pi, np.pi, n_points)
z = np.random.uniform(-np.pi, np.pi, n_points)
strengths = np.random.randn(n_times, n_points) + 1j * np.random.randn(n_times, n_points)

# Initialize
builder = MapBuilder(
    subject_id="example_subject",
    output_dir=Path("./output"),
    x=x, y=y, z=z,
    strengths=strengths,
    enable_enhanced_features=True
)

# Full pipeline
builder.process_map(
    n_centers=2,
    analyses_to_run=['magnitude', 'phase', 'spectral_slope', 'anisotropy'],
    use_analytical_gradient=True  # Faster gradient calculation
)

# Or compute only enhanced metrics (5-9x faster)
enhanced_metrics = builder.compute_enhanced_metrics(
    metrics_to_run=['spectral_slope', 'spectral_entropy', 'anisotropy']
)
```

## Performance Benchmarks

The package includes a comprehensive benchmarking system in the `benchmarks/` directory:

```bash
# Compare enhanced-only vs full pipeline (4-9x speedup)
python benchmarks/benchmark_enhanced_only.py

# Test analytical gradient performance (1.4-2.3x speedup)
python benchmarks/benchmark_analytical_gradient.py

# Evaluate with extremely large datasets
python benchmarks/benchmark_extreme_gradient.py
```

See [benchmarks/README.md](benchmarks/README.md) for detailed information on all available benchmarks and how to interpret the results.

## Documentation

- [HOW-TO.md](./HOW-TO.md): Detailed installation and usage guide
- [Technical Reference](docs/technical_reference.md): In-depth explanation of algorithms and methods
- [Enhanced Features Guide](docs/enhanced_features_guide.md): Advanced features documentation

## Output Structure

The package generates three HDF5 files for each subject:

1. **data.h5**: Raw computational results (FFT, masks, inverse maps)
2. **analysis.h5**: Analysis results (magnitude, phase, variance, etc.)
3. **enhanced.h5**: Advanced metrics (spectral slope, entropy, anisotropy, etc.)

## Developer Information

**Developer:** Dilanjan DK  
**Contact:** ddiyabal@uwo.ca

## License

This package is private and copyrighted. All rights reserved.

---

For detailed usage examples and performance optimization tips, see the [HOW-TO.md](./HOW-TO.md) guide.
