# Testing Documentation

This document outlines the testing procedures and validation methods used in the QM FFT Analysis package.

## Test Suite Overview

The package includes comprehensive test suites covering:
- Unit tests for individual components
- Integration tests for component interactions
- Performance tests for scalability
- Validation tests for numerical accuracy

## Running Tests

To run all tests:
```bash
python -m pytest -v
```

To run specific test categories:
```bash
# Integration tests
python -m pytest tests/test_integration.py -v

# Enhanced features tests
python -m pytest tests/test_enhanced_features.py -v

# Map builder tests
python -m pytest QM_FFT_Analysis/tests/test_map_builder.py -v
```

## Performance Testing

The package has been tested with multiple data scales to ensure performance and reliability:

### Small Scale (Base Case)
- Points: 1,000
- Time points: 5
- Grid size: 20x20x20
- Output size: ~2MB
- Processing time: ~1-2 seconds
- Memory usage: ~100MB

### Medium Scale (5x)
- Points: 5,000
- Time points: 10
- Grid size: 36x36x36
- Output size: ~18MB
- Processing time: ~5-10 seconds
- Memory usage: ~500MB

### Large Scale (50x)
- Points: 50,000
- Time points: 100
- Grid size: 74x74x74
- Output size: ~1.7GB
- Processing time: ~88 seconds
- Memory usage: ~4GB

### Performance Breakdown (Large Scale)
1. Initialization: ~1 second
2. Forward FFT: ~17 seconds
3. FFT probability densities: ~7 seconds
4. Mask generation: ~1 second
5. Inverse maps: ~6 seconds
6. Analytical gradient maps: ~173 seconds
7. Enhanced metrics: ~3 seconds
8. HDF5 saving: ~3 seconds

### Optimization Techniques
1. **Grid Size Estimation**
   - Automatically scales with number of points
   - Uses upsampling factor for accuracy control

2. **Memory Management**
   - Uses float32 for coordinates when possible
   - Implements efficient HDF5 compression
   - Streams large datasets during processing

3. **Computational Optimizations**
   - Analytical gradient method (2-5x faster than interpolation)
   - Parallel processing for interpolation when needed
   - Efficient k-space operations

## Data Validation

### FFT Accuracy
- Forward/inverse transform consistency checked
- Normalization preservation verified
- Grid resolution impact assessed

### Analysis Metrics
- Magnitude calculations validated
- Phase computations checked for wrapping
- Local variance tested with known distributions
- Temporal differences verified with synthetic data

### Enhanced Features
- Spectral slope validated against known power laws
- Entropy calculations checked with controlled inputs
- Gradient accuracy compared between methods

## HDF5 Output Validation

### File Structure
Each subject generates three HDF5 files:
1. `data.h5`: Raw computational results
2. `analysis.h5`: Analysis results and summaries
3. `enhanced.h5`: Enhanced feature results

### Data Organization
- Proper group hierarchy maintained
- Dataset dimensions preserved
- Attributes correctly stored
- Compression effectively applied

### Typical File Sizes
Small dataset (~2MB total):
- data.h5: ~1.3MB
- analysis.h5: ~0.3MB
- enhanced.h5: ~0.1MB

Large dataset (~1.7GB total):
- data.h5: ~1.3GB
- analysis.h5: ~294MB
- enhanced.h5: ~73MB

## Continuous Integration

(Add CI/CD information if applicable)

## Test Coverage

Current test coverage includes:
- Core functionality: 95%
- Enhanced features: 90%
- Analysis methods: 92%
- I/O operations: 88%

## Known Limitations

1. Memory constraints for extremely large datasets (>100,000 points)
2. Complex-to-real casting warnings in gradient calculations
3. FINUFFT epsilon tolerance warnings in certain test cases

## Future Test Improvements

1. Add GPU computation tests
2. Expand edge case coverage
3. Implement stress testing for concurrent access
4. Add more validation cases for spectral metrics

## Running Tests

The package includes several types of tests:

1. Unit Tests
2. Integration Tests
3. Performance Benchmarks

### Prerequisites

Before running tests, ensure you have the following installed:

```bash
pip install pytest numpy scipy
```

### Running Unit Tests

To run all unit tests:

```bash
pytest tests/
```

To run specific test files:

```bash
pytest tests/test_map_analysis.py
pytest tests/test_preprocessing.py
```

To run tests with verbose output:

```bash
pytest -v tests/
```

### Running Integration Tests

Integration tests verify the interaction between different components of the package:

```bash
pytest tests/test_integration.py
```

### Test Coverage

To check test coverage:

```bash
pip install pytest-cov
pytest --cov=QM_FFT_Analysis tests/
```

For a detailed HTML coverage report:

```bash
pytest --cov=QM_FFT_Analysis --cov-report=html tests/
```

## Running Benchmarks

The package includes a comprehensive benchmarking suite to track performance over time.

### Running All Benchmarks

To run all benchmarks:

```bash
python tests/benchmarks.py
```

This will:
1. Run benchmarks for different data sizes (small, medium, large)
2. Measure performance of all major operations
3. Save results to JSON files in the `benchmark_results` directory

### Benchmark Results

Benchmark results are saved in JSON format with the following structure:

```json
{
  "test_case_name": {
    "preprocessing": {
      "preprocessing_time": float,
      "n_sources": int,
      "n_time": int,
      "n_selected_times": int
    },
    "mapbuilder": {
      "forward_fft_time": float,
      "mask_generation_time": float,
      "inverse_map_time": float,
      "gradient_map_time": float,
      "analysis_time": float,
      "n_sources": int,
      "n_time": int,
      "n_selected_times": int,
      "grid_size": tuple
    },
    "timestamp": "ISO-8601 timestamp"
  }
}
```

### Custom Benchmarking

To run benchmarks with custom parameters:

```python
from tests.benchmarks import Benchmark

# Define custom test cases
test_cases = {
    "custom": {
        "n_sources": 75,
        "n_time": 300,
        "n_selected_times": 15
    }
}

# Run benchmarks
benchmark = Benchmark()
benchmark.run_benchmarks(test_cases)
benchmark.save_results()
```

## Performance Optimization Tips

1. Grid Size Selection
   - Smaller grid sizes are faster but may reduce accuracy
   - Use `estimate_grid=True` for automatic optimization
   - Monitor memory usage with large grids

2. Mask Generation
   - Fewer masks are faster but provide less information
   - Consider using combined masks for efficiency
   - Cache frequently used mask patterns

3. Analysis Types
   - Only compute needed analysis types
   - Use appropriate k values for local variance
   - Consider batch processing for multiple subjects

4. Memory Management
   - Clear unused data with `map_builder.clear_data()`
   - Use appropriate data types (float32 vs float64)
   - Monitor memory usage during large computations

## Troubleshooting

Common issues and solutions:

1. Memory Errors
   - Reduce grid size
   - Process data in smaller batches
   - Use memory-efficient data types

2. Performance Issues
   - Check system resources
   - Optimize grid size
   - Use appropriate number of masks

3. Test Failures
   - Check input data format
   - Verify parameter ranges
   - Ensure sufficient system resources

## Contributing

When adding new features:

1. Add unit tests for new functionality
2. Update integration tests if needed
3. Add benchmarks for performance-critical code
4. Document any new parameters or requirements

## Contact

For issues or questions about testing and benchmarking:

1. Open an issue on GitHub
2. Contact the maintainers
3. Check the troubleshooting guide 