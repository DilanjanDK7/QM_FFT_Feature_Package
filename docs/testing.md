# Testing and Benchmarking Guide

This document provides instructions for running tests and benchmarks in the QM_FFT_Analysis package.

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

## Continuous Integration

The package uses GitHub Actions for continuous integration:

1. Tests run on pull requests
2. Benchmarks run on main branch
3. Coverage reports are generated
4. Results are stored as artifacts

## Contact

For issues or questions about testing and benchmarking:

1. Open an issue on GitHub
2. Contact the maintainers
3. Check the troubleshooting guide 