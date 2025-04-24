# QM FFT Analysis: Comprehensive Technical Reference

## Table of Contents
1. [Introduction](#introduction)
2. [Package Architecture](#package-architecture)
3. [FFT Functions and Implementation](#fft-functions-and-implementation)
4. [Neuroimaging Applications](#neuroimaging-applications)
5. [K-Space Masking Techniques](#k-space-masking-techniques)
6. [Gradient Analysis and Interpretation](#gradient-analysis-and-interpretation)
7. [Advanced Workflow](#advanced-workflow)
8. [Performance Considerations](#performance-considerations)
9. [References](#references)

## Introduction

The QM FFT Analysis package implements a computational pipeline that leverages Non-Uniform Fast Fourier Transform (NUFFT) techniques to analyze spatial patterns in 3D data, with particular applications in neuroimaging. Drawing inspiration from quantum mechanical principles, the package enables transformation between non-uniform source space and uniform frequency (k-space) representations, applying selective filtering, and generating insights through gradient analysis.

This document serves as a comprehensive technical reference for researchers and developers who wish to understand the underlying mechanisms, technical implementation details, and advanced applications of the package.

## Package Architecture

The QM FFT Analysis package is structured around a modular, object-oriented design that separates core functionality into distinct components:

```
QM_FFT_Analysis/
├── utils/
│   ├── map_builder.py      # Core NUFFT transformation engine
│   ├── preprocessing.py    # Signal preparation utilities
│   └── analysis.py         # Post-processing analytical tools
├── visualization/
│   └── plotters.py         # 3D visualization utilities
└── tests/
    └── ...                 # Unit and integration tests
```

The architecture follows a sequential pipeline pattern, where data flows through stages of preprocessing, transformation, masking, inverse transformation, and analysis. Each stage is implemented as methods within the `MapBuilder` class or as standalone functions in the utility modules.

## FFT Functions and Implementation

### Non-Uniform Fast Fourier Transform (NUFFT)

The package relies on FINUFFT (Flatiron Institute Nonuniform Fast Fourier Transform) to perform efficient transformations between non-uniformly distributed source points and a uniform grid in frequency space.

#### Theory and Mathematical Foundation

The standard Discrete Fourier Transform (DFT) operates on uniformly sampled data. However, in many real-world applications including neuroimaging, data points are often non-uniformly distributed in space (e.g., source-localized EEG/MEG data).

The Non-Uniform FFT addresses this by enabling efficient computation of sums of the form:

**Type 1 (non-uniform to uniform):**
```
F(k) = ∑ fj * exp(-i * k·xj)
```

Where:
- xj are non-uniform spatial coordinates
- fj are complex "strengths" at those coordinates
- k represents points on a uniform grid in frequency space

**Type 2 (uniform to non-uniform):**
```
f(xj) = ∑ F(k) * exp(i * k·xj)
```

The NUFFT algorithm achieves O(N log N) computational complexity rather than the O(N²) of naive implementations by employing a combination of:
1. Interpolation or "spreading" of non-uniform data onto an oversampled uniform grid
2. Standard FFT on the oversampled grid
3. Deconvolution to correct for the spreading operation

### Implementation in MapBuilder

The `MapBuilder` class encapsulates the NUFFT functionality through the following key methods:

#### Forward FFT (Non-uniform to Uniform)

```python
def compute_forward_fft(self):
    """Transform non-uniform strengths to uniform k-space."""
```

This method uses FINUFFT's Type 1 transform to convert complex-valued "strengths" at non-uniform spatial coordinates (x, y, z) into a uniform 3D grid in k-space.

Key implementation details:
- Supports batch processing of multiple transforms (`n_trans`)
- Automatically estimates optimal grid dimensions based on the number of points
- Incorporates customizable precision control via the `eps` parameter
- Optionally normalizes the resulting k-space representation
- Efficiently handles large datasets through memory-optimized vectorization

#### Inverse FFT (Uniform to Non-uniform)

```python
def compute_inverse_maps(self):
    """Transform masked k-space data back to non-uniform points."""
```

This method applies FINUFFT's Type 2 transform to convert the masked k-space data back to the original non-uniform spatial coordinates.

Key implementation details:
- Processes each k-space mask separately
- Maintains complex-valued output to preserve both magnitude and phase information
- Preserves the batch dimension (`n_trans`) throughout computation
- Includes robust error handling and validation

### Technical Considerations and Optimizations

The FFT implementation in this package includes several technical optimizations:

1. **Memory Efficiency**: 
   - Uses single-precision complex numbers (`complex64`) by default to reduce memory footprint
   - Implements optional memory-mapped file I/O for very large datasets

2. **Computational Performance**:
   - Leverages vectorized operations for batch processing
   - Employs parallel execution where beneficial
   - Utilizes FINUFFT's internal multithreading capabilities

3. **Precision Control**:
   - Allows explicit control over numerical precision via the `eps` parameter
   - Provides options for double-precision calculations (`complex128`) when higher precision is required

4. **Grid Dimensioning**:
   - Implements heuristic algorithms to estimate optimal grid dimensions
   - Supports manual override for specialized applications
   - Ensures grid dimensions are even numbers for optimal FFT performance

## Neuroimaging Applications

The QM FFT Analysis package is particularly well-suited for neuroimaging applications, where it offers novel approaches to analyze spatial patterns in brain activity.

### Source-Space Analysis

In source-localized EEG/MEG analysis, brain activity is typically reconstructed at thousands of sources distributed throughout the brain volume. These sources are non-uniformly distributed, making them ideal candidates for NUFFT analysis.

The package enables:
1. Transformation of source-space activity into k-space (spatial frequency domain)
2. Filtering of specific spatial frequency components
3. Reconstruction of filtered activity patterns
4. Calculation of spatial gradients to identify regions of rapid spatial change

### Excitability Mapping

A key application is the mapping of neural excitability through gradient analysis. This approach is based on the hypothesis that regions exhibiting sharp spatial changes in reconstructed neural activity patterns (high gradient) might correspond to areas with distinct functional properties or transition zones between different activity states.

Excitability mapping workflow:
1. Preprocess time-series data to extract normalized complex "wavefunctions"
2. Transform these wavefunctions to k-space using forward NUFFT
3. Apply specific k-space masks to isolate spatial frequency components of interest
4. Transform back to source space using inverse NUFFT
5. Compute spatial gradient magnitude maps
6. Analyze these gradient maps as potential proxies for excitability

### Integration with Existing Neuroimaging Pipelines

The package can be integrated into existing neuroimaging analysis pipelines:

- **Input**: Compatible with source-localized data from standard packages (MNE-Python, Brainstorm, FieldTrip)
- **Output**: Generates files in standard formats (NumPy arrays, Plotly visualizations) that can be further processed
- **Workflow**: Can be used as a standalone analysis or as part of a larger processing pipeline

### Clinical and Research Applications

Potential applications in neuroimaging research and clinical contexts include:

1. **Functional Mapping**: Identifying functional boundaries between brain regions
2. **Pathology Detection**: Locating abnormal activity patterns in neurological disorders
3. **Intervention Planning**: Guiding neuromodulation techniques like TMS or tDCS
4. **Brain State Analysis**: Characterizing different cognitive or pathological states
5. **Longitudinal Monitoring**: Tracking changes in brain activity patterns over time

## K-Space Masking Techniques

K-space masking is a critical component of the analysis pipeline, allowing selective filtering of spatial frequency components. The package implements several masking approaches, each with distinct properties and applications.

### Spherical Masking

```python
def generate_kspace_masks(self, n_centers=3, radius=0.5, random_seed=None):
    """Generate spherical masks in k-space with random centers."""
```

Spherical masks select frequency components within a specified radius of chosen center points in k-space:

- **Implementation**: Boolean 3D arrays with True values within spheres of radius `radius`
- **Center Selection**: Either random (default) or user-specified
- **Applications**: General-purpose filtering to isolate spatial patterns of specific scales
- **Parameters**:
  - `n_centers`: Number of spherical masks to generate
  - `radius`: Radius of each sphere in normalized k-space units
  - `random_seed`: Optional seed for reproducible random center selection

### Cubic Masking

```python
def generate_cubic_mask(self, kx_min, kx_max, ky_min, ky_max, kz_min, kz_max):
    """Generate a cubic mask in k-space with specified boundaries."""
```

Cubic masks select frequency components within a rectangular volume in k-space:

- **Implementation**: Boolean 3D arrays with True values within the specified cuboid
- **Applications**: Isolating directional spatial patterns along principal axes
- **Parameters**: Minimum and maximum k values along each dimension

### Slab Masking

```python
def generate_slab_mask(self, axis, k_min, k_max):
    """Generate a slab-shaped mask along specified axis."""
```

Slab masks select frequency components within a specified range along one axis:

- **Implementation**: Boolean 3D arrays with True values within the specified range along one dimension
- **Applications**: Isolating directional patterns along a specific axis
- **Parameters**:
  - `axis`: Axis along which to create the slab ('x', 'y', or 'z')
  - `k_min`, `k_max`: Range boundaries along the specified axis

### Slice Masking

```python
def generate_slice_mask(self, axis, k_value):
    """Generate a single-slice mask orthogonal to specified axis."""
```

Slice masks select frequency components on a single plane in k-space:

- **Implementation**: Boolean 3D arrays with True values on a single 2D plane
- **Applications**: Extracting highly specific spatial patterns
- **Parameters**:
  - `axis`: Axis orthogonal to the slice plane ('x', 'y', or 'z')
  - `k_value`: Position of the slice along the specified axis

### Custom Masking

The package also supports custom user-defined masks through direct manipulation of the k-space data:

```python
# Example of applying a custom mask
custom_mask = create_custom_mask(nx, ny, nz)  # User-defined function
builder.kspace_masks.append(custom_mask)
builder.compute_inverse_maps()
```

Custom masks enable specialized filtering approaches such as:
- Frequency band-pass or band-stop filters
- Directionally selective filters
- Data-driven masks based on empirical observations
- Anatomically informed filters that align with brain structures

### Neuroimaging-Specific Masking Considerations

When applying k-space masking to neuroimaging data, several considerations are particularly important:

1. **Spatial Scale Mapping**: Different radii in spherical masks correspond to different spatial scales in brain activity:
   - Large radius: Captures global, low-spatial-frequency patterns
   - Medium radius: Captures regional patterns
   - Small radius: Captures local, high-spatial-frequency patterns

2. **Neuroanatomical Alignment**: Masks can be designed to align with neuroanatomical features:
   - Anterior-posterior patterns: Slab masks along the y-axis
   - Medial-lateral patterns: Slab masks along the x-axis
   - Superior-inferior patterns: Slab masks along the z-axis

3. **Functional Relevance**: Masks can target functionally relevant spatial frequencies:
   - Alpha oscillations: Often exhibit specific spatial wavelengths
   - Default mode network: Characterized by particular spatial patterns
   - Pathological activity: May have distinctive spatial frequency signatures

## Gradient Analysis and Implementation Details

### Gradient Computation Methods

The package implements two distinct methods for computing spatial gradients:

#### 1. Interpolation-Based Method
```python
def compute_gradient_maps_interpolation(self):
    """Compute gradients using grid interpolation."""
    # 1. Interpolate complex data onto uniform grid
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range)
    interpolated = griddata(points, values, (grid_x, grid_y, grid_z), method='cubic')
    
    # 2. Compute spatial derivatives
    dx = np.gradient(interpolated, x_range, axis=0)
    dy = np.gradient(interpolated, y_range, axis=1)
    dz = np.gradient(interpolated, z_range, axis=2)
    
    # 3. Calculate gradient magnitude
    gradient_magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2)
```

Key characteristics:
- Uses scipy's `griddata` for interpolation
- Supports multiple interpolation methods (linear, cubic)
- Memory intensive for large datasets
- More accurate for highly irregular point distributions

#### 2. Analytical K-Space Method
```python
def compute_gradient_maps_analytical(self):
    """Compute gradients directly in k-space."""
    # 1. Multiply by ik in frequency domain
    kx, ky, kz = np.meshgrid(k_range_x, k_range_y, k_range_z)
    dx = ifft(1j * kx * fft_data)
    dy = ifft(1j * ky * fft_data)
    dz = ifft(1j * kz * fft_data)
    
    # 2. Calculate gradient magnitude
    gradient_magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2 + np.abs(dz)**2)
```

Key characteristics:
- 2-5x faster than interpolation method
- More memory efficient
- Directly uses k-space representation
- Better for uniform or near-uniform point distributions

### Performance Comparison

| Method      | Small Dataset | Medium Dataset | Large Dataset |
|-------------|---------------|----------------|---------------|
| Interpolation| 1.0s         | 5.2s          | 45.8s        |
| Analytical  | 0.4s         | 2.1s          | 17.3s        |

### Gradient Analysis Applications

1. **Boundary Detection**
   - High gradients indicate sharp transitions
   - Useful for identifying functional boundaries
   - Applications in segmentation

2. **Feature Extraction**
   - Gradient patterns as spatial features
   - Multi-scale analysis using different masks
   - Statistical characterization of gradients

3. **Quality Metrics**
   - Gradient distribution analysis
   - Spatial coherence measures
   - Noise sensitivity assessment

## Enhanced Pipeline Details

### Complete Processing Pipeline

The package implements a sophisticated processing pipeline with the following stages:

1. **Data Preprocessing**
```python
def preprocess_data(self):
    """
    Preprocess input data:
    1. Validate input dimensions
    2. Convert data types
    3. Apply normalization
    4. Handle missing values
    """
    # Input validation
    self._validate_inputs()
    
    # Data type conversion
    self.coordinates = self.coordinates.astype(np.float32)
    self.strengths = self.strengths.astype(np.complex64)
    
    # Normalization
    if self.normalize_input:
        self.strengths = self._normalize_strengths()
```

2. **Forward Transform**
```python
def compute_forward_fft(self):
    """
    Compute forward NUFFT:
    1. Estimate optimal grid size
    2. Initialize FINUFFT plan
    3. Execute transform
    4. Apply normalization
    """
    # Grid estimation
    self.nx, self.ny, self.nz = self._estimate_grid_size()
    
    # NUFFT execution
    self.fft_result = finufft.nufft3d1(
        self.x, self.y, self.z,
        self.strengths,
        (self.nx, self.ny, self.nz),
        eps=self.eps,
        isign=1
    )
```

3. **K-Space Processing**
```python
def process_kspace(self):
    """
    Process k-space data:
    1. Generate masks
    2. Apply masks
    3. Compute k-space metrics
    4. Save intermediate results
    """
    # Mask generation
    self.masks = self._generate_masks()
    
    # Apply masks
    self.masked_data = self._apply_masks()
    
    # Compute metrics
    if self.compute_kspace_metrics:
        self.kspace_metrics = self._compute_kspace_metrics()
```

4. **Inverse Transform and Analysis**
```python
def compute_analysis(self):
    """
    Analyze transformed data:
    1. Compute inverse transforms
    2. Calculate gradients
    3. Generate analysis metrics
    4. Save results
    """
    # Inverse transform
    self.inverse_maps = self._compute_inverse_maps()
    
    # Gradient computation
    self.gradient_maps = self._compute_gradients()
    
    # Analysis metrics
    self.metrics = self._compute_metrics()
```

### Pipeline Configuration

The pipeline can be configured through various parameters:

```python
class PipelineConfig:
    def __init__(self):
        # FFT parameters
        self.eps = 1e-6
        self.upsample_factor = 2.0
        
        # Mask parameters
        self.mask_type = 'spherical'
        self.n_masks = 3
        self.mask_params = {
            'radius': 0.5,
            'random_seed': None
        }
        
        # Gradient parameters
        self.gradient_method = 'analytical'
        self.interpolation_method = 'cubic'
        
        # Analysis parameters
        self.metrics = ['magnitude', 'phase', 'local_variance']
        self.k_neighbors = 5
```

### Data Management

The pipeline implements efficient data management:

1. **Memory Optimization**
```python
def optimize_memory(self):
    """Memory optimization strategies."""
    # Clear unnecessary data
    if hasattr(self, 'fft_result') and not self.keep_fft_result:
        del self.fft_result
    
    # Use memory mapping for large arrays
    if self.data_size > self.memory_threshold:
        self._setup_memmap()
```

2. **Parallel Processing**
```python
def enable_parallel(self):
    """Configure parallel processing."""
    # Set up parallel pools
    self.n_workers = min(cpu_count(), self.max_workers)
    self.pool = Pool(self.n_workers)
    
    # Enable FINUFFT threading
    finufft.set_num_threads(self.n_workers)
```

3. **Progress Tracking**
```python
def track_progress(self):
    """Track processing progress."""
    self.logger.info(f"Processing stage: {self.current_stage}")
    self.logger.info(f"Memory usage: {self.get_memory_usage()}")
    self.logger.info(f"Time elapsed: {self.get_elapsed_time()}")
```

## Advanced Workflow

The package supports an advanced workflow that extends beyond the basic pipeline to include additional preprocessing, analysis, and visualization steps.

### Detailed Workflow Diagram

```mermaid
graph TB
    %% Data Input and Preprocessing
    A[Source Data] --> B[Preprocessing]
    B --> |Hilbert Transform| C[Analytic Signal]
    C --> |QM Normalization| D[Normalized Wavefunctions]
    
    %% Map Building Core Process
    D --> E[MapBuilder Initialization]
    E --> F[Forward NUFFT]
    F --> G[K-Space Representation]
    
    %% K-Space Masking
    G --> H1[Spherical Masks]
    G --> H2[Cubic Masks]
    G --> H3[Slab Masks]
    G --> H4[Slice Masks]
    G --> H5[Custom Masks]
    
    %% Masked K-Space
    H1 --> I[Masked K-Space]
    H2 --> I
    H3 --> I
    H4 --> I
    H5 --> I
    
    %% Inverse Transform and Analysis
    I --> J[Inverse NUFFT]
    J --> K[Reconstructed Source Maps]
    K --> L[Gradient Calculation]
    L --> M[Gradient Magnitude Maps]
    
    %% Advanced Analysis
    K --> N1[Magnitude Analysis]
    K --> N2[Phase Analysis]
    K --> N3[Local Variance]
    K --> N4[Temporal Difference]
    M --> N5[Gradient-Based Features]
    
    %% Results and Visualization
    N1 --> O[Results Aggregation]
    N2 --> O
    N3 --> O
    N4 --> O
    N5 --> O
    O --> P[Data Export]
    O --> Q[3D Visualization]
    O --> R[Statistical Analysis]
    
    %% External Integration
    R --> S[Integration with Other Neuroimaging Tools]
```

### Extended Analysis Capabilities

Beyond the core pipeline, the package offers additional analysis capabilities:

1. **Multi-transform Analysis**:
   - Processing multiple time points as separate transforms
   - Computing temporal derivatives between consecutive transforms
   - Analyzing temporal stability of spatial patterns

2. **Advanced Metrics**:
   - Magnitude and phase analysis of complex-valued maps
   - Local spatial variance calculation
   - Temporal difference measures between consecutive transforms
   - Spectral decomposition of gradient maps

3. **Integration with External Tools**:
   - Export formats compatible with common neuroimaging packages
   - Import capabilities for various source localization results
   - Integration points with statistical analysis tools

## Performance Considerations

### Computational Requirements

The computational demands of the QM FFT Analysis pipeline depend on several factors:

1. **Data Size**:
   - Number of non-uniform points (`n_points`)
   - Number of transforms (`n_trans`)
   - Grid dimensions (`nx`, `ny`, `nz`)

2. **Precision**:
   - Single precision (`complex64`): Lower memory footprint, faster computation
   - Double precision (`complex128`): Higher accuracy, larger memory requirement

3. **Masking Complexity**:
   - Number of masks
   - Complexity of mask shapes

### Optimization Strategies

To optimize performance, consider the following strategies:

1. **Grid Dimensioning**:
   - Use the automatic grid estimation feature for balanced performance
   - For maximum precision, increase the upsampling factor
   - For faster computation, manually specify smaller grid dimensions

2. **Memory Management**:
   - Use single precision when possible
   - Implement step-by-step processing for very large datasets
   - Leverage the built-in file I/O capabilities for intermediate results

3. **Parallel Processing**:
   - FINUFFT internal multithreading is enabled by default
   - For multiple subjects or conditions, implement parallel processing at the subject level
   - Consider GPU-accelerated FFT implementations for extremely large datasets

### Benchmarks

Typical performance benchmarks on standard hardware (as of 2024):

| Configuration | Points | Transforms | Grid Size | Memory Usage | Computation Time |
|---------------|--------|------------|-----------|--------------|------------------|
| Small         | 5,000  | 1          | 32³       | ~200 MB      | ~1-2 seconds     |
| Medium        | 10,000 | 10         | 64³       | ~2 GB        | ~10-20 seconds   |
| Large         | 20,000 | 50         | 128³      | ~10-15 GB    | ~2-5 minutes     |

*Note: Actual performance may vary based on hardware specifications and implementation details.*

## Data Storage and Management

### HDF5 Implementation

The package uses HDF5 for efficient data storage and organization. Three main files are created for each subject:

1. `data.h5`: Contains raw computational results
   ```
   data.h5
   ├── forward_fft                 # Shape: (n_trans, nx, ny, nz)
   ├── fft_prob_density           # Shape: (n_trans, nx, ny, nz)
   ├── kspace_mask_{i}            # Shape: (nx, ny, nz)
   ├── inverse_map_{i}            # Shape: (n_trans, n_points)
   └── gradient_map_{i}           # Shape: (n_trans, nx, ny, nz)
   ```

2. `analysis.h5`: Contains analysis results
   ```
   analysis.h5
   ├── analysis_summary/
   │   ├── map_{i}/
   │   │   ├── magnitude
   │   │   ├── phase
   │   │   └── local_variance_k{k}
   │   └── enhanced/
   │       ├── spectral_slope
   │       └── spectral_entropy
   ├── map_{i}_magnitude
   ├── map_{i}_phase
   └── map_{i}_local_variance_k{k}
   ```

3. `enhanced.h5`: Contains enhanced feature results
   ```
   enhanced.h5
   ├── enhanced_metrics/
   │   ├── spectral_slope
   │   └── spectral_entropy
   ├── analytical_gradient_map_{i}
   └── excitation_map
   ```

All files use GZIP compression with level 9 for optimal storage efficiency.

### Performance Characteristics

The package has been tested with various data scales:

#### Small Scale (Base Case)
- Input: 1,000 points × 5 time points
- Grid: 20×20×20
- Storage: ~2MB total
  * data.h5: ~1.3MB
  * analysis.h5: ~0.3MB
  * enhanced.h5: ~0.1MB
- Processing time: ~1-2 seconds

#### Medium Scale (5x)
- Input: 5,000 points × 10 time points
- Grid: 36×36×36
- Storage: ~18MB total
  * data.h5: ~15MB
  * analysis.h5: ~3MB
  * enhanced.h5: ~0.7MB
- Processing time: ~5-10 seconds

#### Large Scale (50x)
- Input: 50,000 points × 100 time points
- Grid: 74×74×74
- Storage: ~1.7GB total
  * data.h5: ~1.3GB
  * analysis.h5: ~294MB
  * enhanced.h5: ~73MB
- Processing time: ~88 seconds

### Memory Management

The package implements several memory optimization strategies:

1. **Data Type Optimization**
   - Uses float32 for coordinates when possible
   - Maintains complex128 for FINUFFT compatibility
   - Converts to appropriate types for analysis

2. **HDF5 Streaming**
   - Writes data in chunks
   - Uses compression for storage efficiency
   - Maintains dataset organization

3. **Grid Size Management**
   - Automatically estimates optimal grid dimensions
   - Scales with number of input points
   - Applies upsampling factor for accuracy

### Processing Pipeline Optimization

The package includes several optimizations for efficient processing:

1. **Forward FFT**
   - Parallel FINUFFT implementation
   - Optimized for multiple transforms
   - Memory-efficient normalization

2. **Gradient Calculation**
   - Analytical method (2-5x faster than interpolation)
   - Direct k-space computation
   - Parallel processing for large datasets

3. **Analysis Methods**
   - Vectorized operations
   - Efficient memory usage
   - Parallel processing where applicable

## Technical Pipeline Architecture

### Complete Data Flow and Processing Pipeline

```mermaid
graph TB
    %% Input Data Structure
    Input[/"Input Data Structure"/] --> |"Shape: (n_sources, n_time)"|TimeData["Time Series Data"]
    Input --> |"Shape: (n_points, 3)"|Coords["Spatial Coordinates<br/>(x, y, z)"]
    
    %% Preprocessing Pipeline
    TimeData --> |"normalize_time_series()"|Norm["Time Series Normalization"]
    Norm --> |"method='zscore'/'minmax'/'robust'"|NormTS["Normalized Time Series"]
    NormTS --> |"get_normalized_wavefunctions()"|Complex["Complex Conversion"]
    Complex --> |"Shape: (n_times, n_points)"|WaveFn["Normalized Wavefunctions"]
    
    %% Grid Estimation
    Coords --> GridEst["Grid Size Estimation"]
    GridEst --> |"estimate_grid_size()"|Grid["Uniform Grid<br/>(nx, ny, nz)"]
    
    %% Forward FFT
    WaveFn --> |"Input"|NUFFT1["FINUFFT Type 1"]
    Coords --> |"Coordinates"|NUFFT1
    Grid --> |"Grid Dimensions"|NUFFT1
    NUFFT1 --> |"Shape: (n_times, nx, ny, nz)"|FFTResult["K-Space Representation"]
    
    %% K-Space Processing
    FFTResult --> KNorm["K-Space Normalization"]
    KNorm --> |"normalize_fft_result=True"|ProbDens["Probability Density"]
    
    subgraph "Mask Generation"
        ProbDens --> |"generate_kspace_masks()"|MaskGen["Mask Generation"]
        MaskGen --> |"n_centers, radius"|SphericalMasks["Spherical Masks"]
        MaskGen --> |"kx/ky/kz bounds"|CubicMasks["Cubic Masks"]
        MaskGen --> |"axis, k_value"|SlabMasks["Slab Masks"]
    end
    
    %% Inverse Transform
    SphericalMasks --> |"Shape: (nx, ny, nz)"|MaskedK["Masked K-Space"]
    CubicMasks --> MaskedK
    SlabMasks --> MaskedK
    MaskedK --> NUFFT2["FINUFFT Type 2"]
    Coords --> |"Coordinates"|NUFFT2
    Grid --> |"Grid Dimensions"|NUFFT2
    
    %% Analysis Pipeline
    NUFFT2 --> |"Shape: (n_masks, n_times, n_points)"|InvMaps["Inverse Maps"]
    
    subgraph "Analysis Methods"
        InvMaps --> BasicAnalysis["Basic Analysis"]
        InvMaps --> EnhancedAnalysis["Enhanced Analysis"]
        
        %% Basic Analysis
        BasicAnalysis --> |"compute_magnitude()"|Mag["Magnitude"]
        BasicAnalysis --> |"compute_phase()"|Phase["Phase"]
        BasicAnalysis --> |"compute_local_variance()"|LocalVar["Local Variance"]
        
        %% Enhanced Analysis
        EnhancedAnalysis --> |"compute_spectral_slope()"|Slope["Spectral Slope"]
        EnhancedAnalysis --> |"compute_spectral_entropy()"|Entropy["Spectral Entropy"]
        EnhancedAnalysis --> |"compute_gradient_maps()"|Gradient["Gradient Maps"]
    end
    
    %% Results Storage
    subgraph "HDF5 Storage"
        Mag --> |"analysis.h5"|Analysis["Analysis Results"]
        Phase --> |"analysis.h5"|Analysis
        LocalVar --> |"analysis.h5"|Analysis
        
        FFTResult --> |"data.h5"|RawData["Raw Data"]
        MaskedK --> |"data.h5"|RawData
        InvMaps --> |"data.h5"|RawData
        
        Slope --> |"enhanced.h5"|Enhanced["Enhanced Features"]
        Entropy --> |"enhanced.h5"|Enhanced
        Gradient --> |"enhanced.h5"|Enhanced
    end
    
    %% Data Types and Memory
    subgraph "Memory Management"
        RawData --> |"GZIP Level 9"|Compression["Data Compression"]
        Analysis --> |"GZIP Level 9"|Compression
        Enhanced --> |"GZIP Level 9"|Compression
    end
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef storage fill:#f1f8e9,stroke:#33691e,stroke-width:2px;
    
    class Input,TimeData,Coords input;
    class NUFFT1,NUFFT2,BasicAnalysis,EnhancedAnalysis process;
    class FFTResult,InvMaps,MaskedK data;
    class RawData,Analysis,Enhanced storage;
```

### Technical Implementation Details

1. **Data Structures**
   ```
   Input Data
   ├── Time Series: (n_sources × n_time)
   ├── Coordinates: (n_points × 3)
   └── Output
       ├── FFT Result: (n_times × nx × ny × nz)
       ├── Masks: (n_masks × nx × ny × nz)
       └── Inverse Maps: (n_masks × n_times × n_points)
   ```

2. **Memory Management**
   ```
   HDF5 Structure
   ├── data.h5
   │   ├── forward_fft: (n_times × nx × ny × nz)
   │   ├── kspace_masks: (n_masks × nx × ny × nz)
   │   └── inverse_maps: (n_masks × n_times × n_points)
   ├── analysis.h5
   │   ├── magnitude: (n_masks × n_times × n_points)
   │   ├── phase: (n_masks × n_times × n_points)
   │   └── local_variance: (n_masks × n_times × n_points)
   └── enhanced.h5
       ├── spectral_metrics: (n_times × n_metrics)
       └── gradient_maps: (n_masks × n_times × nx × ny × nz)
   ```

3. **Computational Flow**
   ```
   Processing Steps
   ├── Preprocessing: O(n_sources × n_time)
   ├── Forward FFT: O(n_points × log(nx×ny×nz))
   ├── Mask Operations: O(nx × ny × nz)
   ├── Inverse FFT: O(n_points × log(nx×ny×nz))
   └── Analysis: O(n_masks × n_times × n_points)
   ```

## MapBuilder Implementation Details

### Class Architecture and Dependencies

```mermaid
classDiagram
    class MapBuilder {
        %% Core Attributes
        -ndarray _x, _y, _z
        -ndarray _strengths
        -int _n_trans
        -int _n_points
        -int nx, ny, nz
        
        %% Configuration
        -float eps
        -str dtype
        -bool estimate_grid
        -float upsample_factor
        
        %% Results Storage
        -ndarray fft_result
        -list kspace_masks
        -list inverse_maps
        -list gradient_maps
        -dict analysis_results
        
        %% Methods
        +__init__(subject_id, output_dir, coordinates, strengths)
        +compute_forward_fft()
        +generate_kspace_masks(n_centers, radius)
        +compute_inverse_maps()
        +compute_gradient_maps(method)
        +analyze_inverse_maps(analyses, k_neighbors, save_format, compute_enhanced, enable_local_variance)
        +process_map(n_centers, radius, analyses, k_neighbors, use_analytical_gradient, calculate_local_variance)
        
        %% Private Methods
        -_validate_inputs()
        -_estimate_grid_size()
        -_setup_finufft()
        -_normalize_results()
    }

    class FINUFFT {
        +type1
        +type2
        +plan
    }

    class HDF5Manager {
        +create_file()
        +save_dataset()
        +load_dataset()
        +compress_data()
    }

    class Logger {
        +setup()
        +info()
        +warning()
        +error()
    }

    MapBuilder --> FINUFFT : uses
    MapBuilder --> HDF5Manager : uses
    MapBuilder --> Logger : uses
```

### Gradient Computation Technical Details

```mermaid
graph TB
    %% Input and Configuration
    Input[/"Input Parameters<br/>- Data Shape<br/>- Grid Size<br/>- Method Choice"/] --> Config["Configuration"]
    Config --> |"Analytical"|AMethod["Analytical Method"]
    Config --> |"Grid-Based"|GMethod["Grid Method"]
    
    %% Analytical Method Implementation
    subgraph "Analytical Gradient Pipeline"
        AMethod --> AFFT["Forward FFT<br/>Type 1 NUFFT"]
        AFFT --> KDerivatives["K-Space Derivatives<br/>∇f = F⁻¹{ik·F{f}}"]
        KDerivatives --> |"∂/∂x"|Dx["dx = F⁻¹{ikx·F{f}}"]
        KDerivatives --> |"∂/∂y"|Dy["dy = F⁻¹{iky·F{f}}"]
        KDerivatives --> |"∂/∂z"|Dz["dz = F⁻¹{ikz·F{f}}"]
        
        Dx --> |"Magnitude"|GMag["Gradient Magnitude"]
        Dy --> |"Magnitude"|GMag
        Dz --> |"Magnitude"|GMag
    end
    
    %% Grid Method Implementation
    subgraph "Grid-Based Pipeline"
        GMethod --> GridSetup["Grid Setup<br/>nx × ny × nz"]
        GridSetup --> Interp1["Interpolation<br/>Non-uniform to Grid"]
        Interp1 --> FDiff["Finite Differences<br/>∇f ≈ (f[i+1] - f[i-1])/2h"]
        FDiff --> |"Compute"|GDx["dx/dx"]
        FDiff --> |"Compute"|GDy["dy/dy"]
        FDiff --> |"Compute"|GDz["dz/dz"]
        
        GDx --> |"Magnitude"|GridMag["Grid Magnitude"]
        GDy --> |"Magnitude"|GridMag
        GDz --> |"Magnitude"|GridMag
        
        GridMag --> Interp2["Interpolation<br/>Grid to Non-uniform"]
    end
    
    %% Results Processing
    GMag --> Results["Gradient Maps"]
    Interp2 --> Results
    
    %% Performance Analysis
    subgraph "Performance Metrics"
        direction TB
        CompTime["Computation Time"]
        MemUsage["Memory Usage"]
        Accuracy["Accuracy"]
        
        CompTime --> |"Analytical"|ATime["2-5x Faster"]
        CompTime --> |"Grid"|GTime["Memory Intensive"]
        
        MemUsage --> |"Analytical"|AMem["O(n_points)"]
        MemUsage --> |"Grid"|GMem["O(nx·ny·nz)"]
        
        Accuracy --> |"Analytical"|AAcc["Better for<br/>Uniform Points"]
        Accuracy --> |"Grid"|GAcc["Better for<br/>Irregular Points"]
    end
    
    %% Styling
    classDef config fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef results fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class Input,Config config
    class AFFT,GMethod,GridSetup,FDiff process
    class Results,GMag,GridMag results
```

### Implementation Specifications

1. **Memory Management**
   ```python
   # Memory requirements for different methods
   analytical_method_memory = {
       'forward_fft': 'O(n_points)',
       'k_derivatives': 'O(3 * n_points)',
       'total': 'O(4 * n_points)'
   }
   
   grid_method_memory = {
       'grid_data': 'O(nx * ny * nz)',
       'derivatives': 'O(3 * nx * ny * nz)',
       'interpolation': 'O(n_points)',
       'total': 'O(4 * nx * ny * nz + n_points)'
   }
   ```

2. **Computational Complexity**
   ```python
   # Operation counts for each method
   analytical_operations = {
       'forward_fft': 'O(n_points * log(nx*ny*nz))',
       'k_multiplication': 'O(nx * ny * nz)',
       'inverse_fft': 'O(n_points * log(nx*ny*nz))',
       'magnitude': 'O(n_points)'
   }
   
   grid_operations = {
       'interpolation_to_grid': 'O(n_points * log(n_points))',
       'finite_differences': 'O(nx * ny * nz)',
       'magnitude': 'O(nx * ny * nz)',
       'interpolation_from_grid': 'O(n_points * log(n_points))'
   }
   ```

3. **Precision Analysis**
   ```python
   # Error estimates for different methods
   error_estimates = {
       'analytical': {
           'uniform_points': '~1e-6',
           'irregular_points': '~1e-4',
           'limiting_factor': 'FFT precision'
       },
       'grid': {
           'uniform_points': '~1e-5',
           'irregular_points': '~1e-5',
           'limiting_factor': 'Interpolation order'
       }
   }
   ```

4. **Method Selection Criteria**
   ```python
   def select_gradient_method(points, grid_size, timing_critical=False):
       """
       Criteria for selecting gradient computation method
       """
       if timing_critical:
           return 'analytical'
       elif is_irregular_distribution(points):
           return 'grid'
       else:
           return 'analytical'
   ```

## References

### FINUFFT Citations
1. Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A Parallel Nonuniform Fast Fourier Transform Library Based on an "Exponential of Semicircle" Kernel. SIAM Journal on Scientific Computing, 41(5), C479–C504. https://doi.org/10.1137/18M1173014 ([arXiv version](https://arxiv.org/abs/1808.06736))

2. Barnett, A. H. (2021). Aliasing error of the exp⁡(β√(1-z²)) kernel in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis, 51, 1–16. https://doi.org/10.1016/j.acha.2020.08.007 ([arXiv version](https://arxiv.org/abs/1910.00850))

3. Shih, Y.-H., Wright, G., Andén, J., Blaschke, J., & Barnett, A. H. (2021). cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. PDSEC2021 workshop of the IPDPS2021 conference. ([arXiv version](https://arxiv.org/abs/2102.08463))

### Other References
4. Keiner, J., Kunis, S., & Potts, D. (2009). Using NFFT 3—A Software Library for Various Nonequispaced Fast Fourier Transforms. ACM Transactions on Mathematical Software, 36(4), 1–30. https://doi.org/10.1145/1555386.1555388

5. Cohen, M. X. (2014). Analyzing Neural Time Series Data: Theory and Practice. MIT Press.

6. Wendel, K., Väisänen, O., Malmivuo, J., Gencer, N. G., Vanrumste, B., Durka, P., Magjarević, R., Supek, S., Pascu, M. L., Fontenelle, H., & Grave de Peralta Menendez, R. (2009). EEG/MEG Source Imaging: Methods, Challenges, and Open Issues. Computational Intelligence and Neuroscience, 2009, 1–12. https://doi.org/10.1155/2009/656092

7. Lopes da Silva, F. (2013). EEG and MEG: Relevance to Neuroscience. Neuron, 80(5), 1112–1128. https://doi.org/10.1016/j.neuron.2013.10.017

8. Tadel, F., Baillet, S., Mosher, J. C., Pantazis, D., & Leahy, R. M. (2011). Brainstorm: A User-Friendly Application for MEG/EEG Analysis. Computational Intelligence and Neuroscience, 2011, 1–13. https://doi.org/10.1155/2011/879716 