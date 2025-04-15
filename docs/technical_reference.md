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

## Gradient Analysis and Interpretation

After inverse transformation from masked k-space to source space, the package computes spatial gradient magnitudes to identify regions of rapid spatial change.

### Gradient Calculation

```python
def compute_gradient_maps(self):
    """Compute gradient maps by interpolating inverse maps onto a grid."""
```

The gradient calculation involves:
1. Interpolation of the inverse-transformed complex data onto a uniform grid
2. Computation of spatial derivatives along each dimension
3. Calculation of the gradient magnitude as the Euclidean norm of the derivatives

Mathematical representation:
```
∇f(x,y,z) = (∂f/∂x, ∂f/∂y, ∂f/∂z)
Gradient Magnitude = |∇f| = √[(∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²]
```

### Neurophysiological Interpretation

The gradient magnitude maps can be interpreted in the context of neurophysiology:

1. **Functional Boundaries**: High gradient values may indicate boundaries between distinct functional regions
2. **Excitability Markers**: Steep gradients might correspond to regions of heightened neural excitability
3. **Activity Transitions**: Gradients could represent transition zones between different activity states
4. **Pathological Indicators**: Abnormal gradient patterns might indicate pathological conditions

### Validation Approaches

To validate the neurophysiological relevance of gradient maps, several approaches can be considered:

1. **Correlation with Established Measures**:
   - Compare with TMS motor evoked potentials (MEPs)
   - Correlate with fMRI BOLD response patterns
   - Validate against electrocorticography (ECoG) measurements

2. **Test-Retest Reliability**:
   - Assess consistency of gradient patterns across repeated measures
   - Evaluate stability over different time scales

3. **Sensitivity to Known Interventions**:
   - Brain stimulation effects (TMS, tDCS)
   - Pharmacological manipulations
   - Task-related changes

4. **Clinical Correlations**:
   - Comparison between patient groups and healthy controls
   - Correlation with clinical outcome measures
   - Longitudinal tracking of disease progression

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

## References

### FINUFFT Citations
1. Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A Parallel Nonuniform Fast Fourier Transform Library Based on an "Exponential of Semicircle" Kernel. SIAM Journal on Scientific Computing, 41(5), C479–C504. https://doi.org/10.1137/18M1173014 ([arXiv version](https://arxiv.org/abs/1808.06736))

2. Barnett, A. H. (2021). Aliasing error of the exp⁡(β√(1-z²)) kernel in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis, 51, 1-16. https://doi.org/10.1016/j.acha.2020.08.007 ([arXiv version](https://arxiv.org/abs/1910.00850))

3. Shih, Y.-H., Wright, G., Andén, J., Blaschke, J., & Barnett, A. H. (2021). cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. PDSEC2021 workshop of the IPDPS2021 conference. ([arXiv version](https://arxiv.org/abs/2102.08463))

### Other References
4. Keiner, J., Kunis, S., & Potts, D. (2009). Using NFFT 3—A Software Library for Various Nonequispaced Fast Fourier Transforms. ACM Transactions on Mathematical Software, 36(4), 1–30. https://doi.org/10.1145/1555386.1555388

5. Cohen, M. X. (2014). Analyzing Neural Time Series Data: Theory and Practice. MIT Press.

6. Wendel, K., Väisänen, O., Malmivuo, J., Gencer, N. G., Vanrumste, B., Durka, P., Magjarević, R., Supek, S., Pascu, M. L., Fontenelle, H., & Grave de Peralta Menendez, R. (2009). EEG/MEG Source Imaging: Methods, Challenges, and Open Issues. Computational Intelligence and Neuroscience, 2009, 1–12. https://doi.org/10.1155/2009/656092

7. Lopes da Silva, F. (2013). EEG and MEG: Relevance to Neuroscience. Neuron, 80(5), 1112–1128. https://doi.org/10.1016/j.neuron.2013.10.017

8. Tadel, F., Baillet, S., Mosher, J. C., Pantazis, D., & Leahy, R. M. (2011). Brainstorm: A User-Friendly Application for MEG/EEG Analysis. Computational Intelligence and Neuroscience, 2011, 1–13. https://doi.org/10.1155/2011/879716 