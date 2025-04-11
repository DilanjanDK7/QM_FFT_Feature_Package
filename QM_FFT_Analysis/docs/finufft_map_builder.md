# FINUFFT Map Builder Documentation

## Overview
The `MapBuilder` class provides a comprehensive workflow for analyzing scattered 3D data using FINUFFT (Fast Non-uniform Fast Fourier Transform). It transforms data between spatial and frequency domains, generates masks for specific k-space regions, and computes inverse maps and gradient maps.

## Technical Details
The `MapBuilder` class in `QM_FFT_Analysis/utils/map_builder.py` is designed to process non-uniformly spaced 3D data points. Its main purpose is to:
1. Transform non-uniform coordinates (`x, y, z`) with corresponding `strengths` into a uniform grid in k-space
2. Analyze specific regions of this k-space
3. Transform those regions back to the original space
4. Compute gradients of the transformed data

### Detailed Component Analysis

#### 1. Initialization (`__init__`)
- **Input Parameters**:
  - Subject ID and output directory
  - Non-uniform coordinates (`x`, `y`, `z`)
  - Corresponding data values (`strengths`)
  - Optional parameters:
    - `padding` (default: 0)
    - `stride` (default: 1)
    - FINUFFT precision `eps` (default: 1e-6)
    - Complex data type `dtype` (default: 'complex128')

- **Validation and Setup**:
  - Validates `subject_id`, `output_dir`, `eps`, and `dtype`
  - Creates structured output directories:
    - `plots/` for visualizations
    - `data/` for saved arrays
    - `logs/` for logging information

- **Data Processing**:
  - Converts coordinates to `float64` NumPy arrays
  - Validates shape consistency across all inputs
  - Converts `strengths` to specified complex `dtype`
  - Determines uniform grid dimensions from unique coordinate values
  - Initializes k-space frequency coordinates using `np.fft.fftfreq`

#### 2. FINUFFT Plan Initialization (`_initialize_plans`)
- **Purpose**: Pre-calculate transformation plans for optimal performance
- **Data Type Management**:
  - Determines real data type based on complex `dtype`:
    - `complex64` → `float32`
    - `complex128` → `float64`
  - Converts coordinates to appropriate real type
- **Plan Creation**:
  - Forward Plan (Type 1):
    - Transforms non-uniform points to uniform grid
    - Configures grid dimensions, precision, and data type
    - Sets up non-uniform input points
  - Inverse Plan (Type 2):
    - Transforms uniform grid back to non-uniform points
    - Uses same configuration as forward plan

#### 3. Forward FFT (`compute_forward_fft`)
- **Process**:
  1. Flattens `strengths` array for Type 1 plan
  2. Executes forward transform
  3. Reshapes result to 3D grid
  4. Stores in `self.fft_result` and `self.forward_fft`
  5. Saves to `forward_fft.npy`

#### 4. K-Space Mask Generation (`generate_kspace_masks`)
- **Implementation**:
  - Generates random center points in k-space [-1, 1]
  - Creates spherical masks around each center
  - Stores masks in `self.kspace_masks`
  - Saves individual masks to `kspace_mask_{i}.npy`

#### 5. Inverse Maps (`compute_inverse_maps`)
- **Steps**:
  1. Applies masks to k-space data
  2. Executes inverse transform
  3. Reshapes result to match input coordinates
  4. Stores in `self.inverse_maps`
  5. Saves to `inverse_map_{i}.npy`
  6. Generates visualization plots

#### 6. Gradient Maps (`compute_gradient_maps`)
- **Computation**:
  1. Calculates spatial gradients using `np.gradient`
  2. Computes Euclidean magnitude
  3. Stores in `self.gradient_maps`
  4. Saves to `gradient_map_{i}.npy`
  5. Generates visualization plots

#### 7. Volume Plotting (`generate_volume_plot`)
- Creates interactive 3D visualizations using Plotly
- Saves plots as HTML files in the `plots` directory

#### 8. Processing Pipeline (`process_map`)
- **Sequence**:
  1. Forward FFT computation
  2. K-space mask generation
  3. Inverse map computation
  4. Gradient map computation

## Key Components

### 1. Initialization (`__init__`)
- **Inputs**:
  - `x, y, z`: 3D coordinates of data points
  - `strengths`: Values at each point
  - `grid_size`: Size of the uniform grid (default: 32)
  - `eps`: FINUFFT tolerance (default: 1e-6)
  - `dtype`: Data type for computations (default: 'complex64')
  - `subject_id`: Identifier for output files (default: 'test')
  - `output_dir`: Directory for saving results (default: 'data/output')

- **Setup**:
  - Validates input shapes and data types
  - Creates output directories
  - Initializes k-space grid
  - Sets up FINUFFT plans

### 2. FINUFFT Plan Initialization (`_initialize_plans`)
- Pre-calculates transformation plans for efficiency
- Handles data type conversion:
  - Uses `float32` for real coordinates with `complex64`
  - Uses `float64` for real coordinates with `complex128`
- Flattens coordinate arrays for FINUFFT
- Creates forward and inverse plans

### 3. Forward FFT (`compute_forward_fft`)
- Transforms non-uniform strengths data into uniform k-space grid
- Process:
  1. Prepares input data (reshapes and converts to complex)
  2. Executes forward transform
  3. Reshapes result to 3D grid
  4. Saves result to file
  5. Generates visualization

### 4. K-Space Mask Generation (`generate_kspace_masks`)
- Creates boolean masks for specific regions in k-space
- Process:
  1. Generates random points in k-space
  2. Creates masks for each region
  3. Stores masks for later use
  4. Saves masks to files

### 5. Inverse Maps (`compute_inverse_maps`)
- Transforms selected k-space regions back to original coordinate space
- Process:
  1. Prepares input data (reshapes and converts to complex)
  2. Applies k-space mask
  3. Executes inverse transform
  4. Reshapes result to 3D grid
  5. Saves result to file
  6. Generates visualization

### 6. Gradient Maps (`compute_gradient_maps`)
- Calculates spatial gradient magnitude for each inverse map
- Process:
  1. Computes gradients using numpy's gradient function
  2. Calculates magnitude of gradient vectors
  3. Stores gradient maps
  4. Saves maps to files
  5. Generates visualizations

### 7. Volume Plotting (`generate_volume_plot`)
- Generates interactive 3D volume visualizations using Plotly
- Features:
  - Slice views in all three dimensions
  - Interactive controls
  - Color mapping
  - Axis labels and titles

### 8. Processing Pipeline (`process_map`)
- Orchestrates the main steps in sequence:
  1. Computes forward FFT
  2. Generates k-space masks
  3. Computes inverse maps
  4. Computes gradient maps

## Usage Example
```python
# Initialize MapBuilder
builder = MapBuilder(x, y, z, strengths, grid_size=32)

# Process the map
builder.process_map()

# Access results
forward_fft = builder.forward_fft
inverse_maps = builder.inverse_maps
gradient_maps = builder.gradient_maps
```

## Output Files
The class generates several types of output files:
1. Forward FFT data (`.npy`)
2. K-space masks (`.npy`)
3. Inverse maps (`.npy`)
4. Gradient maps (`.npy`)
5. Visualizations (`.html`)

All files are saved in the specified output directory with appropriate subdirectories for organization.

## Notes
- FINUFFT is used for efficient non-uniform FFT computations
- The class supports both `complex64` and `complex128` data types
- All visualizations are interactive HTML files using Plotly
- The class includes comprehensive error handling and input validation 