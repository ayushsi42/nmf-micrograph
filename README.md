# Constrained NMF for Microstructure Segmentation

Implementation of constrained non-negative matrix factorization (NMF) for segmenting metallographic microstructure images.

## Project Structure

```
crystallography-project/
├── microstructure_segmentation.py  # Main unified implementation
├── app.py                           # Web UI (Gradio)
├── load_dataset.py                  # Dataset loader
├── utils/                           # Utility modules
│   ├── preprocessing.py            # Image preprocessing & feature extraction
│   ├── visualization.py            # Visualization utilities
│   └── advanced_models.py          # Spatial & Orthogonal NMF variants
├── images/                          # Input images
└── segmentation_results/            # Output segmentations
```

## Files

- **`microstructure_segmentation.py`**: 🎯 **Main file** - Complete NMF segmentation pipeline
  - `MicrostructureSegmenter` class with all functionality
  - Support for standard, spatial, and orthogonal NMF
  - Batch processing and model comparison tools
  - Convenience functions for quick usage
  
- **`app.py`**: 🌐 **Web UI** - Interactive Gradio interface
  
- **`utils/`**: Utility modules
  - `preprocessing.py`: Image loading and feature extraction
  - `visualization.py`: Segmentation visualization tools
  - `advanced_models.py`: Advanced NMF variants (spatial smoothness, orthogonality)
  
- **`load_dataset.py`**: Dataset loader for OD_MetalDAM

## Dataset

Uses the **OD_MetalDAM** dataset (Voxel51/Hugging Face):
- 42 SEM micrographs of metal microstructures
- 5 metallurgical phases: Matrix, Austenite, Martensite/Austenite, Precipitate, Defects
- High-resolution images (1024×703 to 1280×895 pixels)

## Requirements

```bash
pip install numpy scipy scikit-learn opencv-python matplotlib scikit-image pillow datasets gradio
```

## Quick Start

**1. Test the setup:**
```bash
python test_setup.py
```

**2. Run the Web UI:**
```bash
pip install gradio  # if not already installed
python app.py
```
Access at http://localhost:7861 or http://YOUR_LOCAL_IP:7861

**3. Process dataset from command line:**
```bash
python microstructure_segmentation.py
```

## Usage

### 🌐 Web UI (Recommended)

**1. Install Gradio:**
```bash
pip install gradio
# or run: bash install_ui.sh
```

**2. Launch the web interface:**
```bash
python app.py
```

**3. Access the UI:**
- **Local:** http://localhost:7861
- **Network:** http://YOUR_LOCAL_IP:7861 (accessible from other devices on your WiFi)
- The app will automatically open in your browser

**Features:**
- Upload any micrograph image
- Adjust number of components (phases)
- Control sparsity for distinct phase separation
- View segmentation and component activation maps with automatic phase naming
- Color bar legend showing activation scale (blue = not present, yellow = present)
- No coding required!

### 1. Load Dataset
```bash
python load_dataset.py
```
Images saved to `images/` directory.

### 2. Basic NMF Segmentation
```python
from nmf_segmentation import segment_microstructure_dataset

segment_microstructure_dataset(
    image_dir="images",
    output_dir="segmentation_results",
    n_components=5,       # Number of phases
    sparsity=0.1,         # Sparsity constraint
    sample_size=10,       # Training images
    target_size=(512, 512)
)
```

```

## Usage

### 🌐 Web UI (Recommended)

**1. Install Gradio:**
```bash
pip install gradio
# or run: bash install_ui.sh
```

**2. Launch the web interface:**
```bash
python app.py
```

**3. Access the UI:**
- **Local:** http://localhost:7861
- **Network:** http://YOUR_LOCAL_IP:7861 (accessible from other devices on your WiFi)
- The app will automatically open in your browser

**Features:**
- Upload any micrograph image
- Adjust number of components (phases)
- Control sparsity for distinct phase separation
- View segmentation and component activation maps with automatic phase naming
- Color bar legend showing activation scale (blue = not present, yellow = present)
- No coding required!

### 1. Load Dataset
```bash
python load_dataset.py
```
Images saved to `images/` directory.

### 2. Basic NMF Segmentation (Command Line)
```python
from microstructure_segmentation import batch_process_dataset

# Process entire dataset
batch_process_dataset(
    image_dir="images",
    output_dir="segmentation_results",
    n_components=5,
    sparsity=0.1,
    training_samples=10,
    model_type='standard'  # or 'spatial', 'orthogonal'
)
```

**Quick single image segmentation:**
```python
from microstructure_segmentation import quick_segment

image, segmentation, components = quick_segment(
    image_path="images/image_0.jpg",
    n_components=5,
    sparsity=0.1,
    output_path="result.png"
)
```

### 3. Advanced Usage
```python
from microstructure_segmentation import MicrostructureSegmenter

# Initialize with spatial smoothness
segmenter = MicrostructureSegmenter(
    n_components=5,
    sparsity=0.1,
    model_type='spatial',
    smoothness=0.5
)

# Train on multiple images
segmenter.train(
    image_paths=["images/img1.jpg", "images/img2.jpg"],
    target_size=(512, 512)
)

# Segment new image
image, segmentation, components = segmenter.segment("images/test.jpg")

# Batch process
segmenter.batch_segment("images", "output")
```

### 4. Compare Models
```python
from microstructure_segmentation import MicrostructureSegmenter

segmenter = MicrostructureSegmenter(n_components=5)

# Define model configurations to compare
configs = {
    'Standard NMF': {'n_components': 5, 'sparsity': 0.0},
    'Sparse NMF': {'n_components': 5, 'sparsity': 0.5},
    'Spatial NMF': {'n_components': 5, 'sparsity': 0.1, 'model_type': 'spatial', 'smoothness': 0.5},
    'Orthogonal NMF': {'n_components': 5, 'model_type': 'orthogonal'}
}

segmenter.compare_models("images/image_0.jpg", configs)
# Saves comparison plot as model_comparison.png
```

## Method Overview

### Constrained NMF Formulation

Decompose feature matrix **X** ≈ **WH** where:
- **X**: (n_pixels × n_features) – image features
- **W**: (n_pixels × n_components) – spatial activation maps
- **H**: (n_components × n_features) – component signatures

**Objective with constraints:**
```
minimize: ||X - WH||² + α(||W||₁ + ||H||₁) + β·Smoothness(W)
subject to: W ≥ 0, H ≥ 0
```

### Features Extracted
1. **Intensity**: Raw pixel values
2. **Gradients**: Sobel (x, y), magnitude
3. **Edges**: Laplacian
4. **Texture**: Local variance
5. **Multi-scale**: Gaussian blur (σ=1, σ=2)

### Constraint Types

#### 1. Sparsity (L1)
- Encourages few active components per pixel
- Improves interpretability
- Parameter: `sparsity_constraint` or `sparsity_w/h`

#### 2. Spatial Smoothness
- Penalizes discontinuities between neighboring pixels
- Uses Gaussian smoothing + Laplacian
- Parameter: `smoothness`

#### 3. Orthogonality
- Forces distinct, non-overlapping components
- Useful when phases are mutually exclusive
- Parameter: `orthogonality_penalty`

## Example Results

After running, you'll get:
- **Segmentation maps**: Color-coded phase assignments
- **Component activation maps**: Individual phase distributions
- **Quantitative metrics**: Reconstruction error, convergence

## Understanding the Results

### What Do the Colors Represent?

The segmentation produces three types of visualizations:

#### 1. **Segmentation Map** (Right side)
- **What it shows**: Hard assignment - each pixel belongs to exactly one microstructural phase
- **Colors**: Each color represents a different phase/component
- **Example**: Red pixels = Phase 1, Blue pixels = Phase 2, etc.

#### 2. **Component Activation Maps** (Bottom row)
- **What it shows**: Soft assignment - how strongly each phase is present at each pixel
- **Colors**: Heatmap from blue (low activation) to yellow (high activation)
- **Why they look similar**: Different components can be active in different regions, but the spatial patterns may overlap
- **Interpretation**: 
  - Bright areas = high probability of that phase
  - Dark areas = low probability of that phase
  - Components are NOT mutually exclusive - a pixel can have high activation for multiple phases

#### 3. **Phase Names**
- Automatically assigned based on activation levels:
  - **Defects/Voids**: Lowest activation areas (dark regions)
  - **Precipitates**: Small, distinct features
  - **Martensite**: Needle-like structures
  - **Austenite**: Medium-contrast regions
  - **Matrix**: Highest activation areas (bright background)

### Example Interpretation
If you see:
- **Component 1** (Matrix): Bright in large, uniform areas
- **Component 2** (Austenite): Bright in medium-contrast regions
- **Component 3** (Defects): Bright in very dark spots

This means the algorithm found three distinct microstructural patterns in your image!


## Tuning Guide

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `n_components` | Number of phases | 3-7 |
| `sparsity_constraint` | More sparse activations | 0.0-1.0 |
| `smoothness` | Smoother regions | 0.0-2.0 |
| `max_iter` | Convergence quality | 200-1000 |
| `sample_size` | Training data | 5-20 images |

**Guidelines:**
- High sparsity (>0.5): Sharp, distinct phases
- High smoothness (>1.0): Remove noise, coherent regions
- More components: Capture fine phase distinctions
- Fewer components: Coarser segmentation


## License

Code: MIT  
Dataset: MIT (images), Apache-2.0 (code)
