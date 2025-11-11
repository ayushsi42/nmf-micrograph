"""
Configuration file for Microstructure Segmentation
Based on literature best practices for NMF-based image segmentation
"""

# NMF Model Parameters (based on Lee & Seung 1999, Hoyer 2004)
NMF_CONFIG = {
    'n_components': 3,           # Number of phases to detect (2-5 typical for metallography)
    'init': 'nndsvda',           # NNDSVDA initialization (deterministic, literature standard)
    'solver': 'cd',              # Coordinate Descent (faster convergence than MU)
    'beta_loss': 'frobenius',    # Frobenius norm (standard for image data)
    'max_iter': 300,             # Sufficient for convergence (increased for stability)
    'tol': 1e-4,                 # Convergence tolerance
    'random_state': 42,          # Reproducibility
    'alpha_W': 0.01,             # Small L1 regularization on W for stability
    'alpha_H': 0.01,             # Small L1 regularization on H for stability
    'l1_ratio': 0.5,             # Balance between L1 and L2 (0.5 = equal mix)
}

# Feature Extraction Parameters (based on microstructure analysis literature)
FEATURE_CONFIG = {
    'use_intensity': True,           # Raw pixel intensity
    'use_gradients': True,           # Sobel gradients for edges/boundaries
    'use_laplacian': True,           # Second-order edges
    'use_texture': True,             # Local variance for texture
    'use_multiscale': True,          # Multi-scale Gaussian features
    
    # Filter parameters
    'sobel_ksize': 3,                # Sobel kernel size (3 or 5)
    'laplacian_ksize': 3,            # Laplacian kernel size
    'variance_window': 5,            # Local variance window (5 or 7)
    'gaussian_sigmas': [0.5, 1.5],   # Multi-scale Gaussian sigmas
    
    # Preprocessing
    'apply_clahe': True,             # CLAHE contrast enhancement
    'clahe_clip_limit': 2.0,         # CLAHE clip limit (1.0-4.0)
    'clahe_grid_size': (8, 8),       # CLAHE tile grid size
}

# Image Processing Parameters
IMAGE_CONFIG = {
    'target_size': 512,              # Resize images to this size (power of 2 recommended)
    'normalize_method': 'minmax',    # 'minmax' or 'zscore'
    'epsilon': 1e-10,                # Small constant for numerical stability
}

# Visualization Parameters
VIZ_CONFIG = {
    'colormap': 'viridis',           # Colormap for component visualization
    'segmentation_colors': 'tab10',  # Color scheme for segmentation map
    'dpi': 100,                      # Figure DPI
    'save_format': 'png',            # Output format
}

# Training Parameters
TRAINING_CONFIG = {
    'n_training_images': 10,         # Number of images for training
    'batch_mode': False,             # Process images individually or batched
    'verbose': True,                 # Print progress
}

# Advanced NMF Variants
ADVANCED_CONFIG = {
    'spatial_smoothness': 0.1,       # Spatial constraint weight (0.0-1.0)
    'orthogonality_penalty': 0.1,    # Orthogonality constraint (0.0-1.0)
}

# Preset configurations for different use cases
PRESETS = {
    'default': {
        'n_components': 3,
        'alpha_W': 0.01,
        'alpha_H': 0.01,
        'solver': 'cd',
        'max_iter': 300,
    },
    'high_sparsity': {
        'n_components': 3,
        'alpha_W': 0.15,
        'alpha_H': 0.15,
        'solver': 'mu',
        'max_iter': 400,
    },
    'fine_detail': {
        'n_components': 4,
        'alpha_W': 0.05,
        'alpha_H': 0.05,
        'solver': 'cd',
        'max_iter': 350,
    },
}
