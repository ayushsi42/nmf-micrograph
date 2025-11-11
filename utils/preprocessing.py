"""
Image preprocessing utilities for microstructure analysis
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def load_and_preprocess_image(
    image_path: str, 
    target_size: Optional[Tuple[int, int]] = None,
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Load and preprocess a microstructure image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    target_size : tuple, optional
        Resize image to (height, width)
    apply_clahe : bool
        Apply CLAHE contrast enhancement
        
    Returns:
    --------
    image : np.ndarray
        Preprocessed grayscale image (normalized to [0, 1])
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize if specified
    if target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Optional: Apply CLAHE for contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_uint8 = (image * 255).astype(np.uint8)
        enhanced = clahe.apply(image_uint8)
        image = enhanced.astype(np.float32) / 255.0
    
    return image


def extract_features(image: np.ndarray, use_texture: bool = True) -> np.ndarray:
    """
    Extract features from the image for NMF decomposition.
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image (normalized)
    use_texture : bool
        Whether to include texture features
        
    Returns:
    --------
    features : np.ndarray
        Feature matrix of shape (n_pixels, n_features)
    """
    h, w = image.shape
    features_list = []
    
    # Convert to float64 for OpenCV operations
    image_64 = image.astype(np.float64)
    
    # Original intensity
    features_list.append(image.flatten())
    
    if use_texture:
        # Gradient magnitude
        grad_x = cv2.Sobel(image_64, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_64, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features_list.append(grad_mag.flatten())
        
        # Laplacian (edge detection)
        laplacian = cv2.Laplacian(image_64, cv2.CV_64F)
        features_list.append(np.abs(laplacian).flatten())
        
        # Local variance (texture)
        kernel_size = 5
        mean_img = cv2.blur(image_64, (kernel_size, kernel_size))
        mean_sq = cv2.blur(image_64**2, (kernel_size, kernel_size))
        variance = mean_sq - mean_img**2
        features_list.append(variance.flatten())
        
        # Gaussian blur at different scales
        blur1 = cv2.GaussianBlur(image_64, (3, 3), 0)
        blur2 = cv2.GaussianBlur(image_64, (7, 7), 0)
        features_list.append(blur1.flatten())
        features_list.append(blur2.flatten())
    
    # Stack features
    features = np.vstack(features_list).T  # Shape: (n_pixels, n_features)
    
    # Normalize each feature to [0, 1] independently
    # This is standard for NMF input (Lee & Seung 1999)
    f_min = features.min(axis=0, keepdims=True)
    f_max = features.max(axis=0, keepdims=True)
    f_range = f_max - f_min
    
    # Avoid division by zero
    f_range = np.where(f_range < 1e-10, 1.0, f_range)
    features = (features - f_min) / f_range
    
    # Ensure all values are strictly positive for NMF
    features = np.maximum(features, 1e-10)
    
    return features
