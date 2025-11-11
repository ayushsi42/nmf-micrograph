"""
Visualization utilities for segmentation results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import cv2


def colorize_segmentation(segmentation: np.ndarray, n_components: int) -> np.ndarray:
    """
    Apply colormap to segmentation map.
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Segmentation map with integer labels
    n_components : int
        Number of components
        
    Returns:
    --------
    colored : np.ndarray
        RGB colored segmentation
    """
    # Use tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_components]
    
    h, w = segmentation.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(n_components):
        mask = segmentation == i
        colored[mask] = (colors[i, :3] * 255).astype(np.uint8)
    
    return colored


def create_component_grid(component_maps: List[np.ndarray]) -> np.ndarray:
    """
    Create a grid visualization of component activation maps.
    
    Parameters:
    -----------
    component_maps : list of np.ndarray
        List of component activation maps
        
    Returns:
    --------
    grid_image : np.ndarray
        Grid image as RGB array
    """
    n_components = len(component_maps)
    
    # Create grid layout
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_components == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, comp_map in enumerate(component_maps):
        axes[i].imshow(comp_map, cmap='viridis')
        axes[i].set_title(f'Component {i}', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert to image (compatible with both old and new matplotlib)
    fig.canvas.draw()
    try:
        # New matplotlib (>= 3.8)
        buf = fig.canvas.buffer_rgba()
        grid_image = np.asarray(buf)
        grid_image = grid_image[:, :, :3]  # Remove alpha channel
    except AttributeError:
        try:
            # Matplotlib 3.x
            grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback for older matplotlib
            grid_image = np.frombuffer(fig.canvas.tobytes(), dtype=np.uint8)
            grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return grid_image


def visualize_segmentation(
    original_image: np.ndarray,
    segmentation: np.ndarray,
    component_maps: Optional[List[np.ndarray]] = None,
    n_components: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize segmentation results.
    
    Parameters:
    -----------
    original_image : np.ndarray
        Original grayscale image
    segmentation : np.ndarray
        Segmentation map
    component_maps : list of np.ndarray, optional
        Individual component activation maps
    n_components : int
        Number of components
    save_path : str, optional
        Path to save the visualization
    """
    # Use actual length of component_maps if provided
    actual_n_components = len(component_maps) if component_maps is not None else n_components
    n_plots = 2 if component_maps is None else 2 + actual_n_components
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(segmentation, cmap='tab10')
    axes[1].set_title(f'NMF Segmentation ({actual_n_components} components)')
    axes[1].axis('off')
    
    # Component maps
    if component_maps is not None:
        for i, comp_map in enumerate(component_maps):
            axes[2 + i].imshow(comp_map, cmap='viridis')
            axes[2 + i].set_title(f'Component {i}')
            axes[2 + i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
