"""
Microstructure Segmentation using Constrained Non-Negative Matrix Factorization

A comprehensive implementation of NMF-based segmentation for metallographic images
with multiple constraint types and analysis tools.

Based on:
- Lee & Seung (1999) "Learning the parts of objects by non-negative matrix factorization"
- Hoyer (2004) "Non-negative Matrix Factorization with Sparseness Constraints"

Author: Ayush Singh
Date: November 2025
"""

import numpy as np
import cv2
from sklearn.decomposition import NMF
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import utilities and config
from utils.preprocessing import load_and_preprocess_image, extract_features
from utils.visualization import visualize_segmentation, colorize_segmentation, create_component_grid
from utils.advanced_models import SpatiallyConstrainedNMF, OrthogonalNMF
from config import NMF_CONFIG, PRESETS


class MicrostructureSegmenter:
    """
    Main class for microstructure segmentation using Constrained NMF.
    
    Combines feature extraction, NMF decomposition with constraints, and
    segmentation analysis into a unified interface.
    """
    
    def __init__(
        self,
        n_components: int = None,
        sparsity: float = None,
        max_iter: int = None,
        model_type: str = 'standard',
        preset: str = 'default',
        **kwargs
    ):
        """
        Initialize the segmenter.
        
        Parameters:
        -----------
        n_components : int
            Number of components (phases) to extract
        sparsity : float
            L1 sparsity constraint strength (used for both alpha_W and alpha_H)
        max_iter : int
            Maximum iterations for NMF
        model_type : str
            Type of NMF model: 'standard', 'spatial', 'orthogonal'
        preset : str
            Configuration preset: 'default', 'high_sparsity', 'fine_detail'
        **kwargs : dict
            Additional model-specific parameters (overrides config)
        """
        # Load preset if specified
        if preset in PRESETS:
            config = PRESETS[preset].copy()
        else:
            config = NMF_CONFIG.copy()
        
        # Override with provided parameters
        self.n_components = n_components if n_components is not None else config.get('n_components', 3)
        self.max_iter = max_iter if max_iter is not None else config.get('max_iter', 200)
        
        # Handle sparsity parameter
        if sparsity is not None:
            self.alpha_W = sparsity
            self.alpha_H = sparsity
        else:
            self.alpha_W = config.get('alpha_W', 0.0)
            self.alpha_H = config.get('alpha_H', 0.0)
        
        # Get other config parameters
        self.solver = kwargs.get('solver', config.get('solver', 'cd'))
        self.beta_loss = kwargs.get('beta_loss', config.get('beta_loss', 'frobenius'))
        self.l1_ratio = kwargs.get('l1_ratio', config.get('l1_ratio', 0.5))
        self.tol = kwargs.get('tol', config.get('tol', 1e-4))
        
        self.model_type = model_type
        self.model = None
        self.W = None
        self.H = None
        self.kwargs = kwargs
        
        self._initialize_model()
        
    def _clone_model(self, **overrides):
        """Create a new sklearn NMF object with optional overrides."""
        params = dict(
            n_components=self.n_components,
            init=self.model.init,
            solver=self.model.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=42,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio
        )
        params.update(overrides)
        return NMF(**params)
    
    @staticmethod
    def _components_valid(W: np.ndarray, H: np.ndarray, min_coverage: float = 0.005) -> bool:
        """Check that every component has meaningful coverage and energy."""
        if W is None or H is None:
            return False
        if np.any(np.isnan(W)) or np.any(np.isnan(H)):
            return False
        if np.min(H.sum(axis=1)) < 1e-8:
            return False
        coverage = (W > 1e-5).sum(axis=0) / W.shape[0]
        return np.all(coverage > min_coverage)
    
    def _fit_with_retries(self, X_train: np.ndarray, verbose: bool = True):
        """Attempt to fit NMF model with multiple inits/solvers for stability."""
        init_candidates = ['nndsvda', 'random']
        solver_candidates = [self.solver, 'mu']
        for attempt in range(3):
            init_choice = init_candidates[min(attempt, len(init_candidates)-1)]
            solver_choice = solver_candidates[min(attempt, len(solver_candidates)-1)]
            alpha_scale = max(0.3 ** attempt, 0.05)
            max_iter = max(self.max_iter, 400) if self.n_components >= 4 else self.max_iter + attempt * 100
            if verbose:
                print(f"Attempt {attempt+1}: init={init_choice}, solver={solver_choice}, alpha_scale={alpha_scale:.3f}, max_iter={max_iter}")
            model = self._clone_model(
                init=init_choice,
                solver=solver_choice,
                alpha_W=self.alpha_W * alpha_scale,
                alpha_H=self.alpha_H * alpha_scale,
                max_iter=max_iter
            )
            try:
                W = model.fit_transform(X_train)
                H = model.components_
            except Exception as exc:
                if verbose:
                    print(f"  Fit failed with error: {exc}")
                continue
            if self._components_valid(W, H):
                self.model = model
                self.W = W
                self.H = H
                if verbose:
                    print(f"  Fit successful. Coverage per component: {(W > 1e-5).sum(axis=0) / W.shape[0]}")
                return True
            if verbose:
                print("  Components collapsed; retrying with relaxed regularization...")
        return False
    
    def _initialize_model(self):
        """Initialize the appropriate NMF model based on model_type."""
        if self.model_type == 'standard':
            # Standard NMF with literature-based parameters
            # Use 'random' init when n_components might be close to n_features
            init_method = 'random' if self.n_components >= 5 else 'nndsvda'
            
            self.model = NMF(
                n_components=self.n_components,
                init=init_method,
                solver=self.solver,
                beta_loss=self.beta_loss,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42,
                alpha_W=self.alpha_W,
                alpha_H=self.alpha_H,
                l1_ratio=self.l1_ratio
            )
        elif self.model_type == 'spatial':
            smoothness = self.kwargs.get('smoothness', 0.5)
            self.model = SpatiallyConstrainedNMF(
                n_components=self.n_components,
                sparsity_w=self.sparsity,
                sparsity_h=self.sparsity,
                smoothness=smoothness,
                max_iter=self.max_iter
            )
        elif self.model_type == 'orthogonal':
            orthogonality = self.kwargs.get('orthogonality_penalty', 1.0)
            self.model = OrthogonalNMF(
                n_components=self.n_components,
                orthogonality_penalty=orthogonality,
                max_iter=self.max_iter
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(
        self,
        image_paths: List[str],
        target_size: Tuple[int, int] = (512, 512),
        use_texture: bool = True,
        verbose: bool = True
    ):
        """
        Train the NMF model on a set of images.
        
        Parameters:
        -----------
        image_paths : list of str
            Paths to training images
        target_size : tuple
            Resize images to this size
        use_texture : bool
            Extract texture features
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"Training {self.model_type} NMF on {len(image_paths)} images...")
        
        # Extract features from all training images
        training_features = []
        for img_path in image_paths:
            if verbose:
                print(f"  Loading {Path(img_path).name}...")
            image = load_and_preprocess_image(img_path, target_size=target_size)
            features = extract_features(image, use_texture=use_texture)
            training_features.append(features)
        
        # Stack features
        X_train = np.vstack(training_features)
        X_train = np.maximum(X_train, 1e-10)  # Ensure positivity
        
        if verbose:
            print(f"Training data shape: {X_train.shape}")
            print(f"Training data range: [{X_train.min():.6f}, {X_train.max():.6f}]")
        
        # Fit model
        if self.model_type == 'standard':
            success = self._fit_with_retries(X_train, verbose=verbose)
            if not success:
                raise RuntimeError("Failed to train a stable NMF model after multiple attempts")
            
            # Check convergence
            n_iter = getattr(self.model, 'n_iter_', self.model.max_iter)
            if verbose:
                print(f"NMF converged in {n_iter} iterations (max: {self.model.max_iter})")
                print(f"W range: [{self.W.min():.6f}, {self.W.max():.6f}]")
                print(f"H range: [{self.H.min():.6f}, {self.H.max():.6f}]")
            
            # Normalize H to have unit norm columns for better numerical stability
            H_norms = np.linalg.norm(self.H, axis=1, keepdims=True) + 1e-10
            self.H = self.H / H_norms
            
            if verbose:
                print(f"After normalization - H range: [{self.H.min():.6f}, {self.H.max():.6f}]")
        else:
            # For custom models
            image_shape = (target_size[0], target_size[1])
            if self.model_type == 'spatial':
                self.model.fit(X_train, image_shape=image_shape, verbose=verbose)
            else:
                self.model.fit(X_train, verbose=verbose)
            self.W = self.model.W
            self.H = self.model.H
        
        # Compute reconstruction error
        reconstruction = self.W @ self.H
        error = np.linalg.norm(X_train - reconstruction, 'fro')
        
        if verbose:
            print(f"Reconstruction error: {error:.4f}")
            print(f"W shape: {self.W.shape}, H shape: {self.H.shape}")
            print("Training complete!")
        
        return self
    
    def transform_features(self, X: np.ndarray, max_iter: int = 200, verbose: bool = False) -> np.ndarray:
        """
        Transform features to component activations using learned H matrix.
        More robust than sklearn's transform method.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        max_iter : int
            Maximum iterations for optimization
        verbose : bool
            Print progress
            
        Returns:
        --------
        W_new : np.ndarray
            Component activations (n_samples, n_components)
        """
        if self.H is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Initialize W with small random values
        W_new = np.random.rand(X.shape[0], self.n_components) * 0.01 + 1e-10
        
        # Use fixed H from training
        H_fixed = np.maximum(self.H, 1e-10)
        
        # Multiplicative update to solve for W
        # Minimize ||X - W*H||^2 with W >= 0
        for iteration in range(max_iter):
            # Update rule: W *= (X @ H^T) / (W @ H @ H^T)
            numerator = X @ H_fixed.T
            denominator = W_new @ (H_fixed @ H_fixed.T) + 1e-10
            
            W_new = W_new * (numerator / denominator)
            W_new = np.maximum(W_new, 1e-10)
            
            # Check convergence every 50 iterations
            if verbose and iteration % 50 == 0:
                reconstruction = W_new @ H_fixed
                error = np.linalg.norm(X - reconstruction, 'fro') / np.linalg.norm(X, 'fro')
                print(f"  Transform iteration {iteration}, relative error: {error:.6f}")
                
                if error < 0.01:  # Good convergence
                    if verbose:
                        print(f"  Converged at iteration {iteration}")
                    break
        
        return W_new
    
    def segment(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (512, 512),
        use_texture: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Segment a single image.
        
        Parameters:
        -----------
        image_path : str
            Path to image
        target_size : tuple
            Resize to this size
        use_texture : bool
            Use texture features
            
        Returns:
        --------
        image : np.ndarray
            Preprocessed image
        segmentation : np.ndarray
            Segmentation map
        component_maps : list of np.ndarray
            Component activation maps
        """
        if self.W is None or self.H is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Load and preprocess
        image = load_and_preprocess_image(image_path, target_size=target_size)
        
        # Extract features
        X = extract_features(image, use_texture=use_texture)
        X = np.maximum(X, 1e-10)
        
        # Transform
        try:
            if self.model_type == 'standard':
                W_new = self.model.transform(X)
            else:
                W_new = self.model.transform(X)
        except (ValueError, RuntimeError):
            # Fallback manual transform
            print("Using fallback transform method")
            W_new = np.abs(np.random.randn(X.shape[0], self.n_components)) + 1e-10
            H_fixed = np.maximum(self.H, 1e-10)
            
            for _ in range(100):
                numerator = X @ H_fixed.T
                denominator = W_new @ H_fixed @ H_fixed.T + 1e-10
                W_new *= numerator / denominator
                W_new = np.maximum(W_new, 1e-10)
        
        # Segment
        segmentation = np.argmax(W_new, axis=1).reshape(image.shape)
        
        # Component maps
        component_maps = [W_new[:, i].reshape(image.shape) for i in range(self.n_components)]
        
        return image, segmentation, component_maps
    
    def batch_segment(
        self,
        image_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        save_masks: bool = True,
        save_visualizations: bool = True
    ):
        """
        Segment all images in a directory.
        
        Parameters:
        -----------
        image_dir : str
            Input directory
        output_dir : str
            Output directory
        target_size : tuple
            Image size
        save_masks : bool
            Save segmentation masks
        save_visualizations : bool
            Save visualization plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_paths = sorted(Path(image_dir).glob('*.jpg'))
        print(f"Segmenting {len(image_paths)} images...")
        
        for img_path in image_paths:
            print(f"  Processing {img_path.name}...")
            
            image, segmentation, component_maps = self.segment(
                str(img_path), target_size=target_size
            )
            
            # Save visualization
            if save_visualizations:
                vis_file = output_path / f"segmented_{img_path.stem}.png"
                visualize_segmentation(
                    image, segmentation, component_maps, 
                    self.n_components, save_path=str(vis_file)
                )
            
            # Save mask
            if save_masks:
                mask_file = output_path / f"mask_{img_path.stem}.png"
                cv2.imwrite(
                    str(mask_file), 
                    (segmentation * (255 // self.n_components)).astype(np.uint8)
                )
        
        print(f"Batch segmentation complete! Results in {output_dir}")
    
    def compare_models(
        self,
        image_path: str,
        model_configs: dict,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Compare different NMF model variants on the same image.
        
        Parameters:
        -----------
        image_path : str
            Path to test image
        model_configs : dict
            Dictionary of {name: config} for different models
        target_size : tuple
            Image size
        """
        image = load_and_preprocess_image(image_path, target_size=target_size)
        X = extract_features(image, use_texture=True)
        
        n_models = len(model_configs)
        fig, axes = plt.subplots(2, (n_models + 2) // 2, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Test each model
        for idx, (name, config) in enumerate(model_configs.items(), start=1):
            print(f"\nTesting {name}...")
            
            segmenter = MicrostructureSegmenter(**config)
            segmenter.train([image_path], target_size=target_size, verbose=False)
            _, segmentation, _ = segmenter.segment(image_path, target_size=target_size)
            
            axes[idx].imshow(segmentation, cmap='tab10')
            axes[idx].set_title(name)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for i in range(n_models + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison saved to model_comparison.png")
        plt.close()


# Convenience functions for quick usage

def quick_segment(
    image_path: str,
    n_components: int = 5,
    sparsity: float = 0.1,
    output_path: Optional[str] = None
):
    """
    Quick segmentation of a single image with default settings.
    
    Parameters:
    -----------
    image_path : str
        Path to image
    n_components : int
        Number of phases
    sparsity : float
        Sparsity constraint
    output_path : str, optional
        Save path for visualization
    """
    segmenter = MicrostructureSegmenter(
        n_components=n_components,
        sparsity=sparsity
    )
    
    # Train on the single image
    segmenter.train([image_path], verbose=True)
    
    # Segment
    image, segmentation, component_maps = segmenter.segment(image_path)
    
    # Visualize
    visualize_segmentation(
        image, segmentation, component_maps,
        n_components, save_path=output_path
    )
    
    return image, segmentation, component_maps


def batch_process_dataset(
    image_dir: str = "images",
    output_dir: str = "segmentation_results",
    n_components: int = 5,
    sparsity: float = 0.1,
    training_samples: int = 10,
    model_type: str = 'standard',
    **kwargs
):
    """
    Process an entire dataset of images.
    
    Parameters:
    -----------
    image_dir : str
        Input directory
    output_dir : str
        Output directory
    n_components : int
        Number of components
    sparsity : float
        Sparsity constraint
    training_samples : int
        Number of images for training
    model_type : str
        'standard', 'spatial', or 'orthogonal'
    **kwargs : dict
        Additional model parameters
    """
    # Get image paths
    image_paths = sorted(Path(image_dir).glob('*.jpg'))
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize segmenter
    segmenter = MicrostructureSegmenter(
        n_components=n_components,
        sparsity=sparsity,
        model_type=model_type,
        **kwargs
    )
    
    # Train on subset
    training_paths = [str(p) for p in image_paths[:training_samples]]
    segmenter.train(training_paths, verbose=True)
    
    # Segment all
    segmenter.batch_segment(image_dir, output_dir)


if __name__ == "__main__":
    # Example usage
    batch_process_dataset(
        image_dir="images",
        output_dir="segmentation_results",
        n_components=5,
        sparsity=0.1,
        training_samples=10,
        model_type='standard'
    )
