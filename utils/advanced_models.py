"""
Advanced NMF models with various constraint types
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


class SpatiallyConstrainedNMF:
    """
    NMF with spatial smoothness constraints for microstructure segmentation.
    
    Uses custom multiplicative update rules that incorporate spatial smoothness
    penalties to encourage coherent segmentation regions.
    """
    
    def __init__(
        self,
        n_components: int = 5,
        sparsity_w: float = 0.0,
        sparsity_h: float = 0.0,
        smoothness: float = 0.1,
        max_iter: int = 500,
        tolerance: float = 1e-4,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.sparsity_w = sparsity_w
        self.sparsity_h = sparsity_h
        self.smoothness = smoothness
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.W = None
        self.H = None
        self.image_shape = None
        
    def _initialize_matrices(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize W and H matrices using NNDSVD strategy."""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Use SVD for better initialization
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        W = np.abs(U[:, :self.n_components] @ np.diag(np.sqrt(S[:self.n_components])))
        H = np.abs(np.diag(np.sqrt(S[:self.n_components])) @ Vt[:self.n_components, :])
        
        # Add small random noise
        W += 0.01 * np.random.rand(*W.shape)
        H += 0.01 * np.random.rand(*H.shape)
        
        return W, H
    
    def _spatial_smoothness_penalty(self, W: np.ndarray) -> np.ndarray:
        """Compute spatial smoothness penalty gradient."""
        if self.image_shape is None or self.smoothness == 0:
            return np.zeros_like(W)
        
        h, w = self.image_shape
        penalty = np.zeros_like(W)
        
        for k in range(self.n_components):
            component_map = W[:, k].reshape(h, w)
            smoothed = gaussian_filter(component_map, sigma=1.0)
            laplacian = component_map - smoothed
            penalty[:, k] = laplacian.flatten()
        
        return self.smoothness * penalty
    
    def _multiplicative_update(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one iteration of multiplicative updates with constraints."""
        eps = 1e-10
        
        # Update H (coefficient matrix)
        numerator_h = W.T @ X
        denominator_h = W.T @ W @ H + self.sparsity_h + eps
        H *= numerator_h / denominator_h
        
        # Update W (basis matrix) with spatial smoothness
        numerator_w = X @ H.T
        smoothness_grad = self._spatial_smoothness_penalty(W)
        denominator_w = W @ H @ H.T + self.sparsity_w + smoothness_grad + eps
        W *= numerator_w / np.maximum(denominator_w, eps)
        
        # Ensure non-negativity
        W = np.maximum(W, eps)
        H = np.maximum(H, eps)
        
        return W, H
    
    def fit(self, X: np.ndarray, image_shape: Optional[Tuple[int, int]] = None, verbose: bool = True):
        """Fit the spatially constrained NMF model."""
        self.image_shape = image_shape
        self.W, self.H = self._initialize_matrices(X)
        
        if verbose:
            print(f"Fitting Spatially Constrained NMF...")
            print(f"Components: {self.n_components}, Smoothness: {self.smoothness}")
        
        prev_error = np.inf
        
        for iteration in range(self.max_iter):
            self.W, self.H = self._multiplicative_update(X, self.W, self.H)
            
            reconstruction = self.W @ self.H
            error = np.linalg.norm(X - reconstruction, 'fro')
            
            if abs(prev_error - error) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Error: {error:.4f}")
            
            prev_error = error
        
        if verbose:
            print(f"Final reconstruction error: {error:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted H."""
        if self.H is None:
            raise ValueError("Model not fitted")
        
        W_new = self.W.copy()
        for _ in range(100):
            numerator = X @ self.H.T
            denominator = W_new @ self.H @ self.H.T + 1e-10
            W_new *= numerator / denominator
        
        return W_new
    
    def segment(self, X: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Segment an image."""
        W_new = self.transform(X)
        segmentation = np.argmax(W_new, axis=1).reshape(image_shape)
        return segmentation


class OrthogonalNMF:
    """Orthogonal NMF for distinct, non-overlapping component discovery."""
    
    def __init__(
        self,
        n_components: int = 5,
        orthogonality_penalty: float = 1.0,
        max_iter: int = 500,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.orthogonality_penalty = orthogonality_penalty
        self.max_iter = max_iter
        self.random_state = random_state
        self.W = None
        self.H = None
    
    def _orthogonality_constraint(self, H: np.ndarray) -> np.ndarray:
        """Compute gradient of orthogonality constraint on H."""
        HHT = H @ H.T
        identity = np.eye(self.n_components)
        return self.orthogonality_penalty * (HHT - identity) @ H
    
    def fit(self, X: np.ndarray, verbose: bool = True):
        """Fit orthogonal NMF."""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        self.W = np.abs(np.random.randn(n_samples, self.n_components))
        self.H = np.abs(np.random.randn(self.n_components, n_features))
        
        if verbose:
            print("Fitting Orthogonal NMF...")
        
        eps = 1e-10
        
        for iteration in range(self.max_iter):
            numerator_h = self.W.T @ X
            ortho_grad = self._orthogonality_constraint(self.H)
            denominator_h = self.W.T @ self.W @ self.H + ortho_grad + eps
            self.H *= numerator_h / np.maximum(denominator_h, eps)
            
            numerator_w = X @ self.H.T
            denominator_w = self.W @ self.H @ self.H.T + eps
            self.W *= numerator_w / denominator_w
            
            self.W = np.maximum(self.W, eps)
            self.H = np.maximum(self.H, eps)
            
            if verbose and (iteration + 1) % 50 == 0:
                reconstruction = self.W @ self.H
                error = np.linalg.norm(X - reconstruction, 'fro')
                print(f"Iteration {iteration + 1}/{self.max_iter}, Error: {error:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.H is None:
            raise ValueError("Model not fitted")
        
        W_new = np.abs(np.random.randn(X.shape[0], self.n_components))
        eps = 1e-10
        
        for _ in range(100):
            numerator = X @ self.H.T
            denominator = W_new @ self.H @ self.H.T + eps
            W_new *= numerator / denominator
        
        return W_new
