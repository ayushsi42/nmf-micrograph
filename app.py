"""
Web UI for Microstructure Segmentation using Constrained NMF

A simple Gradio interface to upload micrographs and visualize segmentation results.
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from microstructure_segmentation import MicrostructureSegmenter as NMFSegmenter
from utils.preprocessing import load_and_preprocess_image, extract_features
from utils.visualization import colorize_segmentation, create_component_grid
from config import PRESETS


class MicrostructureSegmenterUI:
    """Wrapper class for the segmentation model with caching."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.last_config = None  # Track last configuration
        
    def train_model(self, n_components=3, sparsity=0.0, preset='default', solver='cd', beta_loss='frobenius', max_iter=200):
        """Train the NMF model on sample images."""
        print("Training NMF model...")
        
        # Initialize model
        self.model = NMFSegmenter(
            n_components=n_components,
            sparsity=sparsity,
            max_iter=max_iter,
            preset=preset,
            solver=solver,
            beta_loss=beta_loss
        )
        
        # Load sample images for training
        image_dir = Path("images")
        if not image_dir.exists():
            return False, "No training images found in 'images/' directory"
        
        image_paths = sorted(image_dir.glob('*.jpg'))[:10]
        if len(image_paths) == 0:
            return False, "No .jpg images found in 'images/' directory"
        
        # Train model
        training_paths = [str(p) for p in image_paths]
        self.model.train(training_paths, verbose=True)
        self.is_trained = True
        
        return True, f"Model trained successfully on {len(image_paths)} images"
    
    def segment_image(self, input_image, n_components=3, sparsity=0.0, target_size=512, 
                     preset='default', solver='cd', beta_loss='frobenius', max_iter=200):
        """
        Segment an uploaded image.
        
        Parameters:
        -----------
        input_image : PIL.Image or numpy.ndarray
            Input image
        n_components : int
            Number of components
        sparsity : float
            Sparsity constraint
        target_size : int
            Resize dimension
        preset : str
            Configuration preset
        solver : str
            NMF solver
        beta_loss : str
            Beta loss function
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        tuple : (original, segmentation, comp1, comp2, comp3, comp4)
        """
        if input_image is None:
            return [None] * 6
        
        # Convert PIL to numpy if needed
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        
        # Convert to grayscale if RGB
        if len(input_image.shape) == 3:
            image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            image = input_image
        
        # Resize
        image = cv2.resize(image, (target_size, target_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_uint8 = (image * 255).astype(np.uint8)
        enhanced = clahe.apply(image_uint8)
        image = enhanced.astype(np.float32) / 255.0
        
        # Check if model needs retraining (new config or not trained)
        current_config = (n_components, sparsity, preset, solver, beta_loss, max_iter)
        if not self.is_trained or self.last_config != current_config:
            print(f"Training model with config: n_components={n_components}, sparsity={sparsity}, preset={preset}")
            success, message = self.train_model(n_components, sparsity, preset, solver, beta_loss, max_iter)
            if not success:
                return [None] * 6
            self.last_config = current_config
        
        # Extract features and segment
        X = extract_features(image, use_texture=True)
        X = np.maximum(X, 1e-10)
        
        print(f"Feature matrix shape: {X.shape}, range: [{X.min():.6f}, {X.max():.6f}]")
        
        # Use our custom transform method (more robust than sklearn's)
        print("Transforming features to component activations...")
        W_new = self.model.transform_features(X, max_iter=200, verbose=True)
        
        print(f"W_new shape: {W_new.shape}, range: [{W_new.min():.6f}, {W_new.max():.6f}]")
        
        # Check if transformation is valid
        if W_new.max() < 1e-6 or np.std(W_new) < 1e-6:
            print("ERROR: Transform resulted in degenerate values!")
            return [None] * 6
        
        # Normalize W_new for better visualization and segmentation
        # Softmax-like normalization ensures components are comparable
        W_sum = W_new.sum(axis=1, keepdims=True) + 1e-10
        W_normalized = W_new / W_sum
        
        print(f"W_normalized range per component:")
        for i in range(n_components):
            print(f"  Component {i}: [{W_normalized[:, i].min():.4f}, {W_normalized[:, i].max():.4f}], "
                  f"mean: {W_normalized[:, i].mean():.4f}")
        
        # Segment using normalized weights
        segmentation = np.argmax(W_normalized, axis=1).reshape(image.shape)
        
        # Check if segmentation is meaningful
        unique_labels, counts = np.unique(segmentation, return_counts=True)
        print(f"Unique segmentation labels: {unique_labels}")
        print(f"Label distribution: {dict(zip(unique_labels, counts / counts.sum() * 100))}")
        
        if len(unique_labels) < 2:
            print("ERROR: Segmentation failed to find multiple components")
            return [None] * 6
        
        # Check if any label dominates (>95% of pixels)
        max_dominance = counts.max() / counts.sum()
        if max_dominance > 0.95:
            print(f"WARNING: One component dominates {max_dominance*100:.1f}% of image")
        
        component_maps = [W_normalized[:, i].reshape(image.shape) for i in range(n_components)]
        
        # Create visualizations
        original_vis = (image * 255).astype(np.uint8)
        
        # Segmentation with color map
        seg_colored = colorize_segmentation(segmentation, n_components)
        
        # Convert component maps to uint8 images (normalize to 0-255)
        component_images = []
        for i, comp_map in enumerate(component_maps):
            # Clip outliers for better visualization (use percentile normalization)
            p_low = np.percentile(comp_map, 2)
            p_high = np.percentile(comp_map, 98)
            comp_clipped = np.clip(comp_map, p_low, p_high)
            
            # Normalize to 0-255
            comp_normalized = (comp_clipped - comp_clipped.min()) / (comp_clipped.max() - comp_clipped.min() + 1e-10)
            comp_uint8 = (comp_normalized * 255).astype(np.uint8)
            
            # Apply colormap for better visualization
            comp_colored = cv2.applyColorMap(comp_uint8, cv2.COLORMAP_VIRIDIS)
            comp_colored = cv2.cvtColor(comp_colored, cv2.COLOR_BGR2RGB)
            component_images.append(comp_colored)
            
            print(f"Component {i}: activation range [{comp_map.min():.4f}, {comp_map.max():.4f}], coverage: {(comp_map > 0.1).sum() / comp_map.size * 100:.1f}%")
        
        # Pad with None for unused components (up to 4)
        while len(component_images) < 4:
            component_images.append(None)
        
        return original_vis, seg_colored, *component_images
    


# Initialize segmenter
segmenter = MicrostructureSegmenterUI()


def process_image(image, n_components, sparsity, target_size, preset, solver, beta_loss, max_iter):
    """Process uploaded image and return visualizations."""
    try:
        result = segmenter.segment_image(
            image, 
            n_components=n_components, 
            sparsity=sparsity,
            target_size=target_size,
            preset=preset,
            solver=solver,
            beta_loss=beta_loss,
            max_iter=max_iter
        )
        
        if result[0] is None:
            return [None] * 6  # Return None for all outputs
        
        return result  # Returns: original, segmentation, comp1, comp2, comp3, comp4
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return [None] * 6


# Create Gradio interface
with gr.Blocks(title="Microstructure Segmentation") as demo:
    gr.Markdown(
        """
        # 🔬 Microstructure Segmentation using Constrained NMF
        
        Upload a micrograph image to segment different metallurgical phases using 
        Non-negative Matrix Factorization with sparsity constraints.
        
        **Model automatically trains on sample images from the `images/` directory on first use.**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                label="Upload Micrograph",
                type="pil",
                height=400
            )
            
            # Parameters
            gr.Markdown("### Basic Parameters")
            
            preset = gr.Dropdown(
                choices=['default', 'high_sparsity', 'fine_detail'],
                value='default',
                label="Configuration Preset",
                info="Choose a preset configuration"
            )
            
            n_components = gr.Slider(
                minimum=2,
                maximum=4,
                value=3,
                step=1,
                label="Number of Components (Phases)",
                info="Number of distinct phases to identify (2-4)"
            )
            
            sparsity = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.01,
                step=0.01,
                label="Sparsity Constraint",
                info="L1 regularization strength (0.01-0.15 typical, 0 = no constraint)"
            )
            
            target_size = gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=128,
                label="Image Size (pixels)",
                info="Larger = more detail, slower processing"
            )
            
            # Advanced parameters
            with gr.Accordion("Advanced Parameters", open=False):
                gr.Markdown("*Override preset configuration with custom parameters*")
                
                solver = gr.Dropdown(
                    choices=['cd', 'mu'],
                    value='cd',
                    label="Solver",
                    info="cd=Coordinate Descent (fast), mu=Multiplicative Update (stable)"
                )
                
                beta_loss = gr.Dropdown(
                    choices=['frobenius', 'kullback-leibler'],
                    value='frobenius',
                    label="Loss Function",
                    info="frobenius=Euclidean (standard), KL=Kullback-Leibler (sparse)"
                )
                
                max_iter = gr.Slider(
                    minimum=100,
                    maximum=500,
                    value=300,
                    step=50,
                    label="Max Iterations",
                    info="More iterations = better convergence, slower"
                )
            
            segment_btn = gr.Button("Segment Image", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # Outputs
            gr.Markdown("### Results")
            
            with gr.Row():
                original_out = gr.Image(label="Preprocessed Image", height=300)
                segmentation_out = gr.Image(label="Segmentation Map", height=300)
            
            gr.Markdown("### Individual Component Activations")
            gr.Markdown("*Each component represents a different microstructural phase*")
            
            with gr.Row():
                component1_out = gr.Image(label="Component 1", height=250)
                component2_out = gr.Image(label="Component 2", height=250)
            
            with gr.Row():
                component3_out = gr.Image(label="Component 3", height=250)
                component4_out = gr.Image(label="Component 4", height=250)
    
    # Examples
    gr.Markdown("### Example Images")
    gr.Examples(
        examples=[
            ["images/image_0.jpg", 3, 0.01, 512, 'default', 'cd', 'frobenius', 300],
            ["images/image_1.jpg", 3, 0.15, 512, 'high_sparsity', 'mu', 'frobenius', 400],
            ["images/image_2.jpg", 4, 0.05, 512, 'fine_detail', 'cd', 'frobenius', 350],
        ],
        inputs=[input_image, n_components, sparsity, target_size, preset, solver, beta_loss, max_iter],
        outputs=[original_out, segmentation_out, component1_out, component2_out, component3_out, component4_out],
        fn=process_image,
        cache_examples=False,
    )
    
    # Connect button
    segment_btn.click(
        fn=process_image,
        inputs=[input_image, n_components, sparsity, target_size, preset, solver, beta_loss, max_iter],
        outputs=[original_out, segmentation_out, component1_out, component2_out, component3_out, component4_out]
    )
    
    gr.Markdown(
        """
        ---
        **About the Method:**
        - Uses constrained Non-negative Matrix Factorization (NMF)
        - Based on Lee & Seung (1999) and Hoyer (2004)
        - Features: intensity, gradients, Laplacian, texture, multi-scale
        - Constraints: L1/L2 sparsity for distinct phase separation
        - Trained on OD_MetalDAM dataset (42 SEM micrographs)
        
        **Presets:**
        - **Default**: Balanced, no sparsity (general purpose)
        - **High Sparsity**: Strong regularization (distinct phases)
        - **Fine Detail**: Moderate sparsity (subtle features)
        """
    )


if __name__ == "__main__":
    # Launch with network sharing enabled
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,
        share=True,  # Set to True for public Gradio link
        inbrowser=True  # Auto-open in browser
    )
