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


def create_colorbar_legend():
    """
    Create a color bar legend showing blue to yellow gradient.
    
    Returns:
    --------
    legend_image : np.ndarray
        RGB image of the color bar with labels
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 0.5))
    
    # Create colorbar
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    # Create a horizontal colorbar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=ax, orientation='horizontal')
    
    # Remove ticks and labels
    cbar.set_ticks([])
    cbar.set_label('')
    
    # Add custom labels
    ax.text(0.0, -0.5, 'Not Present', transform=ax.transAxes, 
            ha='left', va='top', fontsize=10, fontweight='bold')
    ax.text(1.0, -0.5, 'Present', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10, fontweight='bold')
    
    # Convert to image
    fig.canvas.draw()
    try:
        # New matplotlib (>= 3.8)
        buf = fig.canvas.buffer_rgba()
        legend_image = np.asarray(buf)
        legend_image = legend_image[:, :, :3]  # Remove alpha channel
    except AttributeError:
        try:
            # Matplotlib 3.x
            legend_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            legend_image = legend_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback for older matplotlib
            legend_image = np.frombuffer(fig.canvas.tobytes(), dtype=np.uint8)
            legend_image = legend_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return legend_image


def assign_phase_names(W_normalized, n_components):
    """
    Assign metallographic phase names to components based on their characteristics.
    
    Parameters:
    -----------
    W_normalized : np.ndarray
        Normalized component activations (n_pixels, n_components)
    n_components : int
        Number of components
        
    Returns:
    --------
    list of str
        Phase names for each component
    """
    # Calculate mean activation for each component
    mean_activations = W_normalized.mean(axis=0)
    
    # Sort components by mean activation (lowest to highest)
    sorted_indices = np.argsort(mean_activations)
    
    # Standard metallographic phases
    phase_names = {
        2: ["Defects/Voids", "Matrix"],
        3: ["Defects/Voids", "Austenite", "Matrix"],
        4: ["Defects/Voids", "Martensite", "Austenite", "Matrix"],
        5: ["Defects/Voids", "Precipitates", "Martensite", "Austenite", "Matrix"]
    }
    
    # Get appropriate phase names for n_components
    if n_components in phase_names:
        names = phase_names[n_components]
    else:
        # Fallback for other numbers
        names = [f"Phase {i+1}" for i in range(n_components)]
    
    # Assign names based on sorted activation levels
    component_names = [None] * n_components
    for i, idx in enumerate(sorted_indices):
        if i < len(names):
            component_names[idx] = names[i]
        else:
            component_names[idx] = f"Phase {idx+1}"
    
    return component_names


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
            return [None] * 10
        
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
                return [None] * 10
            self.last_config = current_config
        
        # Extract features and segment
        X = extract_features(image, use_texture=True)
        X = np.maximum(X, 1e-10)
        
        print(f"Feature matrix shape: {X.shape}, range: [{X.min():.6f}, {X.max():.6f}]")
        
        # Use sklearn's transform method for better stability
        print("Transforming features to component activations...")
        print(f"Model type: {type(self.model)}")
        print(f"Internal model type: {type(self.model.model)}")
        print(f"Has transform method: {hasattr(self.model.model, 'transform')}")
        
        try:
            W_new = self.model.model.transform(X)
            print(f"sklearn transform successful, W_new shape: {W_new.shape}")
        except Exception as e:
            print(f"sklearn transform failed: {e}, using manual transform")
            W_new = self.model.transform_features(X, max_iter=200, verbose=True)
        
        print(f"W_new shape: {W_new.shape}, range: [{W_new.min():.6f}, {W_new.max():.6f}]")
        
        # Check if transformation is valid
        if W_new.max() < 1e-6 or np.std(W_new) < 1e-6:
            print("ERROR: Transform resulted in degenerate values!")
            return [None] * 10
        
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
            return [None] * 10
        
        # Check if any label dominates (>95% of pixels)
        max_dominance = counts.max() / counts.sum()
        if max_dominance > 0.95:
            print(f"WARNING: One component dominates {max_dominance*100:.1f}% of image")
        
        # Assign phase names based on component characteristics
        component_names = assign_phase_names(W_normalized, n_components)
        
        component_maps = [W_normalized[:, i].reshape(image.shape) for i in range(n_components)]
        
        # Create visualizations
        original_vis = (image * 255).astype(np.uint8)
        
        # Segmentation with color map
        seg_colored = colorize_segmentation(segmentation, n_components)
        
        # Create component maps with better contrast and overlay information
        component_images = []
        for i, comp_map in enumerate(component_maps):
            # Improve contrast – fall back to segmentation mask if nearly constant
            comp_vis = comp_map.copy()
            dynamic_range = comp_vis.max() - comp_vis.min()
            if dynamic_range < 1e-5:
                comp_vis = (segmentation == i).astype(np.float32)
            
            # Clip outliers for better visualization (use percentile normalization)
            p_low = np.percentile(comp_vis, 2)
            p_high = np.percentile(comp_vis, 98)
            if p_high - p_low < 1e-5:
                p_low, p_high = comp_vis.min(), comp_vis.max()
            comp_clipped = np.clip(comp_vis, p_low, p_high)
            
            # Normalize to 0-255 with stable scaling
            comp_uint8 = cv2.normalize(comp_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            comp_uint8 = comp_uint8.astype(np.uint8)
            
            # Apply colormap for better visualization
            comp_colored = cv2.applyColorMap(comp_uint8, cv2.COLORMAP_VIRIDIS)
            comp_colored = cv2.cvtColor(comp_colored, cv2.COLOR_BGR2RGB)
            
            # Add phase name overlay
            phase_name = component_names[i] if i < len(component_names) else f"Phase {i+1}"
            cv2.putText(comp_colored, phase_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            component_images.append(comp_colored)
            
            print(f"Component {i} ({phase_name}): activation range [{comp_map.min():.4f}, {comp_map.max():.4f}], coverage: {(comp_map > 0.1).sum() / comp_map.size * 100:.1f}%")
        
        # Ensure we always have 4 names for downstream updates
        while len(component_names) < 4:
            component_names.append("")
        
        # Build dynamic updates for component displays and names
        component_image_updates = []
        component_name_updates = []
        for idx in range(4):
            name = component_names[idx] if component_names[idx] else f"Phase {idx+1}"
            if idx < len(component_images):
                img = component_images[idx]
                component_image_updates.append(gr.update(value=img, label=name))
                component_name_updates.append(gr.update(value=name))
            else:
                component_image_updates.append(gr.update(value=None, label=name))
                component_name_updates.append(gr.update(value=""))
        
        return (
            original_vis,
            seg_colored,
            *component_image_updates,
            *component_name_updates
        )
    


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
            return [None] * 10  # Return None for all outputs (6 images + 4 names)
        
        return result  # Returns: original, segmentation, comp1, comp2, comp3, comp4, name1, name2, name3, name4
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return [None] * 10


# Create Gradio interface
with gr.Blocks(title="Microstructure Segmentation") as demo:
    gr.Markdown(
        """
        # 🔬 Microstructure Segmentation using Constrained NMF
        
        Upload a micrograph image to segment different metallurgical phases using 
        Non-negative Matrix Factorization with sparsity constraints.
        
        **Model automatically trains on sample images from the `images/` directory on first use.**
        
        ## Understanding the Results
        
        ### 🎨 What Do the Colors Represent?
        
        **Segmentation Map** (right): Each color shows which microstructural phase "wins" at each pixel location.
        
        **Component Maps** (bottom): Heatmaps showing how strongly each phase is present at each location:
        - 🔵 **Blue** = Low activation (phase not present)
        - 🟡 **Yellow** = High activation (phase strongly present)
        - Components are NOT mutually exclusive - pixels can belong to multiple phases!
        
        **Phase Names**: Automatically identified based on activation patterns (Defects, Austenite, Matrix, etc.)
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
            
            # Add color bar legend
            colorbar_legend = gr.Image(
                value=create_colorbar_legend(),
                label="Activation Scale",
                height=50,
                show_label=False
            )
            gr.Markdown("**🔵 Blue = Not Present** | **🟡 Yellow = Present**")
            
            with gr.Row():
                component1_name = gr.Textbox(label="Phase 1 Name", interactive=False)
                component2_name = gr.Textbox(label="Phase 2 Name", interactive=False)
            
            with gr.Row():
                component3_name = gr.Textbox(label="Phase 3 Name", interactive=False)
                component4_name = gr.Textbox(label="Phase 4 Name", interactive=False)
            
            with gr.Row():
                component1_out = gr.Image(label="Phase 1", height=250)
                component2_out = gr.Image(label="Phase 2", height=250)
            
            with gr.Row():
                component3_out = gr.Image(label="Phase 3", height=250)
                component4_out = gr.Image(label="Phase 4", height=250)
    
    # Examples
    gr.Markdown("### Example Images")
    gr.Examples(
        examples=[
            ["images/image_0.jpg", 3, 0.01, 512, 'default', 'cd', 'frobenius', 300],
            ["images/image_1.jpg", 3, 0.15, 512, 'high_sparsity', 'mu', 'frobenius', 400],
            ["images/image_2.jpg", 4, 0.05, 512, 'fine_detail', 'cd', 'frobenius', 350],
        ],
        inputs=[input_image, n_components, sparsity, target_size, preset, solver, beta_loss, max_iter],
        outputs=[original_out, segmentation_out, component1_out, component2_out, component3_out, component4_out,
                component1_name, component2_name, component3_name, component4_name],
        fn=process_image,
        cache_examples=False,
    )
    
    # Connect button
    segment_btn.click(
        fn=process_image,
        inputs=[input_image, n_components, sparsity, target_size, preset, solver, beta_loss, max_iter],
        outputs=[original_out, segmentation_out, component1_out, component2_out, component3_out, component4_out, 
                component1_name, component2_name, component3_name, component4_name]
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
        
        **💡 Pro Tip:** If components look too similar, try increasing sparsity or using the 'High Sparsity' preset for more distinct phase separation!
        """
    )


if __name__ == "__main__":
    # Launch with network sharing enabled
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7861,  # Changed port to avoid conflicts
        share=True,  # Set to True for public Gradio link
        inbrowser=True  # Auto-open in browser
    )
