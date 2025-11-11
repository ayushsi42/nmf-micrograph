from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("Voxel51/OD_MetalDAM")

# Create directories for images and masks if they don't exist
os.makedirs("images", exist_ok=True)

# Save all images and masks
for i, item in enumerate(dataset['train']):
    image = item['image']
    
    image.save(f"images/image_{i}.jpg")
    
    print(f"Saved image_{i}.jpg")
    
print("All images have been saved locally.")