import os
import torch
from IPython.display import display, HTML
from PIL import Image, ImageDraw
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Optional
from PIL import Image



def set_current_dir(CODE_DIR):
  os.chdir(f'{CODE_DIR}')
  
  notebook_path = os.getcwd()
  print('Current working directory is:', '\n', notebook_path, '\n') 

# Map GPU names to CUDA architectures
cuda_arch_map = {
    "A100": "8.0",
    "H100": "9.0",
    "V100": "7.0",
    "T4": "7.5",
    "RTX 3090": "8.6",
    "RTX 4090": "8.9"
}


def set_cuda_arch_list():
    if not torch.cuda.is_available():
        print("CUDA is not available. TORCH_CUDA_ARCH_LIST will not be set.")
        return

    # Get the GPU name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")

    # Match GPU name to CUDA architecture
    for key, arch in cuda_arch_map.items():
        if key in gpu_name:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print(f"Setting TORCH_CUDA_ARCH_LIST to {arch} for {key}")
            return
    # Verify the environment variable
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")


def show_images(*images, titles=None, img_size=(5, 5)):
    """
    Display multiple images in a single row.
    
    Args:
        images: Variable number of images (PIL.Image, NumPy arrays, or tensors).
        titles: Optional list of titles corresponding to each image.
        img_size: Tuple (width per image, height), default is (5,5).
    """
    num_images = len(images)
    
    # Create subplots dynamically based on number of images
    fig, axes = plt.subplots(1, num_images, figsize=(img_size[0] * num_images, img_size[1]))

    # Ensure axes is always iterable (for a single image case)
    if num_images == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        if not isinstance(img, np.ndarray):
            img = np.array(img)  # Convert PIL.Image to NumPy array if needed

        ax.imshow(img)
        ax.axis("off")  # Remove axes
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)

    plt.show()


def display_images(images, img_size=(256, 256), save_path=None):
    """
    Displays multiple images in a single row using HTML and Base64 encoding.
    
    Args:
        images: Variable number of PIL.Image or tensors to be displayed.
        img_size: Tuple (width, height) for resizing images (default: 256x256).
        save_path: Directory path to save images. If None, images are not saved.
    """
    img_tags = []
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            img = tensor2im(img)  # Convert tensor to PIL.Image if needed

        img = img.resize(img_size)  # Resize image

        # Save the image if save_path is provided
        if save_path:
            img_save_path = os.path.join(save_path, f"image_{i+1}.png")
            img.save(img_save_path)
            print(f"Saved: {img_save_path}")

        # Convert to Base64 for display
        buffer = BytesIO()
        img.save(buffer, format="PNG")  
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")  
        img_tags.append(f"<img src='data:image/png;base64,{img_str}' style='margin:5px;'>")

    # Combine images into a single row
    display(HTML(f"<div style='display:flex; align-items:center;'>{''.join(img_tags)}</div>"))


def display_and_save_images(images, img_size=(256, 256), save_path=None, filename="combined_image.png", padding=20):
    """
    Displays multiple images in a single row with white spacing and saves them as a single combined image.

    Args:
        images: List of PIL.Image or tensors to be displayed and saved.
        img_size: Tuple (width, height) for resizing images (default: 256x256).
        save_path: Directory path to save the combined image. If None, image is not saved.
        filename: Name of the saved image file (default: "combined_image.png").
        padding: Space (in pixels) between images (default: 20px).
    """
    # Convert tensors to PIL if needed
    images = [tensor2im(img) if not isinstance(img, Image.Image) else img for img in images]
    
    # Resize all images
    images = [img.resize(img_size) for img in images]

    # Compute the final size including padding
    total_width = sum(img.width for img in images) + (padding * (len(images) - 1))
    max_height = max(img.height for img in images)

    # Create a blank white image
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Paste images with padding
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding

    # Save the combined image if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        save_filepath = os.path.join(save_path, filename)
        combined_image.save(save_filepath)
        print(f"Saved combined image: {save_filepath}")

    # Display inline in Jupyter Notebook
    buffer = BytesIO()
    combined_image.save(buffer, format="PNG")  
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    display(HTML(f"<img src='data:image/png;base64,{img_str}' style='margin:5px;'>"))


import os
import random
import re

def natural_sort_key(s):
    """Sort file names numerically if they contain numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_files_index(folder_path, num_images, seed=None):
    """
    Selects a random subset of image indices from a folder (with consistent order).

    Args:
        folder_path (str): Path to the folder containing images.
        num_images (int): Number of images to select.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: List of selected image indices.
    """
    # List all image files in the folder and sort naturally (numerical order)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    image_files.sort(key=natural_sort_key)  # Sort numerically

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")

    # Select random indices
    num_available = len(image_files)

    if num_available < num_images:
        print(f"Warning: Only {num_available} images found, selecting all.")
        selected_indices = list(range(num_available))
    else:
        selected_indices = random.sample(range(num_available), num_images)

    print(f"Selected image indices: {selected_indices}")
    return selected_indices

from PIL import Image

def load_images_by_index(folder_path, indices, img_size=(256, 256)):
    """
    Loads specific images by index from a given folder while preserving a consistent order.

    Args:
        folder_path (str): Path to the folder containing images.
        indices (list): List of indices of images to load.
        img_size (tuple): Target size (width, height) for resizing images.

    Returns:
        list: List of resized PIL images.
    """
    # List all image files and sort them using natural sorting (same as get_files_index)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    image_files.sort(key=natural_sort_key)  # Sort numerically

    # Ensure indices are within range
    num_available = len(image_files)
    selected_indices = [i for i in indices if i < num_available]

    if len(selected_indices) != len(indices):
        print(f"Warning: Some indices are out of range. Available images: {num_available}")

    print(f"Selected image indices: {selected_indices}")
    
    images = []
    for idx in selected_indices:
        img_path = os.path.join(folder_path, image_files[idx])
        img = Image.open(img_path).convert("RGB")  # Convert to RGB mode if needed
        img = img.resize(img_size)
        images.append(img)

    return images

def display_and_save_image_lists(images, save_path=None, filename="combined_image.png", padding=20, display_scale=1.0):
    """
    Displays multiple images in a single row with white spacing and saves them as a single combined image.

    Args:
        images: List of PIL.Image to be displayed and saved.
        img_size: Tuple (width, height) for resizing images (default: 256x256).
        save_path: Directory path to save the combined image. If None, image is not saved.
        filename: Name of the saved image file (default: "combined_image.png").
        padding: Space (in pixels) between images (default: 20px).
        display_scale: Scale factor for display size in Jupyter Notebook (default: 1.0 for original size).
    """
    # Compute the final size including padding
    total_width = sum(img.width for img in images) + (padding * (len(images) - 1))
    max_height = max(img.height for img in images)

    # Create a blank white image
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Paste images with padding
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding

    # Save the combined image if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        save_filepath = os.path.join(save_path, filename)
        combined_image.save(save_filepath)
        print(f"Saved combined image: {save_filepath}")

    # Convert to base64 for inline display
    buffer = BytesIO()
    combined_image.save(buffer, format="PNG")  
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Apply scaling to the display size
    display_width = int(total_width * display_scale)
    display_height = int(max_height * display_scale)

    # Display in Jupyter Notebook with adjustable size
    display(HTML(f"<img src='data:image/png;base64,{img_str}' style='margin:5px; width:{display_width}px; height:{display_height}px;'>"))




def tensor2im(var):
    """
    Converts a PyTorch tensor to a PIL Image.
    
    Args:
        var: PyTorch tensor (CHW format, normalized between -1 and 1).
    
    Returns:
        PIL.Image object.
    """
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2) * 255  # Normalize to 0-255
    var = np.clip(var, 0, 255).astype(np.uint8)  # Clip values
    return Image.fromarray(var)


def load_folder_images(
    image_folder: str,
    num_images: int = 4,
    seed: Optional[int] = None
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads up to `num_images` images from `image_folder`.
    If `seed` is provided, samples randomly; otherwise takes the first `num_images`.
    """
    # 1. Gather & sort
    image_paths = sorted(
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg"))
    )
    if not image_paths:
        raise ValueError(f"No .png or .jpg files found in {image_folder}")

    # 2. Select
    if seed is not None:
        random.seed(seed)
        selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
    else:
        selected_paths = image_paths[:num_images]

    # 3. Load
    images = [Image.open(p).convert("RGB") for p in selected_paths]
    return images, selected_paths

import os
import random
from PIL import Image

def load_MRI_images(image_folder, num_images=4, seed=None, slice_range=(70, 100)):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Extract slice range values
    min_slice, max_slice = slice_range

    # Get all image paths from the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

    # Separate the specific image we always want to include
    always_select_path = None  
    filtered_paths = []

    for path in image_paths:
        filename = os.path.basename(path)
        if filename == "BraTS-GLI-00675-000-t1n_slice078.png":
            always_select_path = path  # Store to ensure it's always selected
        else:
            # Extract slice number from filename (assuming format: *_sliceXXX.png)
            slice_num_str = filename.split("_slice")[-1].split(".")[0]
            if slice_num_str.isdigit():
                slice_num = int(slice_num_str)
                if min_slice <= slice_num <= max_slice:
                    filtered_paths.append(path)

    # Ensure at least one valid image is available
    if not filtered_paths:
        raise ValueError("No images found within the specified slice range.")

    # Initialize selection
    selected_paths = []
    if always_select_path:
        selected_paths.append(always_select_path)

    # Keep selecting images until we reach num_images with valid slices
    while len(selected_paths) < num_images:
        new_selection = random.choice(filtered_paths)
        selected_paths.append(new_selection)

    # Load images
    images = [Image.open(image_path).convert("RGB") for image_path in selected_paths]

    return images, selected_paths



def load_paths_images(selected_paths, num_images=4, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # # Get all image paths from the folder
    # image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

    # # Randomly select 'num_images' from the image_paths
    # selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    images = []

    for image_path in selected_paths:
        img = Image.open(image_path).convert("RGB")
        images.append(img)

    return images, selected_paths

def transform_images_to_batch(images_list, transforms):
    
    transformed_images = [transforms(image) for image in images_list] 
    # 'transformed_images' is a transformed version of images_list, each element is convert from Image  to 
    # a torch tensor using torchvision.transforms 

    batched_images = torch.stack(transformed_images, dim=0)
    # batched_images type of image tensors, e.g., 13x3x255x255

    return batched_images


def seed_experiments(seed):
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

    # Ensures deterministic behavior for some PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def PCA_reconstruction(w_pSp, U_load_path, k):
    
    U = np.load(U_load_path)  # Load basis matrix
    
    # Flatten (20000, 18, 512) → (20000, 9216)
    w_pSp_flat = w_pSp.view(w_pSp.shape[0], -1).cpu().numpy()
      
    # Compute W_pca (Projection into lower-dimensional space)
    W_pca = np.dot(w_pSp_flat, U.T)  # Shape: (20000, k)
    W_recon_flat = np.dot(W_pca, U)  
    
    # Convert back to PyTorch tensor and reshape
    W_recon = torch.tensor(W_recon_flat).view(w_pSp.shape)

    # Compute reconstruction error
    error = np.linalg.norm(w_pSp_flat - W_recon_flat) / np.linalg.norm(w_pSp_flat)
    print(f"Reconstruction error for k={k}: {error:.6f}")

    return W_recon 

def PCA_projection(w_pSp, pSp_net, k, index = [0, 1, 2, 3]):
    
    W_latent = PCA_reconstruction(w_pSp, U_load_path = f"./PCA/U_pca_k{k}.npy", k=k)
    
    with torch.no_grad():
        recon_pSp_appr = pSp_net.forward(W_latent[index].to(device), input_code=True, randomize_noise=False, recon_modle=True)
    return recon_pSp_appr