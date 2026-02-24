import os
import torch
import numpy as np
import lpips
import argparse
import json
import re
import csv
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from skimage.metrics import structural_similarity as ssim
import warnings

# Suppress torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LPIPS Loss Model
lpips_model = lpips.LPIPS(net="alex").to(device)

# Load FaceNet (InceptionResnetV1) Model for Identity Similarity
face_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

face_transform = transforms.Compose([
    transforms.Resize((160, 160)),  # FaceNet expects 160x160
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize as expected by FaceNet
])

def natural_sort_key(s):
    """Sort file names numerically if they contain numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_image(image_path):
    """Load and preprocess image for evaluation."""
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def load_face_image(image_path):
    """Load and preprocess image for FaceNet."""
    img = Image.open(image_path).convert("RGB")
    return face_transform(img).unsqueeze(0).to(device)

def mse_loss(img1, img2):
    """Compute Mean Squared Error (MSE)."""
    return torch.mean((img1 - img2) ** 2).item()

def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # Assuming images are normalized to [0,1]
    return 10 * np.log10((max_pixel_value ** 2) / mse)

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) with adaptive window size."""
    img1 = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
    min_size = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_size) if min_size % 2 == 1 else min(7, min_size - 1)  # Ensure odd value
    return ssim(img1, img2, win_size=win_size, channel_axis=-1, data_range=img1.max() - img1.min())

def compute_metrics(real_dir, eval_dir, output_csv, num_test=None):
    """Compute LPIPS, MSE, PSNR, SSIM, and Identity Similarity."""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    
    real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(valid_extensions)]
    eval_images = [f for f in os.listdir(eval_dir) if f.lower().endswith(valid_extensions)]
    
    real_images.sort(key=natural_sort_key)
    eval_images.sort(key=natural_sort_key)

    if len(real_images) != len(eval_images):
        raise ValueError("Mismatch in number of real and eval images!")

    if num_test is not None:
        real_images = real_images[:num_test]
        eval_images = eval_images[:num_test]

    num_images = len(real_images)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["input_dir_real", "input_dir_eval", "LPIPS", "MSE", "PSNR", "SSIM", "ID"])

        total_lpips, total_mse, total_psnr, total_ssim, total_identity = 0, 0, 0, 0, 0

        for real_img_name, eval_img_name in zip(real_images, eval_images):
            real_img_path = os.path.join(real_dir, real_img_name)
            eval_img_path = os.path.join(eval_dir, eval_img_name)

            img_real = load_image(real_img_path)
            img_eval = load_image(eval_img_path)

            lpips_value = lpips_model(img_real, img_eval).mean().item()
            total_lpips += lpips_value

            mse_value = mse_loss(img_real, img_eval)
            total_mse += mse_value

            psnr_value = psnr(img_real, img_eval)
            total_psnr += psnr_value

            ssim_value = compute_ssim(img_real, img_eval)
            total_ssim += ssim_value

            real_face = load_face_image(real_img_path)
            eval_face = load_face_image(eval_img_path)

            with torch.no_grad():
                emb1 = face_model(real_face).cpu().numpy().flatten()
                emb2 = face_model(eval_face).cpu().numpy().flatten()
                identity_sim = cosine_similarity(emb1, emb2)

            total_identity += identity_sim

            writer.writerow([real_img_path, eval_img_path, lpips_value, mse_value, psnr_value, ssim_value, identity_sim])

    avg_lpips = total_lpips / num_images
    avg_mse = total_mse / num_images
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_identity = total_identity / num_images

    return {
        "LPIPS": avg_lpips,
        "MSE": avg_mse,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "Identity_Similarity": avg_identity,
    }

def save_results(output_json, results):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Reconstruction Metrics")
    parser.add_argument("--input_dir_real", type=str, required=True, help="Path to real images")
    parser.add_argument("--input_dir_eval", type=str, required=True, help="Path to evaluated images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--num_test", type=int, default=None, help="Number of images to test (default: all)")
    args = parser.parse_args()

    # Define file paths
    output_json = os.path.join(args.output_dir, "evaluation_results.json")
    output_csv = os.path.join(args.output_dir, "image_comparison_results.csv")

    # Compute image-level metrics (LPIPS, MSE, Identity Similarity)
    image_metrics = compute_metrics(args.input_dir_real, args.input_dir_eval, output_csv, args.num_test)

    # Save results as JSON
    save_results(output_json, image_metrics)
