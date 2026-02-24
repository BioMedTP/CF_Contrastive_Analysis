import torch
import os
import sys
import yaml
sys.path =  ['.'] + sys.path

root_dir = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN"
sfe_root = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS"
medminist_path = '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets'

Paths = {
        "base_models": {
            "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_rosinality.pt",
            "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_resume/00000-brats256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-002419.pkl",
            "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_ht_Ros_resume/checkpoints/iteration_20000.pt",
            "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_ht/fse_inverter_train_000/iteration_140000.pt",
            "sfe_path": sfe_root + "/experiments/fse_cs_editor_train/brats_ht/brats_ht_000/iteration_370000.pt",
        },

        "wo_regular": {
        "pSp_cs_path": root_dir +  "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/wo_regular/checkpoints/iteration_100000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/wo_regular_000/iteration_430000.pt",
        },
        "with_D_only": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_D_start_step1000/checkpoints/iteration_100000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/adv_D_only_000/iteration_350000.pt",
        },
        "with_DR_DiscMI": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_DiscMI/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_110000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/adv_DR_DiscMI_000/iteration_430000.pt",
        },
        
        "with_DR_RegrMI_90k": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_RegrR/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_90000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/RegrR_D_90k_000/iteration_430000.pt",
        },
        
        "with_DR_RegrMI_110k": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_RegrR/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_110000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/RegrR_D_110k_000/iteration_430000.pt",
        }, 
        "with_DR_RegrMI_120k": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_RegrR/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_120000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/RegrR_D_120k_000/iteration_430000.pt",
        },
        "with_DR_RegrMI_140k": {
        "pSp_cs_path": root_dir + "/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_RegrR/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_140000.pt",
        "sfe_path": root_dir + "/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/Regularization/RegrR_D_140k_000/iteration_430000.pt",
        },        
        }



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = 'with_DR_RegrMI_90k'


sfe_ckpt = Paths[model_name]["sfe_path"]

# Load config.yaml
config_yaml_path = os.path.join(os.path.dirname(sfe_ckpt), "config.yaml")
from omegaconf import OmegaConf
config = OmegaConf.load(config_yaml_path)
config.model.w_space_encoder = "pSp"

config.data.dataset="brats_ht_new"
config.model.checkpoint_path = sfe_ckpt
config.model.pSp_cs_path = Paths[model_name]["pSp_cs_path"]

config.model.batch_size = 4  # Set batch size for inference

# Pretty YAML (my favorite)
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(config))


from envs import make_gpu_mem_reporter
print_memory = make_gpu_mem_reporter(gpu_index=0, clear_cache_on_setup=True)

print_memory("Before loading model")

from runners.simple_runner import SimpleRunner
runner = SimpleRunner(config=config)
sfe_model = runner.inference_runner
device = sfe_model.device

sfe_model.method.train()

print_memory("After loading model")


from torchvision.utils import save_image
from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
from inference_ipynb.inference_funcs import get_latents_cs, calculate_delta_by_latent, recon_by_latent_w, recon_by_delta, configure_datasets, configure_Seg_datasets, preprocess_image, get_random_seeds
import shutil
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random

def inference(
    sfe_model,
    test_bg_dataloader,
    test_t_dataloader,
    idx,
    vis_mode="row",
    num_pairs=4,
    target_batch_idx=None,   # ⭐ 新增
):
    """
    vis_mode:
        - 'row': paper-style visualization (X on top, Y below)
        - 'col': column-wise visualization (X/Y side by side, multiple rows)

    target_batch_idx:
        - None: use first batch (default/debug)
        - int:  use the specified batch index (reproducible)
    """

    with torch.no_grad():
        for batch_idx, (batch_bg, batch_t) in tqdm(
            enumerate(zip(test_bg_dataloader, test_t_dataloader)),
            total=len(test_bg_dataloader)
        ):
            # ----------------------------------
            # skip until target batch
            # ----------------------------------
            if target_batch_idx is not None and batch_idx != target_batch_idx:
                continue

            X = batch_bg.to(device).float()
            Y = batch_t.to(device).float()

            X_resh = F.interpolate(X, size=(256, 256), mode="bilinear", align_corners=False)
            Y_resh = F.interpolate(Y, size=(256, 256), mode="bilinear", align_corners=False)

            # ---- latent decomposition ----
            c_x, s_x, c_y, s_y = get_latents_cs(X_resh, Y_resh, sfe_model)

            # ---- recon in W space ----
            s_x_zero = torch.zeros_like(s_x)
            recon_w_X = recon_by_latent_w(c_x + s_x_zero, sfe_model)
            recon_w_Y = recon_by_latent_w(c_y + s_y, sfe_model)

            swap_w_X2Y = recon_by_latent_w(c_x + s_y, sfe_model)
            swap_w_Y2X = recon_by_latent_w(c_y + s_x_zero, sfe_model)

            # ---- recon in image (delta) space ----
            recon_f_X = recon_by_delta(X_resh, delta=None, sfe_model=sfe_model)
            recon_f_Y = recon_by_delta(Y_resh, delta=None, sfe_model=sfe_model)

            delta_x2y = calculate_delta_by_latent(c_x + s_x_zero, c_x + s_y, sfe_model)
            delta_y2x = calculate_delta_by_latent(c_y + s_y, c_y + s_x_zero, sfe_model)

            swap_f_X2Y = recon_by_delta(X_resh, delta_x2y, sfe_model)
            swap_f_Y2X = recon_by_delta(Y_resh, delta_y2x, sfe_model)

            # 👉 我们只需要一个 batch
            break

    # ==========================================================
    # Visualization
    # ==========================================================
    if vis_mode == "row":
        row1 = torch.stack(
            [X[idx], recon_w_X[idx], swap_w_X2Y[idx], recon_f_X[idx], swap_f_X2Y[idx]],
            dim=0
        )
        row2 = torch.stack(
            [Y[idx], recon_w_Y[idx], swap_w_Y2X[idx], recon_f_Y[idx], swap_f_Y2X[idx]],
            dim=0
        )

        columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

        visualize_batch_grid(
            image_batches=columns,
            titles=["Input", "Recon W", "Swap W", "Recon F", "Swap F"],
            save_path=None
        )

    elif vis_mode == "col":
        for i in range(num_pairs):
            col_X = torch.stack(
                [X[i], recon_f_X[i], swap_f_X2Y[i]],
                dim=0
            )
            col_Y = torch.stack(
                [Y[i], recon_f_Y[i], swap_f_Y2X[i]],
                dim=0
            )

            visualize_batch_grid(
                image_batches=[col_X, col_Y],
                titles=[f"X #{i}", f"Y #{i}"],
                save_path=None
            )

    else:
        raise ValueError(f"Unknown vis_mode: {vis_mode}")

def get_paired_random_batch(loader_X, loader_Y):
    """
    Returns one random paired batch (same batch index)
    from loader_X and loader_Y.
    """
    num_batches = len(loader_X)
    idx = random.randint(0, num_batches - 1)

    # iterate both loaders in sync
    for i, (batch_X, batch_Y) in enumerate(zip(loader_X, loader_Y)):
        if i == idx:
            return batch_X, batch_Y

    raise RuntimeError("Batch index out of range")

import numpy as np
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
    
    
seed = get_random_seeds()

# from time import time
# from datetime import datetime

# )

bg_path = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Segmentation/data/bratsHT_TUMOR_new/test_X.npy"

t_name = "BraTS-GLI-01368-000_slice_109" # BraTS-GLI-00604-000_slice_100    BraTS-GLI-01072-000_slice_080   BraTS-GLI-01101-000_slice_107.npy  BraTS-GLI-01120-000_slice_072.npy  BraTS-GLI-01368-000_slice_109.npy


t_path = f"/home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion_quantative_results/Segment_statics/segmentY_list/{t_name}.npy"


test_bg_dataloader, test_t_dataloader = configure_Seg_datasets(sfe_model.config, seed=seed, shuffle=False, bg_path=bg_path, t_path=t_path)

# exp_name = f"{model_name}/images/{ds_name}"
save_image_dir = f"/home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion_quantative_results/Segment_statics/segmentY_list/{t_name}"

max_eval_batch = None

if os.path.exists(save_image_dir):
    shutil.rmtree(save_image_dir)

# image_paths = ['real_X', 'real_Y', 'recon_w_X', 'recon_w_Y', 'swap_w_Y2X', 'swap_w_X2Y', 'recon_f_X', 'recon_f_Y', 'swap_f_X2Y', 'swap_f_Y2X']
image_paths = ['swap_w_Y2X', 'swap_w_X2Y', 'swap_f_X2Y', 'swap_f_Y2X']
for path in image_paths:
    dir_path = os.path.join(save_image_dir, path)
    os.makedirs(dir_path)


import os
import shutil
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

def infinite_loader(dataloader):
    """不缓存数据的无限循环 dataloader"""
    while True:
        for batch in dataloader:
            yield batch

# ------------------------
# build infinite target iterator
# ------------------------
t_iter = infinite_loader(test_t_dataloader)

global_idx = 0  # 用全局计数，避免 batch_size 变化导致 idx 重复/跳号

with torch.no_grad():
    for batch_idx, batch_bg in tqdm(enumerate(test_bg_dataloader), total=len(test_bg_dataloader)):
        if max_eval_batch is not None and batch_idx >= max_eval_batch:
            break

        batch_t = next(t_iter)  # ⭐ 永远能取到（t 会循环）

        X = batch_bg.to(device).float()
        Y = batch_t.to(device).float()

        # ------------------------
        # 关键：以 X 的 batch 为准，让 Y 对齐到同样的 batch size
        # ------------------------
        Bx = X.size(0)
        By = Y.size(0)
        if By < Bx:
            reps = (Bx + By - 1) // By
            Y = Y.repeat(reps, 1, 1, 1)[:Bx]
        else:
            Y = Y[:Bx]

        # resize
        X_resh = F.interpolate(X, size=(256, 256), mode="bilinear", align_corners=False)
        Y_resh = F.interpolate(Y, size=(256, 256), mode="bilinear", align_corners=False)

        # cs
        c_x, s_x, c_y, s_y = get_latents_cs(X_resh, Y_resh, sfe_model)

        # w recon/swap
        s_x = torch.zeros_like(s_x)
        swap_w_X2Y = recon_by_latent_w(c_x + s_y, sfe_model)
        swap_w_Y2X = recon_by_latent_w(c_y + s_x, sfe_model)

        # f recon + delta + swap_f
        delta_x2y = calculate_delta_by_latent(c_x + s_x, c_x + s_y, sfe_model)
        delta_y2x = calculate_delta_by_latent(c_y + s_y, c_y + s_x, sfe_model)

        swap_f_X2Y = recon_by_delta(X_resh, delta_x2y, sfe_model=sfe_model)
        swap_f_Y2X = recon_by_delta(Y_resh, delta_y2x, sfe_model=sfe_model)

        images = {
            "swap_w_X2Y": swap_w_X2Y,
            "swap_w_Y2X": swap_w_Y2X,
            "swap_f_X2Y": swap_f_X2Y,
            "swap_f_Y2X": swap_f_Y2X,
        }

        images = {k: (None if v is None else preprocess_image(v)) for k, v in images.items()}

        # save (用 global_idx 确保每个 X 对应唯一编号)
        B = X.size(0)
        for i in range(B):
            idx = global_idx
            global_idx += 1
            for key, tensor in images.items():
                if tensor is None or tensor.ndim != 4 or i >= tensor.size(0):
                    continue
                img_i = tensor[i].detach().cpu().clamp(0, 1)
                save_image(img_i, os.path.join(save_image_dir, key, f"{idx}.png"))

print(f"Results images saved to {save_image_dir}")
