# from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
# from utils.class_registry import ClassRegistry
from datasets.transforms import transforms_registry
from datasets.datasets import ImageDataset, ImageDatasetFlexible, ImagesDataset_mednpy
from datasets.loaders import InfiniteLoader
from configs.data_paths import DATASETS
import torch.nn.functional as F
# from tqdm import tqdm
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch

# Your normalization function
def norm(img):
    return (img.clamp(-1, 1) + 1) / 2

# Converts a normalized tensor image to a NumPy array
def tensor_to_img(t):
    t = norm(t).detach().cpu()
    if t.dim() == 4:  # if batched, take the first one
        t = t[0]
    if t.shape[0] == 1:  # grayscale
        return t.squeeze(0).numpy()
    else:
        return t.permute(1, 2, 0).numpy()


def get_latents_cs(X, Y, sfe_model):
    _, w_x = sfe_model.method.pSp_encoder(X, return_latents=True)
    _, w_x_cond = sfe_model.method.pSp_encoder(Y, return_latents=True)
    c_x, s_x = sfe_model.method.pSp_cs_model(w_x)
    c_y, s_y = sfe_model.method.pSp_cs_model(w_x_cond)    
    
    return c_x, s_x, c_y, s_y


def get_latents_cs1s2(X, Y, sfe_model):
    _, w_x = sfe_model.method.pSp_encoder(X, return_latents=True)
    _, w_x_cond = sfe_model.method.pSp_encoder(Y, return_latents=True)
    c_x, s_x1, s_x2 = sfe_model.method.pSp_cs_model(w_x)
    c_y, s_y1, s_y2 = sfe_model.method.pSp_cs_model(w_x_cond)    
    
    return c_x, s_x1, s_x2, c_y, s_y1, s_y2


def calculate_delta_by_latent(latent_orn, latent_edit, sfe_model):
    _, fx = sfe_model.method.decoder(
        [latent_orn],
        input_is_latent=True,
        randomize_noise=False,
        return_latents=False,
        return_features=True,
        early_stop= 64
    )

    _, fy = sfe_model.method.decoder(
        [latent_edit], 
        is_stylespace=False,
        input_is_latent=True,
        randomize_noise=False,
        return_features=True,
        early_stop= 64
    )

    delta = fx[9] - fy[9]
    
    return delta


def recon_by_latent_w(latent_w, sfe_model):
    encoder_recon, _ = sfe_model.method.decoder(
        [latent_w], 
        is_stylespace=False,
        input_is_latent=True,
        randomize_noise=False,
        return_features=True
    )    

    return encoder_recon

def recon_by_delta(x_resh, delta, sfe_model):

    w_recon, predicted_feat = sfe_model.method.inverter.fs_backbone(x_resh)
    w_recon = w_recon + sfe_model.method.latent_avg
            
    _, w_feats = sfe_model.method.decoder(
        [w_recon],
        input_is_latent=True,
        return_features=True,
        is_stylespace=False,
        randomize_noise=False,
        early_stop=64
    )

    w_feat = w_feats[9]  # bs x 512 x 64 x 64 
    
    fused_feat = sfe_model.method.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
    if delta is None:
        delta = torch.zeros_like(fused_feat)  # inversion case
    
    edited_feat = sfe_model.method.encoder(torch.cat([fused_feat, delta], dim=1))
    feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

    images, _ = sfe_model.method.decoder(
        [w_recon],
        input_is_latent=True,
        return_features=True,
        new_features=feats,
        feature_scale=1.0,
        is_stylespace=False,
        randomize_noise=False
    )

    return images


def configure_Seg_datasets(config, seed=None, shuffle=False, ds_name = None, bg_path = None, t_path = None):
    """
    Loads dataset and returns the first batch of background and target.
    Uses a fixed seed to make the first batch reproducible.
    """
    import torch
    import random
    import numpy as np

    print("Loading dataset")

    # Fix seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    # Load transforms and paths
    transforms = transforms_registry[config.data.transform]().get_transforms()
    if ds_name is None:
        ds_name    = config.data.dataset
        
    print(f"Using dataset: {ds_name}")
    
    cfg        = DATASETS.get(ds_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {ds_name!r}")

    dataset_bg = ImagesDataset_mednpy(bg_path,  transforms["test"])
    dataset_t  = ImagesDataset_mednpy(t_path,  transforms["test"])

    # Wrap with InfiniteLoader or standard DataLoader
    test_loader_bg = InfiniteLoader(
        dataset_bg,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=config.model.workers,
        is_infinite=False,
        generator=generator  # ✅ needs to be passed through if supported
    )
    test_loader_t = InfiniteLoader(
        dataset_t,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=config.model.workers,
        is_infinite=False,
        generator=generator
    )

    # # Fetch first batch
    # batch_bg = next(iter(test_loader_bg))
    # batch_t  = next(iter(test_loader_t))

    return test_loader_bg, test_loader_t



def configure_datasets(config, test_images=False, seed=None, shuffle=False, ds_name = None):
    """
    Loads dataset and returns the first batch of background and target.
    Uses a fixed seed to make the first batch reproducible.
    """
    import torch
    import random
    import numpy as np

    print("Loading dataset")

    # Fix seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    # Load transforms and paths
    transforms = transforms_registry[config.data.transform]().get_transforms()
    if ds_name is None:
        ds_name    = config.data.dataset
        
    print(f"Using dataset: {ds_name}")
    
    cfg        = DATASETS.get(ds_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {ds_name!r}")

    # Build datasets
    if test_images:
        dataset_bg = ImageDatasetFlexible(cfg["val_bg"],  transforms["test"])
        dataset_t  = ImageDatasetFlexible(cfg["val_t"],  transforms["test"])
    else:
        dataset_bg = ImageDatasetFlexible(cfg["train_bg"],  transforms["test"])
        dataset_t  = ImageDatasetFlexible(cfg["train_t"],  transforms["test"])  

    # Wrap with InfiniteLoader or standard DataLoader
    test_loader_bg = InfiniteLoader(
        dataset_bg,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=config.model.workers,
        is_infinite=False,
        generator=generator  # ✅ needs to be passed through if supported
    )
    test_loader_t = InfiniteLoader(
        dataset_t,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=config.model.workers,
        is_infinite=False,
        generator=generator
    )

    # # Fetch first batch
    # batch_bg = next(iter(test_loader_bg))
    # batch_t  = next(iter(test_loader_t))

    return test_loader_bg, test_loader_t


def configure_datasets_v2(data_transform='face_256', batch_size=4, workers=4, test_images=True, seed=None, shuffle=False, ds_name = None):
    """
    Loads dataset and returns the first batch of background and target.
    Uses a fixed seed to make the first batch reproducible.
    """
    import torch
    import random
    import numpy as np

    print("Loading dataset")

    # Fix seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    # Load transforms and paths
    transforms = transforms_registry[data_transform]().get_transforms()
        
    print(f"Using dataset: {ds_name}")
    
    cfg        = DATASETS.get(ds_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {ds_name!r}")

    # Build datasets
    if test_images:
        dataset_bg = ImageDatasetFlexible(cfg["val_bg"],  transforms["test"])
        dataset_t  = ImageDatasetFlexible(cfg["val_t"],  transforms["test"])
    else:
        dataset_bg = ImageDatasetFlexible(cfg["train_bg"],  transforms["test"])
        dataset_t  = ImageDatasetFlexible(cfg["train_t"],  transforms["test"])  

    # Wrap with InfiniteLoader or standard DataLoader
    test_loader_bg = InfiniteLoader(
        dataset_bg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        is_infinite=False,
        generator=generator  # ✅ needs to be passed through if supported
    )
    test_loader_t = InfiniteLoader(
        dataset_t,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        is_infinite=False,
        generator=generator
    )

    # # Fetch first batch
    # batch_bg = next(iter(test_loader_bg))
    # batch_t  = next(iter(test_loader_t))

    return test_loader_bg, test_loader_t


def preprocess_image(img: torch.Tensor, size=(256, 256)) -> torch.Tensor:
    """
    Preprocess an image tensor:
      1. Map from [-1, 1] to [0, 1]
      2. Resize to target size with bilinear interpolation

    Args:
        img (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W)
        size (tuple): Target spatial size (H, W), default (256, 256)

    Returns:
        torch.Tensor: Preprocessed tensor in [0, 1], shape (B, C, size[0], size[1])
    """
    # Ensure 4D tensor
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # Map from [-1,1] → [0,1]
    img = (img * 0.5) + 0.5

    # Resize
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)

    return img

def get_random_seeds():
    seed = random.randint(0, 20000)
    print('Random seed:', seed)
    return seed
# #seed = 103  # fixed
# seed_old =  random.randint(0, 20000)  # optional random   Go to young: 12665  13002 15542 10833 18681    Go to old:  14620  10190 6009 2588
# seed_young = random.randint(0, 20000)  # optional random


# from utils.common_utils import tensor2im, get_keys, visualize_batch_grid
# from inference_ipynb.inference_funcs import get_latents_cs, get_latents_cs1s2, calculate_delta_by_latent, recon_by_latent_w, recon_by_delta, configure_datasets, preprocess_image, get_random_seeds
# import tqdm

# def maybe_crop(img, crop_cfg):
#     if crop_cfg is None:
#         return img
#     return crop_tensor(img, **crop_cfg)

# def crop_tensor(
#     img,        # Tensor: [C, H, W] or [H, W]
#     w_l=0,
#     w_r=0,
#     h_top=0,
#     h_bot=0,
# ):
#     """
#     Crop tensor image.

#     Args:
#         img: torch.Tensor
#         w_l, w_r: pixels to crop from left / right
#         h_top, h_bot: pixels to crop from top / bottom
#     """
#     if img.dim() == 3:
#         _, H, W = img.shape
#         return img[:, h_top:H - h_bot, w_l:W - w_r]
#     elif img.dim() == 2:
#         H, W = img.shape
#         return img[h_top:H - h_bot, w_l:W - w_r]
#     else:
#         raise ValueError(f"Unsupported tensor shape: {img.shape}")


# def inference_cs1s2(
#     sfe_model,
#     test_bg_dataloader,
#     test_t_dataloader,
#     idx,
#     vis_mode="row",
#     num_pairs=4,
#     target_batch_idx=None, 
#     crop_cfg=None,
#     device='cuda'
# ):
#     """
#     vis_mode:
#         - 'row': paper-style visualization (X on top, Y below)
#         - 'col': column-wise visualization (X/Y side by side, multiple rows)

#     target_batch_idx:
#         - None: use first batch (default/debug)
#         - int:  use the specified batch index (reproducible)
#     """

#     with torch.no_grad():
#         for batch_idx, (batch_bg, batch_t) in tqdm(
#             enumerate(zip(test_bg_dataloader, test_t_dataloader)),
#             total=len(test_bg_dataloader)
#         ):
#             X = batch_bg.to(device).float()
#             Y = batch_t.to(device).float()

#             X_resh = F.interpolate(X, size=(256, 256), mode="bilinear", align_corners=False)
#             Y_resh = F.interpolate(Y, size=(256, 256), mode="bilinear", align_corners=False)

#             # ---- latent decomposition ----
#             c_x, s_x1, s_x2, c_y, s_y1, s_y2 = get_latents_cs1s2(X_resh, Y_resh, sfe_model)
#             # s_x = torch.zeros_like(s_x)
#             recon_w_X = recon_by_latent_w(c_x + s_x1, sfe_model)
#             recon_w_Y = recon_by_latent_w(c_y + s_y2, sfe_model)

#             swap_w_X2Y = recon_by_latent_w(c_x + s_y2, sfe_model)
#             swap_w_Y2X = recon_by_latent_w(c_y + s_x1, sfe_model)

#             recon_f_X = recon_by_delta(X_resh, delta=None, sfe_model=sfe_model)
#             recon_f_Y = recon_by_delta(Y_resh, delta=None, sfe_model=sfe_model)

#             delta_x2y = calculate_delta_by_latent(c_x + s_x1 + s_x2, c_x + s_y1 + s_y2, sfe_model)
#             delta_y2x = calculate_delta_by_latent(c_y + s_y1 + s_y2, c_y + s_x1 + s_x2, sfe_model)
#             swap_f_X2Y = recon_by_delta(X_resh, delta_x2y, sfe_model)
#             swap_f_Y2X = recon_by_delta(Y_resh, delta_y2x, sfe_model)

#             break  # only visualize first batch


#     # ==========================================================
#     # Visualization
#     # ==========================================================
#     if vis_mode == "row":
#         row1 = torch.stack(
#             [
#                 maybe_crop(X[idx], crop_cfg),
#                 maybe_crop(recon_w_X[idx], crop_cfg),
#                 maybe_crop(swap_w_X2Y[idx], crop_cfg),
#                 maybe_crop(recon_f_X[idx], crop_cfg),
#                 maybe_crop(swap_f_X2Y[idx], crop_cfg),
#             ],
#             dim=0
#         )

#         row2 = torch.stack(
#             [
#                 maybe_crop(Y[idx], crop_cfg),
#                 maybe_crop(recon_w_Y[idx], crop_cfg),
#                 maybe_crop(swap_w_Y2X[idx], crop_cfg),
#                 maybe_crop(recon_f_Y[idx], crop_cfg),
#                 maybe_crop(swap_f_Y2X[idx], crop_cfg),
#             ],
#             dim=0
#         )


#         columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

#         visualize_batch_grid(
#             image_batches=columns,
#             titles=[f"Input", "Recon W", "Swap W", "Recon F", "Swap F"],
#             save_path=None,
#         )

#     elif vis_mode == "col":
#         for i in range(num_pairs):
#             col_X = torch.stack(
#                 [
#                     maybe_crop(X[i], crop_cfg),
#                     maybe_crop(recon_f_X[i], crop_cfg),
#                     maybe_crop(swap_f_X2Y[i], crop_cfg),
#                 ],
#                 dim=0
#             )

#             col_Y = torch.stack(
#                 [
#                     maybe_crop(Y[i], crop_cfg),
#                     maybe_crop(recon_f_Y[i], crop_cfg),
#                     maybe_crop(swap_f_Y2X[i], crop_cfg),
#                 ],
#                 dim=0
#             )

#             visualize_batch_grid(
#                 image_batches=[col_X, col_Y],
#                 titles=[f"X #{i}", f"Y #{i}"],
#                 save_path=None,
#             )

#     else:
#         raise ValueError(f"Unknown vis_mode: {vis_mode}")
