import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import torch
import torch.nn.functional as F

# ==========
# Common preprocess
# ==========

def rgb_to_gray_luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def ensure_1ch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] == 3:
        return rgb_to_gray_luma(x)
    return x.mean(dim=1, keepdim=True)

def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        return x[:, :3]
    reps = (3 + x.shape[1] - 1) // x.shape[1]
    return x.repeat(1, reps, 1, 1)[:, :3]

def quantize_like_png(x: torch.Tensor) -> torch.Tensor:
    """
    x: in [-1,1]
    模拟保存成 8-bit PNG 再读回的量化误差 (对齐你以前 folder inference 那种)
    """
    x = x.clamp(-1, 1)
    u8 = (((x + 1) / 2) * 255.0).round().clamp(0, 255).to(torch.uint8)
    xq = (u8.float() / 255.0) * 2.0 - 1.0
    return xq

def map_01_to_m11_if_needed(x: torch.Tensor) -> torch.Tensor:
    # 如果是 [0,1]，映射到 [-1,1]，避免喂错范围
    if x.min() >= 0.0 and x.max() <= 1.0 + 1e-3:
        return x * 2.0 - 1.0
    return x

# ==========
# Wrapper
# ==========

Backend = Literal["fastdime", "timm"]

@dataclass
class ClassifierBundle:
    backend: Backend
    model: torch.nn.Module
    device: torch.device

    @torch.inference_mode()
    def prob(self, imgs: torch.Tensor, t_val: int = 0, png_like: bool = False) -> torch.Tensor:
        """
        imgs: [B,C,H,W]
        return: prob(class=1) shape [B]
        """
        if self.backend == "fastdime":
            x = ensure_1ch(imgs.float())
            x = map_01_to_m11_if_needed(x).to(self.device)
            t = torch.full((x.size(0),), int(t_val), device=self.device, dtype=torch.long)
            logits = self.model(x, t).view(-1)
            return torch.sigmoid(logits)

        if self.backend == "timm":
            x = ensure_3ch(imgs.float())
            x = map_01_to_m11_if_needed(x)
            if png_like:
                x = quantize_like_png(x)
            x = x.to(self.device)
            logits = self.model(x)              # [B,2]
            prob = F.softmax(logits, dim=1)     # [B,2]
            return prob[:, 1]

        raise ValueError(f"Unknown backend={self.backend}")

# ==========
# Loaders
# ==========

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def load_fastdime_classifier(
    cls_ckpt: str,
    cls_type: str,
    device: Optional[str] = None,
    fastdime_repo_root: str = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/FastDiME_Med",
) -> ClassifierBundle:
    """
    使用你现有的 CFgenerating.model_loading.load_classifier
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    repo_root = str(Path(fastdime_repo_root).resolve())
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    # 清缓存避免 import 错模块
    for k in list(sys.modules.keys()):
        if k == "models" or k.startswith("models.") or k == "CFgenerating" or k.startswith("CFgenerating."):
            sys.modules.pop(k, None)

    from CFgenerating.model_loading import load_classifier
    model = load_classifier(cls_ckpt, cls_type, device)
    model.eval()

    return ClassifierBundle(backend="fastdime", model=model, device=dev)

def load_timm_binary_classifier(
    ckpt_path: str,
    model_type: str,
    img_size: int = 256,
    device: Optional[str] = None,
) -> ClassifierBundle:
    """
    对齐你 eval_folders.py：timm 二分类 num_classes=2 in_chans=3
    """
    import timm

    TIMM_MODEL_ZOO = {
        "swinv2_small": "swinv2_small_window12to16_192to256",
        "swinv2_large": "swinv2_large_window12to16_192to256",
        "densenet121":  "densenet121",
        "resnet34":     "resnet34",
    }
    assert model_type in TIMM_MODEL_ZOO, f"Unknown model_type={model_type}, choose {list(TIMM_MODEL_ZOO.keys())}"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    timm_name = TIMM_MODEL_ZOO[model_type]
    kwargs = dict(pretrained=False, num_classes=2, in_chans=3)
    if model_type.startswith("swin"):
        kwargs["img_size"] = img_size

    model = timm.create_model(timm_name, **kwargs).to(dev).eval()

    ckpt = torch.load(ckpt_path, map_location=dev)
    sd = _extract_state_dict(ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}  # 兼容 DDP
    model.load_state_dict(sd, strict=True)

    return ClassifierBundle(backend="timm", model=model, device=dev)