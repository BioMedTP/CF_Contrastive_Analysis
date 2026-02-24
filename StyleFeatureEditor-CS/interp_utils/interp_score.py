from typing import Callable, Optional, Tuple, List
import torch

from .classifiers import ClassifierBundle

@torch.inference_mode()
def score_alpha_series(
    base_imgs: torch.Tensor,                 # [B,C,H,W]
    delta: torch.Tensor,                     # 任意 shape，只要 recon_fn 兼容
    alphas: torch.Tensor,                    # [K]
    recon_by_delta_fn: Callable,             # recon_by_delta(base, alpha*delta, sfe_model=...) -> [B,C,H,W]
    classifier: ClassifierBundle,
    t_val: int = 0,
    png_like: bool = False,
    keep_index_in_batch: Optional[int] = None,   # 想可视化的那张在 batch 的 idx
    recon_kwargs: Optional[dict] = None,         # 传给 recon_by_delta_fn 的 kwargs（比如 sfe_model=...）
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    return:
      probs: [B, K+1]  (第0列是 real)
      vis_recons: list of [1,C,H,W] (real + K 个 alpha)，如果 keep_index_in_batch 不为 None
    """
    recon_kwargs = recon_kwargs or {}

    probs_list = [classifier.prob(base_imgs, t_val=t_val, png_like=png_like)]  # [B]
    vis_recons = None

    if keep_index_in_batch is not None:
        i = int(keep_index_in_batch)
        vis_recons = [base_imgs[i:i+1].detach().cpu()]

    for a in alphas:
        recon = recon_by_delta_fn(base_imgs, a * delta, **recon_kwargs)
        probs_list.append(classifier.prob(recon, t_val=t_val, png_like=png_like))
        if vis_recons is not None:
            vis_recons.append(recon[i:i+1].detach().cpu())

    probs = torch.stack(probs_list, dim=1)  # [B, K+1]
    return probs, vis_recons