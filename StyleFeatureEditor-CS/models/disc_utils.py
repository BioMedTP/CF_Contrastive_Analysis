# models/disc_utils.py
from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
from collections import OrderedDict

import pickle
import torch


def get_keys(ckpt: Dict[str, Any], name: str) -> Dict[str, torch.Tensor]:
    """
    Extract sub-state-dict for a given prefix from ckpt["state_dict"].
    Example: name="discriminator" pulls keys like "discriminator.xxx" -> "xxx"
    """
    if "state_dict" not in ckpt:
        raise KeyError("Checkpoint has no 'state_dict' key.")
    sd = {}
    prefix = name + "."
    for k, v in ckpt["state_dict"].items():
        if k.startswith(prefix):
            sd[k[len(prefix):]] = v
    return sd


def adapt_disc_state_dict_1ch_to_3ch(
    sd: Dict[str, torch.Tensor],
    target_in_ch: int = 3,
    fromrgb_key: str = "b256.fromrgb.weight",
) -> Dict[str, torch.Tensor]:
    """
    If discriminator 'fromrgb' conv weights are 1ch but target expects 3ch, repeat weights.
    If already target_in_ch, no-op.
    """
    if fromrgb_key not in sd:
        return sd

    w = sd[fromrgb_key]
    if not isinstance(w, torch.Tensor) or w.ndim != 4:
        return sd

    cin = w.shape[1]
    if cin == target_in_ch:
        return sd

    if cin == 1 and target_in_ch == 3:
        sd = dict(sd)  # copy
        sd[fromrgb_key] = w.repeat(1, 3, 1, 1) / 3.0
        print(f"✅ adapted {fromrgb_key}: {tuple(w.shape)} -> {tuple(sd[fromrgb_key].shape)} (1ch->3ch)")
        return sd

    return sd


def disc_family_from_sd(sd: Dict[str, torch.Tensor]) -> str:
    """
    Detect discriminator family from state_dict keys.
    """
    keys = sd.keys()

    # StyleGAN2-ADA / NVLabs: b256.fromrgb.*, b128.*, ... b4.*
    if "b256.fromrgb.weight" in keys or any(k.startswith("b256.") for k in keys):
        return "ada"

    # rosinality / stylegan2-pytorch / pSp-style: convs.*, final_conv.*, final_linear.*
    if any(k.startswith("convs.") for k in keys) or any(k.startswith("final_conv") for k in keys) or any(k.startswith("final_linear") for k in keys):
        return "rosinality"

    return "unknown"


def ensure_rgb(x: torch.Tensor) -> torch.Tensor:
    """Safety: convert [B,1,H,W] -> [B,3,H,W] by repeating channels."""
    if x.ndim == 4 and x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def _to_args_dict(args_obj: Any) -> Dict[str, Any]:
    if args_obj is None:
        return {}
    if isinstance(args_obj, dict):
        return args_obj
    try:
        return vars(args_obj)  # argparse.Namespace
    except Exception:
        return {}


def load_state_dict_skip_mismatch(
    module: torch.nn.Module,
    sd: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    prefix: str = "disc",
    verbose: int = 8,
) -> torch.nn.Module:
    """
    Load only keys that exist in module.state_dict() AND shapes match.
    This avoids RuntimeError on size mismatch (even when strict=False).
    """
    msd = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped = []

    for k, v in sd.items():
        if k not in msd:
            skipped.append((k, "not_in_model"))
            continue
        if not hasattr(v, "shape") or msd[k].shape != v.shape:
            skipped.append((k, f"shape {tuple(getattr(v, 'shape', ())) } -> {tuple(msd[k].shape)}"))
            continue
        filtered[k] = v

    module.load_state_dict(filtered, strict=strict)

    if skipped:
        print(f"⚠️ {prefix}: loaded {len(filtered)}/{len(sd)} keys, skipped {len(skipped)} mismatched keys.")
        for s in skipped[:verbose]:
            print("   skip:", s[0], "|", s[1])
        if len(skipped) > verbose:
            print(f"   ... {len(skipped) - verbose} more skipped")
    else:
        print(f"✅ {prefix}: loaded all {len(filtered)} keys (no mismatch).")

    return module


def load_disc_any(
    path: str,
    *,
    device: torch.device,
    opts: Any,
    legacy_load,                 # callable: legacy_load(fileobj) -> dict
    DiscriminatorADA,            # ADA discriminator class (positional c_dim,img_resolution,img_channels) OR kwargs-based init_kwargs path
    DiscriminatorR=None,         # rosinality discriminator class (convs.*, final_conv.*)
    strict: bool = False,        # False = robust; True = enforce exact match
) -> torch.nn.Module:
    """
    Universal discriminator loader.

    Supports:
      - .pkl via legacy_load (persistent ids) with fallback pickle.load
      - .pt/.pth via torch.load (with fallback to legacy_load if torch.load fails)
      - ckpt keys: "D" or "d"
      - object types: Module or state_dict (OrderedDict/dict)
      - families: ADA vs rosinality (by state_dict key patterns)

    IMPORTANT:
      - For rosinality family, only uses opts.channel_multiplier (NO decoder_multiplier).
      - strict=False will skip shape-mismatched keys to avoid RuntimeError.
    """
    print("Loading discriminator from", path)
    ext = Path(path).suffix.lower()

    # ---------- load checkpoint ----------
    ckpt = None
    if ext == ".pkl":
        with open(path, "rb") as f:
            try:
                ckpt = legacy_load(f)
                print("✅ pkl loaded via legacy_load")
            except Exception as e:
                print(f"⚠️ legacy_load failed, fallback pickle.load: {type(e).__name__}: {e}")
                f.seek(0)
                ckpt = pickle.load(f)
                print("✅ pkl loaded via pickle.load (fallback)")
    else:
        try:
            ckpt = torch.load(path, map_location="cpu")
            print("✅ checkpoint loaded via torch.load")
        except Exception as e:
            # sometimes .pkl is renamed .pt; try legacy loader before giving up
            print(f"⚠️ torch.load failed, fallback legacy_load: {type(e).__name__}: {e}")
            with open(path, "rb") as f:
                ckpt = legacy_load(f)
            print("✅ loaded via legacy_load (fallback)")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected ckpt type: {type(ckpt)}")

    # ---------- get discriminator object / state_dict ----------
    if "D" in ckpt:
        D_obj = ckpt["D"]
    elif "d" in ckpt:
        D_obj = ckpt["d"]
    else:
        raise KeyError(f"Can't find discriminator in ckpt. keys={list(ckpt.keys())}")

    # ============================================================
    # Case A: discriminator is a MODULE
    # ============================================================
    if isinstance(D_obj, torch.nn.Module):
        D_original = D_obj.eval().requires_grad_(False).to(device).float()

        # If ADA module exposes init_kwargs, re-instantiate as RGB and load weights
        init_kwargs = dict(getattr(D_original, "init_kwargs", {}))
        if init_kwargs:
            # Force RGB
            if "img_channels" in init_kwargs:
                init_kwargs["img_channels"] = 3
            elif "channels" in init_kwargs:
                init_kwargs["channels"] = 3

            disc = DiscriminatorADA(**init_kwargs)
            sd = D_original.state_dict()
            sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3, fromrgb_key="b256.fromrgb.weight")

            if strict:
                disc.load_state_dict(sd, strict=True)
            else:
                load_state_dict_skip_mismatch(disc, sd, strict=False, prefix="load_disc_any(module->ada)")

            disc = disc.to(device).eval().requires_grad_(False)
            print("✅ discriminator loaded from MODULE (re-instantiated RGB)")
            return disc

        # No init_kwargs: use module as-is
        disc = D_original.to(device).eval().requires_grad_(False)
        print("✅ discriminator loaded from MODULE (used as-is)")
        return disc

    # ============================================================
    # Case B: discriminator is a STATE_DICT
    # ============================================================
    if isinstance(D_obj, (dict, OrderedDict)):
        sd = dict(D_obj)
        family = disc_family_from_sd(sd)
        print("→ discriminator sd family:", family)

        # ---------------- ADA ----------------
        if family == "ada":
            sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3, fromrgb_key="b256.fromrgb.weight")

            a = _to_args_dict(ckpt.get("args", None))
            c_dim = int(a.get("c_dim", 0))
            img_resolution = int(a.get("img_resolution", getattr(opts, "stylegan_size", 256)))
            img_channels = 3

            optional = {}
            for k in ["architecture", "channel_base", "channel_max", "num_fp16_res", "conv_clamp", "cmap_dim"]:
                if k in a:
                    optional[k] = a[k]

            disc = DiscriminatorADA(c_dim, img_resolution, img_channels, **optional)

            if strict:
                disc.load_state_dict(sd, strict=True)
            else:
                load_state_dict_skip_mismatch(disc, sd, strict=False, prefix="load_disc_any(ada)")

            disc = disc.to(device).eval().requires_grad_(False)
            print("✅ discriminator loaded from STATE_DICT (ada)")
            return disc

        # ---------------- Rosinality ----------------
        if family == "rosinality":
            if DiscriminatorR is None:
                raise RuntimeError("rosinality discriminator weights detected, but DiscriminatorR was not provided/imported.")

            sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3, fromrgb_key="convs.0.0.weight")

            # ✅ your requirement: ONLY use channel_multiplier
            mult = getattr(opts, "channel_multiplier", 2)

            disc = None
            last_err = None
            for ctor in [
                lambda: DiscriminatorR(getattr(opts, "stylegan_size", 256), channel_multiplier=mult),
                lambda: DiscriminatorR(size=getattr(opts, "stylegan_size", 256), channel_multiplier=mult),
                lambda: DiscriminatorR(getattr(opts, "stylegan_size", 256)),
            ]:
                try:
                    disc = ctor()
                    last_err = None
                    break
                except Exception as e:
                    last_err = e

            if disc is None:
                raise RuntimeError(f"Could not construct DiscriminatorR. Last error: {last_err}")

            if strict:
                disc.load_state_dict(sd, strict=True)
            else:
                load_state_dict_skip_mismatch(disc, sd, strict=False, prefix="load_disc_any(rosinality)")

            disc = disc.to(device).eval().requires_grad_(False)
            print("✅ discriminator loaded from STATE_DICT (rosinality)")
            return disc

        raise RuntimeError(
            f"Unknown discriminator sd format. family={family}. Example keys: {list(sd.keys())[:12]}"
        )

    raise TypeError(f"Unsupported discriminator object type: {type(D_obj)}")



def load_disc_from_ckpt_into(
    discriminator: torch.nn.Module,
    ckpt: Dict[str, Any],
    *,
    strict: bool = False
) -> torch.nn.Module:
    """
    Load discriminator weights from a training checkpoint that stores weights
    under ckpt["state_dict"] with prefix "discriminator.*".
    Returns the discriminator (updated in-place).

    NOTE: strict=False does NOT prevent shape mismatch RuntimeError in PyTorch,
          so we filter mismatched keys when strict=False.
    """
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print("Checkpoint missing 'state_dict'; skipping discriminator restore.")
        return discriminator

    has_disc = any(k.startswith("discriminator.") for k in ckpt["state_dict"].keys())
    if not has_disc:
        print("Can not find Discriminator weights in checkpoint, leave current discriminator.")
        return discriminator

    sd = get_keys(ckpt, "discriminator")

    fam = disc_family_from_sd(sd)
    if fam == "ada":
        sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3, fromrgb_key="b256.fromrgb.weight")
    elif fam == "rosinality":
        sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3, fromrgb_key="convs.0.0.weight")

    if strict:
        discriminator.load_state_dict(sd, strict=True)
    else:
        load_state_dict_skip_mismatch(discriminator, sd, strict=False, prefix=f"load_disc_from_ckpt_into({fam})")

    print(f"✅ discriminator loaded from ckpt (family={fam}, strict={strict})")
    return discriminator
