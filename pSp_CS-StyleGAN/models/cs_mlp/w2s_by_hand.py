# File: w2s_by_hand.py

import torch
from torch import nn
from models.psp.stylegan2.model import EqualLinear

class W2SByHand(nn.Module):
    """
    A “by-hand” W→S mapper that uses the SAME EqualLinear(...) blocks
    as StyleGAN2’s Generator, but initialized from scratch.

    In StyleGAN2’s Generator, every ModulatedConv2d(conv) and every ToRGB(conv)
    contains:
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

    We extract exactly those 26 EqualLinear modules (for 17 convs + 9 toRGBs),
    but build them ourselves here, from scratch.

    Forward(w) takes w: [B,18,512] and returns two lists:
      - style_space:       length 17, each tensor [B, out_channels_for_that_conv]
      - to_rgb_stylespace: length  9, each tensor [B, out_channels_for_that_toRGB]
    """

    def __init__(self, style_dim=512):
        super().__init__()
        self.style_dim = style_dim

        # S_info[k] = (W_idx, out_channels, is_conv)
        #   - W_idx: which of the 18 style-vectors this slot consumes
        #   - out_channels: number of channels in that convolution (or ToRGB)
        #   - is_conv=True  → goes into style_space list
        #   - is_conv=False → goes into to_rgb_stylespace list
        #
        # This matches the exact resolution & channel layout from StyleGAN2’s Generator:
        self.S_info = [
            ( 0, 512, True ),   # S0  ← W0,  4×4 Conv (512 ch)
            ( 1, 512, False),   # S1  ← W1,  4×4 ToRGB (512 ch)

            ( 1, 512, True ),   # S2  ← W1,  8×8 Conv0_up (512 ch)
            ( 2, 512, True ),   # S3  ← W2,  8×8 Conv1 (512 ch)
            ( 3, 512, False),   # S4  ← W3,  8×8 ToRGB (512 ch)

            ( 3, 512, True ),   # S5  ← W3,  16×16 Conv0_up (512 ch)
            ( 4, 512, True ),   # S6  ← W4,  16×16 Conv1 (512 ch)
            ( 5, 512, False),   # S7  ← W5,  16×16 ToRGB (512 ch)

            ( 5, 512, True ),   # S8  ← W5,  32×32 Conv0_up (512 ch)
            ( 6, 512, True ),   # S9  ← W6,  32×32 Conv1 (512 ch)
            ( 7, 512, False),   # S10 ← W7,  32×32 ToRGB (512 ch)

            ( 7, 512, True ),   # S11 ← W7,  64×64 Conv0_up (512 ch)
            ( 8, 512, True ),   # S12 ← W8,  64×64 Conv1 (512 ch)
            ( 9, 512, False),   # S13 ← W9,  64×64 ToRGB (512 ch)

            ( 9, 512, True ),   # S14 ← W9,  128×128 Conv0_up (512 ch)
            (10, 256, True ),   # S15 ← W10, 128×128 Conv1 (256 ch)
            (11, 256, False),   # S16 ← W11, 128×128 ToRGB (256 ch)

            (11, 256, True ),   # S17 ← W11, 256×256 Conv0_up (256 ch)
            (12, 128, True ),   # S18 ← W12, 256×256 Conv1 (128 ch)
            (13, 128, False),   # S19 ← W13, 256×256 ToRGB (128 ch)

            (13, 128, True ),   # S20 ← W13, 512×512 Conv0_up (128 ch)
            (14, 64,  True ),   # S21 ← W14, 512×512 Conv1 (64 ch)
            (15, 64,  False),   # S22 ← W15, 512×512 ToRGB (64 ch)

            (15, 64,  True ),   # S23 ← W15, 1024×1024 Conv0_up (64 ch)
            (16, 32,  True ),   # S24 ← W16, 1024×1024 Conv1 (32 ch)
            (17, 32,  False),   # S25 ← W17, 1024×1024 ToRGB (32 ch)
        ]
        # Len = 26

        # ───────────────────────────────────────────────────────────────────
        # STEP 2: create 26 EqualLinear(style_dim→out_ch) modules, one per S_slot
        # ───────────────────────────────────────────────────────────────────
        self.linears = nn.ModuleList()
        for k in range(26):
            _, out_ch, _ = self.S_info[k]
            # Use EqualLinear exactly as in StyleGAN2’s ModulatedConv2d / ToRGB:
            lin = EqualLinear(
                in_dim=style_dim,
                out_dim=out_ch,
                bias=True,
                bias_init=1,       # same bias_init as in ModulatedConv2d
                lr_mul=1,          # default lr_mul=1
                activation=None    # no fused_lrelu here, because .modulation(...) in forward does no activation
            )
            # EqualLinear defaults to randn(weight)/lr_mul and zero bias
            # We leave it at that (random) for “scratch” initialization.
            self.linears.append(lin)

    def forward(self, w):
        """
        w: Tensor of shape [B, 18, 512]
        Returns two Python lists:
          - style_space       (17 tensors, one for each convolutional slot)
          - to_rgb_stylespace ( 9 tensors, one for each ToRGB slot)

        Each tensor has shape [B, out_channels] exactly matching its generator layer.
        """
        B, n_w, dim = w.shape
        assert n_w == 18 and dim == self.style_dim, \
               f"Expected w.shape = [B,18,512], got {tuple(w.shape)}"

        # Split w into 18 slices along dim=1
        w_slices = list(w.unbind(dim=1))  # list of length 18, each [B,512]

        style_space       = []
        to_rgb_stylespace = []

        for k in range(26):
            W_idx, out_ch, is_conv = self.S_info[k]
            lin = self.linears[k]
            # S_k: shape [B, out_ch]
            S_k = lin(w_slices[W_idx])
            if is_conv:
                style_space.append(S_k)
            else:
                to_rgb_stylespace.append(S_k)

        # style_space has 17 entries; to_rgb_stylespace has 9 entries
        return style_space, to_rgb_stylespace




# This S_INFO must match exactly the order used in get_stylespace_from_w
S_INFO = [
    ( 0, 512, True ),   # S0  ← W0,  4×4 Conv (512 ch)
    ( 1, 512, False),   # S1  ← W1,  4×4 ToRGB (512 ch)

    ( 1, 512, True ),   # S2  ← W1,  8×8 Conv0_up (512 ch)
    ( 2, 512, True ),   # S3  ← W2,  8×8 Conv1 (512 ch)
    ( 3, 512, False),   # S4  ← W3,  8×8 ToRGB (512 ch)

    ( 3, 512, True ),   # S5  ← W3,  16×16 Conv0_up (512 ch)
    ( 4, 512, True ),   # S6  ← W4,  16×16 Conv1 (512 ch)
    ( 5, 512, False),   # S7  ← W5,  16×16 ToRGB (512 ch)

    ( 5, 512, True ),   # S8  ← W5,  32×32 Conv0_up (512 ch)
    ( 6, 512, True ),   # S9  ← W6,  32×32 Conv1 (512 ch)
    ( 7, 512, False),   # S10 ← W7,  32×32 ToRGB (512 ch)

    ( 7, 512, True ),   # S11 ← W7,  64×64 Conv0_up (512 ch)
    ( 8, 512, True ),   # S12 ← W8,  64×64 Conv1 (512 ch)
    ( 9, 512, False),   # S13 ← W9,  64×64 ToRGB (512 ch)

    ( 9, 512, True ),   # S14 ← W9,  128×128 Conv0_up (512 ch)
    (10, 256, True ),   # S15 ← W10, 128×128 Conv1 (256 ch)
    (11, 256, False),   # S16 ← W11, 128×128 ToRGB (256 ch)

    (11, 256, True ),   # S17 ← W11, 256×256 Conv0_up (256 ch)
    (12, 128, True ),   # S18 ← W12, 256×256 Conv1 (128 ch)
    (13, 128, False),   # S19 ← W13, 256×256 ToRGB (128 ch)

    (13, 128, True ),   # S20 ← W13, 512×512 Conv0_up (128 ch)
    (14, 64,  True ),   # S21 ← W14, 512×512 Conv1 (64 ch)
    (15, 64,  False),   # S22 ← W15, 512×512 ToRGB (64 ch)

    (15, 64,  True ),   # S23 ← W15, 1024×1024 Conv0_up (64 ch)
    (16, 32,  True ),   # S24 ← W16, 1024×1024 Conv1 (32 ch)
    (17, 32,  False),   # S25 ← W17, 1024×1024 ToRGB (32 ch)
]
# Total entries = 26, with 17 “is_conv=True” and 9 “is_conv=False”


class W2SInitFromG(nn.Module):
    """
    A W→S mapper that clones (and then trains) the 26 EqualLinear layers
    from a frozen StyleGAN2 Generator (G).  Each clone starts from G’s
    pretrained weights, but is a separate module so you can fine-tune it.
    """
    def __init__(self, G: nn.Module, w_scale: float):
        super().__init__()
        self.style_dim = 512
        self.S_info = S_INFO

        # 1) Collect the conv-modulation and toRGB-modulation layers from G:
        #    (a) 4×4 conv1 + 4×4 to_rgb1
        conv_mods  = [G.conv1.conv.modulation]    # for S0
        toRGB_mods = [G.to_rgb1.conv.modulation]  # for S1

        #    (b) then for each pair of StyledConv (conv1 & conv2) plus one ToRGB:
        for conv1, conv2, to_rgb in zip(G.convs[::2], G.convs[1::2], G.to_rgbs):
            conv_mods.append(conv1.conv.modulation)
            conv_mods.append(conv2.conv.modulation)
            toRGB_mods.append(to_rgb.conv.modulation)

        assert len(conv_mods) == 17, f"Expected 17 conv mods, got {len(conv_mods)}"
        assert len(toRGB_mods) == 9,  f"Expected 9 toRGB mods, got {len(toRGB_mods)}"

        # 2) Now create brand-new clones of those 26 modulation layers:
        self.linears = nn.ModuleList()
        idx_conv  = 0
        idx_toRGB = 0

        for (W_idx, out_ch, is_conv) in self.S_info:
            if is_conv:
                src_lin = conv_mods[idx_conv]
                idx_conv += 1
            else:
                src_lin = toRGB_mods[idx_toRGB]
                idx_toRGB += 1

            # src_lin is an EqualLinear(style_dim→out_ch) from the pretrained G.
            # We want a fresh module with the exact same weight & bias.

            # Create a new EqualLinear with identical shape & hyperparams:
            cloned_lin = EqualLinear(
                in_dim=src_lin.weight.shape[1],
                out_dim=src_lin.weight.shape[0],
                bias=True,
                bias_init=0,   # we will overwrite bias directly
                lr_mul=1,
                activation=None
            )

            # Copy weight and bias from src_lin into cloned_lin:
            with torch.no_grad():
                # src_lin.weight is [out_ch, in_dim]; cloned_lin.weight is same shape
                cloned_lin.weight.copy_(w_scale * src_lin.weight)
                # src_lin.bias is [out_ch]; cloned_lin.bias is same shape
                cloned_lin.bias.copy_(w_scale * src_lin.bias)

            # By default, cloned_lin’s parameters require_grad=True, so these clones
            # are trainable.  (We do NOT freeze them.)
            self.linears.append(cloned_lin)

        # Now self.linears[k] is a fresh, trainable EqualLinear that started from G’s pretrained weights.


    def forward(self, w: torch.Tensor):
        """
        Inputs:
          w: [B, 18, 512]  (W⁺ latent from e.g. pSp or e4e)
        Returns two lists:
          - style_space       (17 tensors, each [B, out_channels_for_that_conv])
          - to_rgb_stylespace ( 9 tensors, each [B, out_channels_for_that_toRGB])
        """
        B, n_w, dim = w.shape
        assert n_w == 18 and dim == self.style_dim, f"Expected w.shape=[B,18,512], got {tuple(w.shape)}"

        # Split along the “layer” dimension so that w_slices[i] is [B,512]:
        w_slices = list(w.unbind(dim=1))

        style_space_list       = []
        to_rgb_stylespace_list = []
        idx_conv  = 0
        idx_toRGB = 0

        for k, (W_idx, out_ch, is_conv) in enumerate(self.S_info):
            lin = self.linears[k]         # an EqualLinear(512 → out_ch)
            S_k = lin(w_slices[W_idx])    # shape [B, out_ch]

            if is_conv:
                style_space_list.append(S_k)
                idx_conv += 1
            else:
                to_rgb_stylespace_list.append(S_k)
                idx_toRGB += 1

        return style_space_list, to_rgb_stylespace_list



