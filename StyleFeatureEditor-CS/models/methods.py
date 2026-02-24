import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from models.hyperinverter.stylegan2_ada import Discriminator 
from models.psp.stylegan2.model import Discriminator as Ros_Discriminator
from utils.stylegan2_ada_legacy import load_network_pkl as legacy_load


from utils.class_registry import ClassRegistry
# from utils.common_utils import get_keys
from utils.model_utils import toogle_grad
from configs.paths import DefaultPathsClass
from argparse import Namespace
from training.loggers import BaseTimer
from runners.model_funcs import load_e4e_cs_model, load_pSp_cs_models

import os
from typing import Dict, Any
import os
from pathlib import Path
from collections import OrderedDict
from models.disc_utils import get_keys, adapt_disc_state_dict_1ch_to_3ch
from models.disc_utils import load_disc_any, load_disc_from_ckpt_into

sys.path.append("./utils")
methods_registry = ClassRegistry()




@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path", "w_space_encoder", "dataset", "stylegan_size", "pSp_cs_path", "pSp_path", "inverter_pth", "channel_multiplier"))
class FSEFull(nn.Module):
    def __init__(self,
        device="cuda:0",
        checkpoint_path=None,
        inverter_pth=None,
        stylegan_size=None,
        w_space_encoder=None,
        dataset=None,
        pSp_cs_path=None,
        pSp_path=None,          # ✅ 加这一行
        channel_multiplier=None
    ):

        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "w_space_encoder": w_space_encoder,
            "dataset" : dataset, 
            "stylegan_size": stylegan_size,
            "channel_multiplier": channel_multiplier,
            "pSp_cs_path": pSp_cs_path,
            "pSp_path": pSp_path,
            "inverter_pth": inverter_pth
        }

        per_dataset_paths = DefaultPathsClass(dataset=dataset)
        self.opts.update({k: v for k, v in per_dataset_paths})
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = self.opts.inverter_pth
        self.channel_multiplier = self.opts.channel_multiplier
        print("[DEBUG----------------------------] channel_multiplier =", self.channel_multiplier)
        print("[DEBUG----------------------------] stylegan_size =", self.opts.stylegan_size)
        # print("[DEBUG----------------------------] pSp_cs_path =", self.opts.pSp_cs_path)
        # print("[DEBUG----------------------------] checkpoint_path =", self.opts.checkpoint_path)
        # print("[DEBUG----------------------------] inverter_pth =", self.opts.inverter_pth)
        # print("[DEBUG----------------------------] pSp_path =", self.opts.pSp_path)
        # --------------------
        # INVERTER + ENCODER
        # --------------------
        self.inverter = self.set_inverter()
        self.encoder = self.set_encoder()

        # --------------------
        # 🔴 FEATURE ADAPTER（核心）
        # --------------------
        # SFE encoder outputs 512 channels
        # ADA StyleGAN (cm=1) expects 256 channels
        # if self.channel_multiplier == 1:
        #     self.feature_adapter = nn.Conv2d(512, 256, kernel_size=1)
        # else:
        #     self.feature_adapter = nn.Identity()

        # --------------------
        # W-SPACE ENCODER
        # --------------------
        if self.opts.w_space_encoder == "e4e":
            print("Using e4e as w encoder, loading e4e-cs model ....")
            self.e4e_cs_model = load_e4e_cs_model(model_path=self.opts.e4e_cs_path, device=self.device)
            
        elif self.opts.w_space_encoder == "pSp":
            print("Using pSp as w encoder, loading pSp-cs model ....")
            if pSp_cs_path is not None:
                self.opts.pSp_cs_path = pSp_cs_path
            if pSp_path is not None:
                self.opts.pSp_path = pSp_path 
            self.pSp_encoder, self.pSp_cs_model, self.pSp_cs_opts = load_pSp_cs_models(pSp_cs_path=self.opts.pSp_cs_path, 
                            pSp_path=self.opts.pSp_path, device=self.device, eval_models=True)   
        else:
            raise ValueError(
                f"Unknown training w encoder type: {self.opts.w_space_encoder}"
            )   
        # --------------------
        # DECODER (StyleGAN)
        # --------------------
        self.decoder = Generator(
            self.opts.stylegan_size,
            512,
            8,
            channel_multiplier=self.channel_multiplier   
        )

        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    # ======================================================
    # DISCRIMINATOR
    # ======================================================
    def load_disc(self):
        dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        self.discriminator = load_disc_any(
            self.opts.stylegan_weights_pkl,
            device=dev,
            opts=self.opts,
            legacy_load=legacy_load,
            DiscriminatorADA=Discriminator,
            DiscriminatorR=Ros_Discriminator,
            strict=False,
        )


    # ======================================================
    # WEIGHTS
    # ======================================================
    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print(f"Loading refined-pSp-cs model from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

            self.discriminator = load_disc_from_ckpt_into(self.discriminator, ckpt, strict=False)

            enc_sd = get_keys(ckpt, "encoder")
            inv_sd = get_keys(ckpt, "inverter")

            # ---- debug: print mismatch BEFORE loading ----
            keys_to_check = [
                "body.4.body.5.bias",
                "body.4.body.5.running_mean",
                "conv.weight",
                "conv.bias",
            ]

            for k in keys_to_check:
                print("\n[CHECK]", k)

                print("  encoder has key? ", k in enc_sd)
                if k in enc_sd and k in self.encoder.state_dict():
                    print("    encoder ckpt :", tuple(enc_sd[k].shape))
                    print("    encoder model:", tuple(self.encoder.state_dict()[k].shape))

                print("  inverter has key? ", k in inv_sd)
                if k in inv_sd and k in self.inverter.state_dict():
                    print("    inverter ckpt :", tuple(inv_sd[k].shape))
                    print("    inverter model:", tuple(self.inverter.state_dict()[k].shape))

            # ---- now actually load, with try/except to know who crashes ----
            try:
                self.encoder.load_state_dict(enc_sd, strict=True)
                print("✅ encoder strict load OK")
            except RuntimeError as e:
                print("❌ encoder strict load FAILED")
                raise

            try:
                self.inverter.load_state_dict(inv_sd, strict=True)
                print("✅ inverter strict load OK")
            except RuntimeError as e:
                print("❌ inverter strict load FAILED")
                raise

        else:
            print(f"Loading Discriminator and Inverter from Inverter checkpoint: {self.inverter_pth}")
            ckpt = torch.load(self.inverter_pth, map_location="cpu")

            self.discriminator = load_disc_from_ckpt_into(self.discriminator, ckpt, strict=False)

            inv_sd = get_keys(ckpt, "inverter")
            # 也可以同样 debug 一下
            self.inverter.load_state_dict(inv_sd, strict=True)


        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        print("Loading Decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        # print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.e4e_path))
        # e4e_ckpt = torch.load(self.opts.e4e_path, map_location='cpu', weights_only=True)
        # self.decoder.load_state_dict(get_keys(e4e_ckpt, 'decoder'), strict=True)
        self.decoder = self.decoder.eval().to(self.device)
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.device)
        else:
            # generate a latent_avg by sampling (e.g. 1e5 random z’s) and taking the mean
            self.latent_avg = self.decoder.mean_latent(int(1e5))[0].detach()
        toogle_grad(self.decoder, False)

        if self.opts.w_space_encoder == "e4e":
            print("Loading E4E from", self.opts.e4e_path)
            ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
            self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
            toogle_grad(self.e4e_encoder, False)
        elif self.opts.w_space_encoder == "pSp":
            toogle_grad(self.pSp_encoder, False)
        else:
            raise ValueError(
                f"Unknown training w encoder type: {self.opts.w_space_encoder}"
            )        

    # # ======================================================
    # # ENCODER FACTORY
    # # ======================================================
    # def set_encoder(self):
    #     n_styles = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
    #     self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) 
    #     if self.opts.w_space_encoder == "e4e":
    #         self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
    #     # StyleGAN-dependent channels
    #     if self.opts.channel_multiplier == 1:
    #         in_channels = 768
    #     else:
    #         in_channels = 1024
    #     # w_feat_channels = 512
    #     # in_channels = style_feat_channels         
    #     feat_editor = psp_encoders.Encoder_adapter(opts=self.opts, n_styles=18)
    #     return feat_editor  # trainable part
    def set_inverter(self):
        n_styles = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=n_styles)
        print('n_styles: ', n_styles)
        return inverter  # trainable part
        
    def set_encoder(self):
        if self.opts.w_space_encoder == "e4e":
            self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        if self.opts.dataset == "brats_edit":
            feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        else:   
            feat_editor = psp_encoders.ContentLayerDeepFast(6, 512, 256)
        return feat_editor  # trainable part
    
    
    # ======================================================
    # FORWARD
    # ======================================================
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            #print("[predicted_feat----------------------------] =", predicted_feat.shape)
            
            w_recon = w_recon + self.latent_avg
            #print("[w_recon----------------------------] =", w_recon.shape)        
            _, w_feats = self.decoder(
                [w_recon],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )

            w_feat = w_feats[9]  # bs x 512 x 64 x 64 
            #print("[w_feat----------------------------] =", w_feat.shape)
            
            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            #print("[fused_feat----------------------------] =", fused_feat.shape)
            
            delta = torch.zeros_like(fused_feat)  # inversion case
            #print("[delta----------------------------] =", delta.shape)

        cat_fuse = torch.cat([fused_feat, delta], dim=1)
        #print("[cat_fuse----------------------------] =", cat_fuse.shape)                     
                             
        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        #print("[edited_feat----------------------------] =", edited_feat.shape)
        
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images




@methods_registry.add_to_registry(
    "fse_inverter",
    stop_args=("self", "checkpoint_path", "dataset", "stylegan_size", "channel_multiplier")
)
class FSEInverter(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 checkpoint_path=None,
                 stylegan_size=1024,
                 channel_multiplier=2,
                 dataset=None):
        super(FSEInverter, self).__init__()

        # --------------------
        # OPTS
        # --------------------
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "dataset": dataset,
            "stylegan_size": stylegan_size,
            "channel_multiplier": channel_multiplier
        }

        per_dataset_paths = DefaultPathsClass(dataset=dataset)
        self.opts.update({k: v for k, v in per_dataset_paths})
        self.opts = Namespace(**self.opts)

        self.device = device

        # --------------------
        # ENCODER (Inverter)
        # --------------------
        self.inverter = self.set_inverter()

        # --------------------
        # 🔴 FEATURE ADAPTER (关键新增)
        # --------------------
        # SFE encoder outputs 512-ch features
        # ADA StyleGAN (channel_multiplier=1) expects 256-ch mid features
        # if self.opts.channel_multiplier == 1:
        #     self.feature_adapter = nn.Conv2d(512, 256, kernel_size=1)
        # else:
        #     self.feature_adapter = nn.Identity()

        # --------------------
        # DECODER (StyleGAN)
        # --------------------
        self.decoder = Generator(
            self.opts.stylegan_size,
            512,
            8,
            channel_multiplier=self.opts.channel_multiplier
        )

        self.latent_avg = None

        # --------------------
        # DISCRIMINATOR + WEIGHTS
        # --------------------
        self.load_disc()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    # ======================================================
    # DISCRIMINATOR
    # ======================================================
    # ======================================================
    # DISCRIMINATOR
    # ======================================================
    def load_disc(self):
        dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        self.discriminator = load_disc_any(
            self.opts.stylegan_weights_pkl,
            device=dev,
            opts=self.opts,
            legacy_load=legacy_load,
            DiscriminatorADA=Discriminator,
            DiscriminatorR=Ros_Discriminator,
            strict=False,
        )

    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(k.split(".")[0] for k in ckpt["state_dict"].keys())
        if "discriminator" not in unique_keys:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")
            return

        sd = get_keys(ckpt, "discriminator")

        # If checkpoint is grayscale but model is RGB, adapt fromrgb weights
        sd = adapt_disc_state_dict_1ch_to_3ch(sd, target_in_ch=3)

        self.discriminator.load_state_dict(sd, strict=True)
        print("✅ discriminator loaded from ckpt (1ch->3ch fromrgb adapted)")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading  from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)

        print("Loading decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        toogle_grad(self.decoder, False)

    # ======================================================
    # ENCODER FACTORY

    def set_inverter(self):
        n_styles = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=n_styles)
        return inverter  # trainable part

    # ======================================================
    # FORWARD
    # ======================================================
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        w_recon, predicted_feat = self.inverter.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
        # print("predicted_feat.shape------------------", predicted_feat.shape )
        # print("w_recon.shape------------------", w_recon.shape )
        _, w_feats = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        # print("w_feat.shape------------------", w_feat.shape )
        fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)
        # print("fused_feat.shape------------------", fused_feat.shape )
        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )
        
        if return_latents:
            if not self.inverter.training:
                fused_feat = fused_feat.cpu()
                w_feat = w_feat.cpu()
            return images, w_recon, fused_feat, w_feat
        return images


