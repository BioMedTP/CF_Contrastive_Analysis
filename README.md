# Counterfactual Contrastive Analysis

# Counterfactual Contrastive Analysis

This repository contains the official implementation of **Counterfactual Contrastive Analysis**.


## Overview

We propose a **classifier-free** approach for Visual Counterfactual Explanations (VCEs) based on **Contrastive Analysis (CA)**.  
Given two datasets \(X\) and \(Y\), we disentangle **common** factors \(c\) and **salient** factors \(s_x, s_y\), and generate counterfactuals by swapping salient factors while preserving common content.  
The method is built on **StyleGAN2** and includes an optional **feature-space (F-space) refinement** stage for higher-fidelity edits.

---

## Repository structure

- `psp_CS-StyleGAN/` : CS-StyleGAN training & inference code (Stage 1 + Stage 2 refinement)
- `StyleFeatureEditor-CS/` : feature-space refinement modules / utilities
- `diffusionDDIM/` : optional DDIM diffusion module (data prep, training, sampling)

---

## Environment setup

We recommend using `conda`:

```bash
conda create -n cfca python=3.10 -y
conda activate cfca
pip install -r requirements.txt

## Framework

![Framework](example/architecture.png)

Our framework learns disentangled **common** and **salient** factors in latent space, and generates counterfactuals by **swapping salient factors** while preserving common content. An optional **F-space refinement** module further improves fine details.

## Training (Two-stage)

We train CS-StyleGAN in two stages.

### Stage 1 — Common/Salient separator (W-space)
We optimize the separator \(H_{cs}\) to disentangle **common** and **salient** factors in the StyleGAN latent space.  
Training alternates between:
- updating the discriminator \(\mathcal{D}\) and regressors \(\mathcal{R}\) (to enforce common alignment and common–salient independence), and  
- updating \(H_{cs}\) with \(\mathcal{D}\) and \(\mathcal{R}\) frozen.

**Typical hyperparameters**
- learning rate: \(1e{-4}\) for \(\mathcal{D}, \mathcal{R}\); \(1e{-3}\) for \(H_{cs}\)
- \(\lambda_{lat}=0.01\), \(\lambda_D=\lambda_R=0.02\), LPIPS weight \(=0.8\) (others \(=1\))
- warm-up: 2,000 steps with reconstruction losses before alternating updates (optional)
- total steps: ~160k (Adam)

### Stage 2 — F-space refinement (optional)
We refine synthesis details in StyleGAN intermediate **feature space (F-space)** using feature shifts induced by salient swapping.  
The refinement module \((E_f, F_{\text{Adpt}})\) is trained with reconstruction/perceptual losses and an adversarial term.

**Typical hyperparameters**
- \(\lambda_{adv}=0.02\), LPIPS weight \(=0.8\)
- total steps: ~200k (Ranger), learning rate \(2e{-4}\)

> We use pSp for inversion, StyleGAN2 as generator backbone, and an F-space refinement module following prior feature-space editing practice.
