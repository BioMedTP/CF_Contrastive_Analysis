# Counterfactual Contrastive Analysis

This repository contains the official implementation of **Counterfactual Contrastive Analysis**.

---

## Overview

We propose a **classifier-free** approach for Visual Counterfactual Explanations (VCEs) based on **Contrastive Analysis (CA)**.  
Given two datasets **X** and **Y**, we disentangle **common** factors (**c**) and **salient** factors (**s_x**, **s_y**), and generate counterfactuals by **swapping salient factors** while preserving common content.  
The method is built on **StyleGAN2** and includes an optional **feature-space (F-space) refinement** stage for higher-fidelity edits.

---

## Framework

![Framework](examples/architecture.png)

Our framework learns disentangled **common** and **salient** factors in latent space, and generates counterfactuals by **swapping salient factors**.  
An optional **F-space refinement** module further improves fine details.

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
