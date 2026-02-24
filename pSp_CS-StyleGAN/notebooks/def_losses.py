
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calc_cs_loss(latent_bg_s, latent_bg, latent_t, latent_bg_pSp, latent_t_pSp):
    loss_dict = {}
    loss = 0.0
    # silent of target is zero

    loss_sbg = F.mse_loss(latent_bg_s, torch.zeros(latent_bg_s.shape))
    loss_dict['loss_sbg'] = float(loss_sbg)
    loss += loss_sbg
    # latent distance	
    loss_ld_bg = F.mse_loss(latent_bg, latent_bg_pSp) 
    loss_dict['loss_ld_bg'] = float(loss_ld_bg)
    loss += loss_ld_bg
    loss_ld_t = F.mse_loss(latent_t, latent_t_pSp) 
    loss_dict['loss_ld_t'] = float(loss_ld_t)
    loss += loss_ld_t		

    loss_dict['loss'] = float(loss)
    return loss, loss_dict  #, id_logs	

def L1_regularization(output, reg_type='row'):
    """Compute loss using L1 regularization."""

    # Apply row-wise or column-wise L1 regularization
    if reg_type == 'row':
        # Row-wise L1 penalty: sum over columns, apply L1 norm to each row
        l1_penalty = torch.sum(torch.abs(output).sum(dim=2))  # Summing across columns (width)
    elif reg_type == 'column':
        # Column-wise L1 penalty: sum over rows, apply L1 norm to each column
        l1_penalty = torch.sum(torch.abs(output).sum(dim=1))  # Summing across rows (height)
    elif reg_type == 'element':
        l1_penalty = torch.sum(torch.abs(output))  # Summing across all elements
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}. Use 'row' or 'column' or all elements.")

    return  l1_penalty


