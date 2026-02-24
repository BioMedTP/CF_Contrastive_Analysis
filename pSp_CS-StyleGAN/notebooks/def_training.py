
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from notebooks.def_losses import L1_regularization, calc_cs_loss



def train_mlp(mlp_net, optimizer, n_epoch, lambda_l1, reg_type='row'):

    for n in range(n_epoch):
        lambda_l1 = lambda_l1 * (0.01 ** n)
        latent_bg_pSp = torch.randn(8,18,512)
        latent_t_pSp = torch.randn(8,18,512)

        latent_bg_c, latent_bg_s = mlp_net(latent_bg_pSp)
        latent_t_c, latent_t_s = mlp_net(latent_t_pSp) 

        #latent_t_s = soft_threshold(latent_t_s, lambda_l1)

        latent_bg = latent_bg_c
        latent_t = latent_t_c + latent_t_s 

        loss_rec, loss_dict_rec= calc_cs_loss(latent_bg_s, latent_bg, latent_t, latent_bg_pSp, latent_t_pSp)
        
        
        # Add L1 or L2 regularization for sparse 
        # Compute L1 norm (for regularization)
        l1_penalty = L1_regularization(latent_t_s, reg_type=reg_type)  # Use thresholded output directly

        loss = loss_rec + lambda_l1 * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if n % 100 == 0:
            print('step '+ str(n))
            print(loss_dict_rec, '\n')
            print(l1_penalty, '\n')

    return mlp_net


def train(mlp_net, optimizer, n_epoch, lambda_l1, reg_type='row'):

    for n in range(n_epoch):
        lambda_l1 = lambda_l1 * (0.01 ** n)
        latent_bg_pSp = torch.randn(8,18,512)
        latent_t_pSp = torch.randn(8,18,512)

        latent_bg_c, latent_bg_s = mlp_net(latent_bg_pSp)
        latent_t_c, latent_t_s = mlp_net(latent_t_pSp) 

        #latent_t_s = soft_threshold(latent_t_s, lambda_l1)

        latent_bg = latent_bg_c
        latent_t = latent_t_c + latent_t_s 

        loss_rec, loss_dict_rec= calc_cs_loss(latent_bg_s, latent_bg, latent_t, latent_bg_pSp, latent_t_pSp)
        
        
        # Add L1 or L2 regularization for sparse 
        # Compute L1 norm (for regularization)
        l1_penalty = L1_regularization(latent_t_s, reg_type=reg_type)  # Use thresholded output directly

        loss = loss_rec + lambda_l1 * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if n % 100 == 0:
            print('step '+ str(n))
            print(loss_dict_rec, '\n')
            print(l1_penalty, '\n')

    return mlp_net

