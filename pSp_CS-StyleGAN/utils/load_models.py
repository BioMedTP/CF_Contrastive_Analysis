import torch
from argparse import Namespace
from models.psp import pSp
from models.cs_mlp.mlp3D import MappingNetwork_cs_3Dmlp, MappingNetwork_pSp_cs, MappingNetwork_c_s1_s2
from utils.paths import model_paths


def load_pSp_cs_models(opts, device="cuda"):
    
    if opts.stylegan_weights is None:
        opts.stylegan_weights = model_paths[opts.dataset_type]["stylegan_weights"]
    if opts.pSp_checkpoint_path is None:
        opts.pSp_checkpoint_path = model_paths[opts.dataset_type]["pSp_checkpoint_path"]
        
    opts.encoder_type = opts.pSp_encoder_type
    opts.latent_dim = 512    
    #     if not hasattr(opts, 'encoder_type'):
#         opts.encoder_type = 'GradualStyleEncoder'
    pSp_net = pSp(opts).to(device)

    if pSp_net.latent_avg is None:
        pSp_net.latent_avg = pSp_net.decoder.mean_latent(int(1e5))[0].detach()
    pSp_net.eval()
    
    cs_mlp_net = MappingNetwork_cs_3Dmlp(opts).to(device)	
    if opts.csmlp_checkpoint_path is not None:
        print(f'Loading csmlp model from checkpoint {opts.csmlp_checkpoint_path}')
        ckpt = torch.load(opts.csmlp_checkpoint_path, map_location='cpu')
        cs_mlp_net.load_state_dict(ckpt['state_dict_cs_net'])  
    return pSp_net, cs_mlp_net


def load_pSp_cs1s2_models(opts, device="cuda"):
    
    if opts.stylegan_weights is None:
        opts.stylegan_weights = model_paths[opts.dataset_type]["stylegan_weights"]
    if opts.pSp_checkpoint_path is None:
        opts.pSp_checkpoint_path = model_paths[opts.dataset_type]["pSp_checkpoint_path"]
        
    opts.encoder_type = opts.pSp_encoder_type
    #opts.latent_dim = 512    
    #     if not hasattr(opts, 'encoder_type'):
#         opts.encoder_type = 'GradualStyleEncoder'
    pSp_net = pSp(opts).to(device)

    if pSp_net.latent_avg is None:
        pSp_net.latent_avg = pSp_net.decoder.mean_latent(int(1e5))[0].detach()
    pSp_net.eval()
    
    cs_mlp_net = MappingNetwork_c_s1_s2(opts).to(device)	
    if opts.csmlp_checkpoint_path is not None:
        print(f'Loading csmlp model from checkpoint {opts.csmlp_checkpoint_path}')
        cs_ckpt = torch.load(opts.csmlp_checkpoint_path, map_location=device, weights_only=True) 
        print(f"Loading csmlp from path: {opts.csmlp_checkpoint_path}")   
        cs_mlp_net.load_state_dict(cs_ckpt['state_dict_cs_net'])   

    return pSp_net, cs_mlp_net

# def load_pSp_cs_models(model_path, pSp_path, device, eval_models=True):
#     # load model ckpts
#     #print('Loading trained checkpoint from path: {}'.format(model_path))
#     model_ckpt = torch.load(model_path, map_location='cpu', weights_only=True)  
#     opts = model_ckpt['opts']
#     if opts['output_size']:
#         opts['stylegan_size'] = opts['output_size']
#     opts = Namespace(**opts)
#     if not hasattr(opts, 'encoder_type'):
#         opts.encoder_type = 'GradualStyleEncoder'
#     if not hasattr(opts, 'output_size'):
#         opts.output_size = 1024     
#     #opts.pSp_checkpoint_path = '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_encoder_constructive/results/celebaHQ/best_model.pt'
#     #### load pSp model including StyleGAN2 generator ####

#     opts.pSp_checkpoint_path = pSp_path
#     pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
    
#     cs_mlp_net = MappingNetwork_pSp_cs(opts).to(device)
#     print(f"Loading csmlp from path: {model_path}")   
#     cs_ckpt  = model_ckpt['state_dict_cs_enc']  
#     cs_mlp_net.load_state_dict(cs_ckpt)

#     pSp_net.eval()
#     # if eval_models:
#     #     pSp_net.eval()
#     #     cs_mlp_net.eval()

#     return pSp_net, cs_mlp_net, opts