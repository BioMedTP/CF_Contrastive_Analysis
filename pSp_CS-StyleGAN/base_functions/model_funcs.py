import torch
from argparse import Namespace
from models.psp import pSp
from models.cs_mlp.mlp3D import MappingNetwork_pSp_cs, MappingNetwork_cs_3Dmlp, MappingNetwork_c_s1_s2
import h5py
from tqdm import tqdm



# def load_pSp_cmlp_models(model_path, device, eval_models=True):
#     # load model ckpts
#     #print('Loading trained checkpoint from path: {}'.format(model_path))
#     model_ckpt = torch.load(model_path, map_location='cpu', weights_only=True)  
#     opts = model_ckpt['opts']
#     if opts['output_size']:
#         opts['stylegan_size'] = opts['output_size']
#     opts = Namespace(**opts)
#     if not hasattr(opts, 'encoder_type'):
#         opts.encoder_type = 'GradualStyleEncoder'
#     if not hasattr(opts, 'pSp_encoder_type'):
#         opts.pSp_encoder_type = 'GradualStyleEncoder'
#     if not hasattr(opts, 'output_size'):
#         opts.output_size = 1024     
#     #opts.pSp_checkpoint_path = '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_encoder_constructive/results/celebaHQ/best_model.pt'
#     #### load pSp model including StyleGAN2 generator ####
#     pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
    
#     cs_mlp_net = MappingNetwork_cs_independent(opts).to(device)
#     print(f"Loading csmlp from path: {model_path}")   
#     cs_ckpt  = model_ckpt['state_dict_cs_enc']  
#     cs_mlp_net.load_state_dict(cs_ckpt)

#     if eval_models:
#         pSp_net.eval()
#         cs_mlp_net.eval()

#     return pSp_net, cs_mlp_net, opts


def load_pSp_cmlp_models(pSp_cs_path, pSp_path=None, dataset_type='bloodmnist', device='cuda', eval_models=True):
    # load model ckpts
    #print('Loading trained checkpoint from path: {}'.format(pSp_cs_path))
    model_ckpt = torch.load(pSp_cs_path, map_location='cpu', weights_only=True)  
    opts = model_ckpt['opts']
    # if opts['output_size']:
    #     opts['stylegan_size'] = opts['output_size']
    opts = Namespace(**opts)
    if not hasattr(opts, 'encoder_type'):
        opts.encoder_type = 'GradualStyleEncoder'
    if not hasattr(opts, 'stylegan_size'):
        opts.stylegan_size = 1024    
    if not hasattr(opts, 'decoder_multiplier'):
        opts.decoder_multiplier = 2 
    opts.decoder_multiplier
    
    if dataset_type is not None:
        opts.dataset_type = dataset_type
    if pSp_path is not None:
        opts.pSp_checkpoint_path = pSp_path
    # print("self.opts.pSp_checkpoint_path--------", opts.pSp_checkpoint_path)
    pSp_net = pSp(opts, previous_train_ckpt=None).to(device)
    if opts.exp_scheme == "baseline_c_s1s2":
        print(f"Loading c_s1s2 mlp from path: {pSp_cs_path}")   
        cs_mlp_net = MappingNetwork_c_s1_s2(opts).to(device)	
    else:
        print(f"Loading cs mlp from path: {pSp_cs_path}")   
        cs_mlp_net = MappingNetwork_pSp_cs(opts).to(device)

    
    cs_ckpt  = model_ckpt['state_dict_cs_enc']  
    cs_mlp_net.load_state_dict(cs_ckpt)

    if eval_models:
        pSp_net.eval()
        cs_mlp_net.eval()

    return pSp_net, cs_mlp_net, opts



def process_recon_swap(cs_mlp_net, pSp_net, input_images_bg, input_images_t, opts, zero_sx=True):

    with torch.no_grad():

        # w_pSp_bg = pSp_net.forward(input_images_bg, encode_only=True)  
        # w_pSp_t = pSp_net.forward(input_images_t, encode_only=True)  
        
        rec_bg_pSp, w_pSp_bg = pSp_net.forward(input_images_bg, return_latents=True)
        rec_t_pSp, w_pSp_t = pSp_net.forward(input_images_t, return_latents=True)
        
        latent_bg_c, latent_bg_s = cs_mlp_net(w_pSp_bg, zero_out_silent=opts.zero_out_silent_bg)
        latent_t_c, latent_t_s = cs_mlp_net(w_pSp_t, zero_out_silent=opts.zero_out_silent_t) 
        if zero_sx:
            print('set sx = 0')
            recon_bg = pSp_net.forward(latent_bg_c, input_code=True, randomize_noise=False, recon_modle=True)
            recon_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
        
            swap_bg = pSp_net.forward(latent_bg_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
            swap_t = pSp_net.forward(latent_t_c, input_code=True, randomize_noise=False, recon_modle=True) 
        
        else:
            print('use sx')
            recon_bg = pSp_net.forward(latent_bg_c + latent_bg_s, input_code=True, randomize_noise=False, recon_modle=True)
            recon_t = pSp_net.forward(latent_t_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
        
            swap_bg = pSp_net.forward(latent_bg_c + latent_t_s, input_code=True, randomize_noise=False, recon_modle=True)
            swap_t = pSp_net.forward(latent_t_c + latent_bg_s, input_code=True, randomize_noise=False, recon_modle=True) 
        
        output_latents = {
                    'w_bg_pSp': w_pSp_bg,
                    'w_t_pSp': w_pSp_t,
                    'latent_bg_c':latent_bg_c,
                    'latent_bg_s':latent_bg_s,
                    'latent_t_c':latent_t_c,
                    'latent_t_s':latent_t_s,
                   }
        
        output_images = {
                    'recon_pSp_bg':rec_bg_pSp,
                    'recon_pSp_t':rec_t_pSp,
                    'recon_bg':recon_bg,
                    'recon_t':recon_t,
                    'swap_bg':swap_bg,
                    'swap_t':swap_t
                   }
        return output_latents, output_images 







