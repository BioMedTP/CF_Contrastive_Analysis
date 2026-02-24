"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import pickle

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

def infer_input_nc_from_transform(transform_name):
    if transform_name.startswith("rgb"):
        return 3
    elif transform_name.startswith("grayscale"):
        return 1
    else:
        raise ValueError(f"Unknown transform type: {transform_name}")

class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
		self.opts.input_nc = infer_input_nc_from_transform(self.opts.data_transform)
  
		# Define architecture
		self.encoder = self.set_encoder()
		#self.encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		# self.decoder = Generator(self.opts.stylegan_size, 512, 8)
  
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		# self.previous_train_ckpt = previous_train_ckpt
		self.load_weights()

	def set_encoder(self):
		if self.opts.pSp_encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.pSp_encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.pSp_encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.pSp_encoder_type))
		return encoder

	def load_weights(self):
		print('Loading pSp from checkpoint: {}'.format(self.opts.pSp_checkpoint_path))
		ckpt = torch.load(self.opts.pSp_checkpoint_path, map_location='cpu', weights_only=True)
		self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)	

		# Load StyleGAN generator weights
		G_path = self.opts.stylegan_weights
		with open(G_path, 'rb') as f:
			self.G = pickle.load(f)['G_ema'].to(self.opts.device)
		print('Loading generator weights from %s' % G_path)
		if self.opts.learn_in_w:
			self.__load_latent_avg(G_path, repeat=1)
		else:
			self.__load_latent_avg(G_path, repeat=self.opts.n_styles)
 
	def forward(self, x, resize=True, latent_mask=None, input_code=False,
				randomize_noise=False, inject_latent=None, alpha=None, 
				recon_model=False, encode_model=False):

		if input_code:
			codes = x
		else:
			codes = self.encoder(x)

			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		# Return only latent codes
		if encode_model:
			return codes

		# Reconstruction path
		elif recon_model:
			noise_mode = 'random' if randomize_noise else 'const'
			images = self.G.synthesis(codes, noise_mode=noise_mode, force_fp32=True)
			if resize:
				images = self.face_pool(images)
			return images

		# Default: return both reconstruction + codes
		else:
			noise_mode = 'random' if randomize_noise else 'const'
			images = self.G.synthesis(codes, noise_mode=noise_mode, force_fp32=True)
			if resize:
				images = self.face_pool(images)
			return images, codes


		

	def set_opts(self, opts):
		self.opts = opts

	# def __load_latent_avg(self, ckpt, repeat=None):
	# 	if 'latent_avg' in ckpt:
	# 		self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
	# 		if repeat is not None:
	# 			self.latent_avg = self.latent_avg.repeat(repeat, 1)
	# 	else:
	# 		self.latent_avg = None

	def __load_latent_avg(self, G_path, repeat=None):
		if G_path is not None:
			self.latent_avg = self.G.mapping.w_avg.to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
