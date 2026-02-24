import os
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
from pytorch_msssim import ssim
matplotlib.use('Agg')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import tqdm
from utils import common, train_utils
from criteria import id_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from models.mlp3D import MappingNetwork_cs_independent, EqualizedLinear
# from models.mlp2D import MappingNetwork_cs_independent, EqualizedLinear
import math
from utils.common_utils import visualize_batch_grid
from utils.data_utils import get_special_images

def get_paired_random_batch(loader_X, loader_Y):
    """
    Returns one random paired batch (same batch index)
    from loader_X and loader_Y.
    """
    num_batches = len(loader_X)
    idx = random.randint(0, num_batches - 1)

    # iterate both loaders in sync
    for i, (batch_X, batch_Y) in enumerate(zip(loader_X, loader_Y)):
        if i == idx:
            return batch_X, batch_Y

    raise RuntimeError("Batch index out of range")


class Coach_csmlp:
	def __init__(self, opts, previous_train_ckpt=None):
		self.opts = opts
		self.global_step = 0
		self.n_print = 0
		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device
		self.opts.style_dim = int(math.log(self.opts.output_size, 2)) * 2 - 2
		
		self.seed_experiments(opts.seed)

		# Initialize network
		self.pSp_c_net = pSp(self.opts, None).to(self.device)
		self.pSp_s_net = pSp(self.opts, None).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.pSp_c_net.latent_avg is None:
			self.pSp_c_net.latent_avg = self.pSp_c_net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:	
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_bg_dataset, self.train_t_dataset, self.test_bg_dataset, self.test_t_dataset = self.configure_datasets()

		self.train_bg_dataloader = DataLoader(self.train_bg_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		
		self.train_t_dataloader = DataLoader(self.train_t_dataset,
									batch_size=self.opts.batch_size,
									shuffle=True,
									num_workers=int(self.opts.workers),
									drop_last=True)
		
		self.test_bg_dataloader = DataLoader(self.test_bg_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)
		
		self.test_t_dataloader = DataLoader(self.test_t_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)		

		batch_X, batch_Y = get_paired_random_batch(
			self.test_X_dataloader,
			self.test_Y_dataloader
		)

		self.train_X_dataset = batch_X.to(self.device)
		self.train_Y_dataset = batch_Y.to(self.device)
		# Initialize logger
		self.log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(self.log_dir, exist_ok=True)
		# self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		self.init_additional_params(previous_train_ckpt)
		
	def seed_experiments(self, seed):
		# Set the random seed for reproducibility
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.

		# Ensures deterministic behavior for some PyTorch operations
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


	def	init_additional_params(self, prev_train_ckpt):
		if self.opts.save_training_data and prev_train_ckpt is not None:
			self.best_val_loss = prev_train_ckpt['best_val_loss']
			if self.opts.keep_optimizer :
				self.optimizer.load_state_dict(prev_train_ckpt['optimizer'])
	
	def inference_special(self, x_bg, x_t, name):

		with torch.no_grad():

			_, w_c_bg = self.pSp_c_net.forward(x_bg, return_latents=True)
			_, w_s_bg = self.pSp_s_net.forward(x_bg, return_latents=True)
			_, w_c_t = self.pSp_c_net.forward(x_t, return_latents=True) 
			_, w_s_t = self.pSp_s_net.forward(x_t, return_latents=True) 

			rec_x_bg = self.pSp_c_net.forward(w_c_bg, input_code=True, randomize_noise=True, recon_modle=True)
			rec_x_t = self.pSp_c_net.forward(w_c_t + w_s_t, input_code=True, randomize_noise=True, recon_modle=True)	
			
			swap_cs_X = self.pSp_c_net.forward(w_c_bg + w_s_t, input_code=True, randomize_noise=True, recon_modle=True)
			swap_cs_Y = self.pSp_c_net.forward(w_c_t, input_code=True, randomize_noise=True, recon_modle=True)	

		idx = self.opts.special_idx
			
		row1 = torch.stack([x_bg[idx], rec_x_bg[idx], swap_cs_X[idx]], dim=0)
		row2 = torch.stack([x_t[idx], rec_x_t[idx], swap_cs_Y[idx]], dim=0)
		# Assuming row1 and row2 are [5, C, H, W]
		columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

		visualize_batch_grid(
			image_batches=columns,  # list of [2, C, H, W]
			titles=["Input", "Recon w-cs", "Swap w-cs"],
			save_path=f"{self.log_dir}/images/{name}/train_step_{self.global_step}.png"
		)              


	def train(self):
		self.pSp_c_net.train()
		self.# The code `pSp_s_net` is not valid Python code as it is just a string of characters. It does
		# not perform any specific action or operation in Python.
		pSp_s_net.train()
		while self.global_step < self.opts.max_steps:

			for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.train_bg_dataloader, self.train_t_dataloader)):
				
				self.optimizer.zero_grad()

				x_bg, _ = batch_bg
				x_t, _ = batch_t

				x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()

				_, w_c_bg = self.pSp_c_net.forward(x_bg, return_latents=True)
				_, w_s_bg = self.pSp_s_net.forward(x_bg, return_latents=True)
				_, w_c_t = self.pSp_c_net.forward(x_t, return_latents=True) 
				_, w_s_t = self.pSp_s_net.forward(x_t, return_latents=True) 

				rec_x_bg = self.pSp_c_net.forward(w_c_bg, input_code=True, randomize_noise=True, recon_modle=True)
				rec_x_t = self.pSp_c_net.forward(w_c_t + w_s_t, input_code=True, randomize_noise=True, recon_modle=True)	

				# Calculate loss
				loss_lat, loss_lat_dict = self.calc_latent_loss(w_s_bg)
				loss_img_bg, loss_img_dict_bg, _ = self.calc_image_loss(x_bg, rec_x_bg)
				loss_img_t, loss_img_dict_t, _ = self.calc_image_loss(x_t, rec_x_t)

				#train_loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

				loss = loss_lat + loss_img_bg + loss_img_t

				loss.backward()
				self.optimizer.step()

				# 12) Collect all sub‐losses into a single dict for logging
				train_loss_dict = {}
				train_loss_dict.update({f"{k}": v for k, v in loss_lat_dict.items()})
				train_loss_dict.update({f"{k}": v for k, v in loss_img_dict_bg.items()})
				train_loss_dict.update({f"{k}": v for k, v in loss_img_dict_t.items()})
				train_loss_dict["total_loss"] = loss.item()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 :
					self.pSp_c_net.eval()
					self.pSp_s_net.eval()	
					self.inference_special(self.special_batch_x, self.special_batch_y, name="val")
					self.inference_special(x_bg, x_t, name="train")
					self.pSp_c_net.train()
					self.pSp_s_net.train()					

				# if self.global_step < 300 and self.global_step % 25 == 0 :
				# 	self.print_metrics(train_loss_dict, prefix='train')
				if self.global_step % 25 == 0 and self.n_print < 100:
					self.write_metrics_to_txt(train_loss_dict, prefix='train', filename='loss_for_check.txt')
					self.n_print += 1

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss_sum'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss_sum']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(train_loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.pSp_c_net.eval()
		self.pSp_s_net.eval()

		agg_loss_lat_dict = []
		agg_loss_img_dict_bg = []
		agg_loss_img_dict_t = []

		# for batch_idx, batch in enumerate(self.test_dataloader):
		# 	x, y = batch
		for batch_idx, (batch_bg, batch_t) in enumerate(zip(self.test_bg_dataloader, self.test_t_dataloader)):
			
			x_bg, _ = batch_bg
			x_t, _ = batch_t

			with torch.no_grad():

				x_bg, x_t = x_bg.to(self.device).float(), x_t.to(self.device).float()

				_, w_c_bg = self.pSp_c_net.forward(x_bg, return_latents=True)
				_, w_s_bg = self.pSp_s_net.forward(x_bg, return_latents=True)
				_, w_c_t = self.pSp_c_net.forward(x_t, return_latents=True) 
				_, w_s_t = self.pSp_s_net.forward(x_t, return_latents=True) 

				rec_x_bg = self.pSp_c_net.forward(w_c_bg, input_code=True, randomize_noise=True, recon_modle=True)
				rec_x_t = self.pSp_c_net.forward(w_c_t + w_s_t, input_code=True, randomize_noise=True, recon_modle=True)	
				# Calculate loss
				_, loss_lat_dict = self.calc_latent_loss(w_s_bg)
				_, loss_img_dict_bg, _ = self.calc_image_loss(x_bg, rec_x_bg)
				_, loss_img_dict_t, _ = self.calc_image_loss(x_t, rec_x_t)

			agg_loss_lat_dict.append(loss_lat_dict)
			agg_loss_img_dict_bg.append(loss_img_dict_bg)
			agg_loss_img_dict_t.append(loss_img_dict_t)

			# For first step just do sanity test on small amount of data

			if batch_idx >= 10:
				break

		loss_lat_dict = train_utils.aggregate_loss_dict(agg_loss_lat_dict)
		loss_img_dict_bg = train_utils.aggregate_loss_dict(agg_loss_img_dict_bg)
		loss_img_dict_t = train_utils.aggregate_loss_dict(agg_loss_img_dict_t)

		loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

		#self.log_metrics(loss_dict, prefix='test')
		# self.print_metrics(loss_dict, prefix='test')

		self.pSp_c_net.train()
		self.pSp_s_net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				# if self.opts.use_wandb:
				# 	self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def write_metrics_to_txt(self, metrics_dict, prefix, filename):

		with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
			f.write(f'Metrics for {prefix}, Step - {self.global_step}')
			f.write(f'\n{metrics_dict}\n')		

	
	def configure_optimizers(self):
		# Combine parameters from both networks
		params = list(self.pSp_c_net.encoder.parameters()) + list(self.pSp_s_net.encoder.parameters())

		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)

		return optimizer

	def configure_datasets(self):

		# ---------------------------
		# 1. Validate dataset type
		# ---------------------------
		if self.opts.dataset_type not in data_configs.Medical_DATASETS:
			raise Exception(f'{self.opts.dataset_type} is not a valid dataset_type')

		print(f'Loading dataset for {self.opts.dataset_type}')

		# Copy base config
		dataset_args = dict(data_configs.Medical_DATASETS[self.opts.dataset_type])
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

		# ---------------------------
		# 2. Handle special case: bratsMultiMod
		# ---------------------------
		if self.opts.dataset_type == "bratsMultiMod":

			X_mod, Y_mod = self.opts.XY_mods      # e.g. ["t1n","t1c"]
			mode = self.opts.pairing              # "paired" or "unpaired"

			dset = data_configs.Medical_DATASETS['bratsMultiMod']

			# Build lookup keys
			train_X_key = f"train_X_{X_mod}_{mode}"
			test_X_key  = f"test_X_{X_mod}_{mode}"
			train_Y_key = f"train_Y_{Y_mod}_{mode}"
			test_Y_key  = f"test_Y_{Y_mod}_{mode}"

			# Override with the correct modality paths
			dataset_args['train_X_root'] = dset[train_X_key]
			dataset_args['test_X_root']  = dset[test_X_key]
			dataset_args['train_Y_root'] = dset[train_Y_key]
			dataset_args['test_Y_root']  = dset[test_Y_key]

		# ---------------------------
		# 3. Debug print
		# ---------------------------
		print("Using dataset paths:")
		for k in ["train_X_root", "train_Y_root", "test_X_root", "test_Y_root"]:
			print(f"  {k}: {dataset_args[k]}")

		# ---------------------------
		# 4. Load datasets
		# ---------------------------
		def load(path):
			return ImagesDataset_mednpy(
				npy_path=path,
				transform=transforms_dict[self.opts.data_transform],
				opts=self.opts
			)

		train_X_dataset = load(dataset_args['train_X_root'])
		train_Y_dataset = load(dataset_args['train_Y_root'])
		test_X_dataset  = load(dataset_args['test_X_root'])
		test_Y_dataset  = load(dataset_args['test_Y_root'])

		print(f"Number of training samples (X+Y): {len(train_X_dataset)*2}")
		print(f"Number of test samples (X+Y): {len(test_Y_dataset)*2}")

		return train_X_dataset, train_Y_dataset, test_X_dataset, test_Y_dataset
	
	def output_regularization(self, output, direction='element', reg_type='traditional', lambda_reg=0.1, a=3.7, gamma=3, threshold=0.0, epsilon=1e-8):

		abs_output = torch.abs(output)

		if reg_type == 'traditional':
			regularization_fn = torch.abs
		elif reg_type == 'logarithmic':
			regularization_fn = lambda x: torch.log(1 + torch.abs(x))
		elif reg_type == 'adaptive':
			regularization_fn = lambda x: (1.0 / (torch.abs(x) + epsilon)) * torch.abs(x)
		elif reg_type == 'thresholded':
			regularization_fn = lambda x: torch.where(torch.abs(x) < threshold, torch.abs(x), torch.tensor(0.0, device=x.device))
		elif reg_type == 'SCAD':
			# SCAD penalty function
			def scad_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1]
				mask2 = (abs_output > lambda_reg) & (abs_output <= a * lambda_reg)
				penalty[mask2] = (-x[mask2]**2 + 2 * a * lambda_reg * abs_output[mask2] - lambda_reg**2) / (2 * (a - 1))
				mask3 = abs_output > a * lambda_reg
				penalty[mask3] = (a + 1) * lambda_reg**2 / 2
				return penalty
			regularization_fn = scad_penalty
		elif reg_type == 'MCP':
			# MCP penalty function
			def mcp_penalty(x):
				penalty = torch.zeros_like(x)
				mask1 = abs_output <= gamma * lambda_reg
				penalty[mask1] = lambda_reg * abs_output[mask1] - (x[mask1]**2) / (2 * gamma)
				mask2 = abs_output > gamma * lambda_reg
				penalty[mask2] = (gamma * lambda_reg**2) / 2
				return penalty
			regularization_fn = mcp_penalty
		else:
			raise ValueError(f"Invalid regularization type: '{reg_type}'.")

		# Apply the selected regularization function based on direction
		if direction == 'row':
			return torch.sum(regularization_fn(output).sum(dim=2))
		elif direction == 'column':
			return torch.sum(regularization_fn(output).sum(dim=1))
		elif direction == 'element':
			return torch.sum(regularization_fn(output))
		else:
			raise ValueError(f"Invalid direction: '{direction}'.")


	def elastic_net_regularization(self, model, network='net_s', reg_type='all', alpha=0.5):
		"""
		Apply Elastic Net regularization on the specified layers of either `net_s` or `net_c` in the model, 
		handling both 2D and 3D weight matrices.

		Args:
			model (MappingNetwork_cs_sparsity): The model with the network to regularize.
			network (str): The network to apply Elastic Net regularization ('net_s' or 'net_c').
			reg_type (str): Type of weight regularization ('all' or 'last').
				- 'all': Applies Elastic Net to all EqualizedLinear layers in the specified network.
				- 'last': Applies Elastic Net only to the last EqualizedLinear layer in the specified network.
			alpha (float): The balance between L1 and L2 regularization. 0.0 corresponds to pure L2 (Ridge),
						1.0 to pure L1 (Lasso), and values in between apply both.

		Returns:
			torch.Tensor: Computed Elastic Net regularization penalty for the selected layers.
		"""
		elastic_net_penalty = 0.0

		# Select the target network based on the network argument
		target_network = getattr(model, network, None)
		if target_network is None:
			raise ValueError(f"Invalid network '{network}'. Use 'net_s' or 'net_c'.")

		# Get layers of the selected network
		layers = list(target_network.children())

		# Apply Elastic Net regularization based on reg_type
		if reg_type == 'all':
			# Apply Elastic Net to all EqualizedLinear layers in the target network
			for layer in layers:
				if isinstance(layer[0], EqualizedLinear):
					weights = layer[0].weight.weight
					l1_penalty = torch.sum(torch.abs(weights))
					l2_penalty = torch.sum(weights ** 2)
					elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		elif reg_type == 'last':
			# Apply Elastic Net only to the last EqualizedLinear layer in the target network
			last_layer = layers[-1][0]
			if isinstance(last_layer, EqualizedLinear):
				weights = last_layer.weight.weight
				l1_penalty = torch.sum(torch.abs(weights))
				l2_penalty = torch.sum(weights ** 2)
				elastic_net_penalty += alpha * l1_penalty + (1 - alpha) * l2_penalty
		else:
			raise ValueError(f"Invalid reg_type '{reg_type}'. Use 'all' or 'last'.")

		return elastic_net_penalty

	
	def calc_image_loss(self, x, x_hat):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# Calculate pSp id, rec, lpips losses for images
		if self.opts.id_lambda > 0:			
			loss_id, sim_improvement, id_logs = self.id_loss(x_hat, x, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda

		if self.opts.pix_lambda > 0:
			loss_pix = F.mse_loss(x, x_hat)
			loss_dict['loss_pix'] = float(loss_pix)
			loss += loss_pix * self.opts.pix_lambda

		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(x_hat, x)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda

		# SSIM Loss using pytorch-msssim
		if self.opts.ssim_lambda > 0:
			ssim_loss = 1 - ssim((x + 1) / 2, (x_hat + 1) / 2, data_range=1.0) # convert x and x_hat are in range [0,1]
			loss_dict['loss_ssim'] = float(ssim_loss)
			loss += ssim_loss * self.opts.ssim_lambda

		loss_dict['loss'] = float(loss)

		return loss, loss_dict, id_logs


	def calc_latent_loss(self, latent_bg_s):

		loss_dict = {}
		loss = 0.0

		if self.opts.sbg_lambda > 0:

			loss_silent_bg = F.mse_loss(latent_bg_s, torch.zeros_like(latent_bg_s).to(self.device))
			loss_dict['loss_silent_bg'] = float(loss_silent_bg)
			loss += loss_silent_bg * self.opts.sbg_lambda
			
		loss_dict['loss'] = float(loss)
		return loss, loss_dict		


	def merge_loss_dict(self, loss_lat_dict, loss_img_dict_bg, loss_img_dict_t):
		
		# loss_dict = loss_cs_dict | loss_dict_bg_pSp | loss_dict_t_pSp
		loss_dict = {}
		loss_dict['loss_lat'] = loss_lat_dict
		loss_dict['loss_img_bg'] = loss_img_dict_bg
		loss_dict['loss_img_t'] = loss_img_dict_t
		loss_dict['loss_sum'] = loss_lat_dict['loss'] + loss_img_dict_bg['loss'] + loss_img_dict_t['loss']
		
		return loss_dict
		

	# def log_metrics(self, metrics_dict, prefix):
	# 	for key, value in metrics_dict.items():
	# 		self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
	# 	if self.opts.use_wandb:
	# 		self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	# def parse_and_log_images(self, id_logs, x, y_pSp, y_ours, title, subscript=None, display_count=2):
	# 	im_data = []
	# 	for i in range(display_count):
	# 		cur_im_data = {
	# 			'input_face': common.log_input_image(x[i], self.opts),
	# 			'pSp_output_face': common.tensor2im(y_pSp[i]),
	# 			'Our_output_face': common.tensor2im(y_ours[i])
	# 		}
	# 		if id_logs is not None:
	# 			for key in id_logs[i]:
	# 				cur_im_data[key] = id_logs[i][key]
	# 		im_data.append(cur_im_data)
	# 	self.log_images(title, im_data=im_data, subscript=subscript)

	# def log_images(self, name, im_data, subscript=None, log_latest=False):
	# 	fig = common.vis_faces(im_data)
	# 	step = self.global_step
	# 	if log_latest:
	# 		step = 0
	# 	if subscript:
	# 		path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
	# 	else:
	# 		path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
	# 	os.makedirs(os.path.dirname(path), exist_ok=True)
	# 	fig.savefig(path)
	# 	plt.close(fig)

	def __get_save_dict(self):
		save_dict = {

			'pSp_c_net': self.pSp_c_net.state_dict(),
			'pSp_s_net': self.pSp_c_net.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizer'] = self.optimizer.state_dict()
			save_dict['best_val_loss'] = self.best_val_loss
				
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.pSp_c_net.latent_avg
		return save_dict
	
		
