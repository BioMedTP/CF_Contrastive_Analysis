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
from datasets.images_dataset import ImagesDataset_folder, ImagesDataset_mednpy
from criteria.lpips.lpips import LPIPS
from models.psp_ada import pSp
from training.ranger import Ranger
from models.mlp3D import MappingNetwork_cs_independent, MappingNetwork_c_s1_s2
# from models.mlp2D import MappingNetwork_cs_independent, EqualizedLinear
import math
from utils.common_utils import visualize_batch_grid
# from utils.data_utils import get_special_images
import random
from pytorch_msssim import ssim, ms_ssim	

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
		self.opts.style_dim = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
		
		self.seed_experiments(opts.seed)

		# Initialize network
		self.init_networks(previous_train_ckpt)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.pSp_net.latent_avg is None:
			self.pSp_net.latent_avg = self.pSp_net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:	
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize datasets
		self.train_X_dataset, self.train_Y_dataset, self.test_X_dataset, self.test_Y_dataset = self.configure_datasets()

		# ----------------------------
		# TRAIN LOADERS
		# ----------------------------
		if self.opts.dataset_type == 'bratsMultiMod': 
			train_shuffule = False
		else:
			train_shuffule = True

		self.train_X_dataloader = DataLoader(
			self.train_X_dataset,
			batch_size=self.opts.batch_size,
			shuffle=train_shuffule,
			num_workers=int(self.opts.workers),
			drop_last=True
		)

		self.train_Y_dataloader = DataLoader(
			self.train_Y_dataset,
			batch_size=self.opts.batch_size,
			shuffle=train_shuffule,
			num_workers=int(self.opts.workers),
			drop_last=True
		)

		# ----------------------------
		# TEST LOADERS
		# ----------------------------

		self.test_X_dataloader = DataLoader(
			self.test_X_dataset,
			batch_size=self.opts.test_batch_size,
			shuffle=False,
			num_workers=int(self.opts.test_workers),
			drop_last=True
		)

		self.test_Y_dataloader = DataLoader(
			self.test_Y_dataset,
			batch_size=self.opts.test_batch_size,
			shuffle=False,
			num_workers=int(self.opts.test_workers),
			drop_last=True
		)
		
		# self.special_batch_x, self.special_batch_y = get_special_images(self.device, size=self.opts.transform_size, datasets=self.opts.dataset_type)

		batch_X, batch_Y = get_paired_random_batch(
			self.test_X_dataloader,
			self.test_Y_dataloader
		)

		self.fixed_batch_X = batch_X.to(self.device)
		self.fixed_batch_Y = batch_Y.to(self.device)

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

		# # Save an initial checkpoint once on fresh runs, using the same format as iteration checkpoints
		# if previous_train_ckpt is None and getattr(self.opts, "save_on_init", True):
		# 	init_dir = os.path.join(self.opts.exp_dir, "initial_ckpt")   # matches your JSON
		# 	os.makedirs(init_dir, exist_ok=True)

		# 	init_path = os.path.join(init_dir, "initial_checkpoint.pt")  # or "initial.pt" if you prefer
		# 	if not os.path.exists(init_path):  # don’t overwrite if it already exists
		# 		init_save = self.__get_save_dict()   # same structure as iteration_*.pt
		# 		torch.save(init_save, init_path)
		# 		print(f"[init] Saved initial checkpoint to: {init_path}")

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
		
	def init_networks(self, prev_train_ckpt):
		pSp_ckpt = None
		cs_ckpt = None
		if prev_train_ckpt is not None:
			# pSp_ckpt = prev_train_ckpt['state_dict_pSp']
			cs_ckpt  = prev_train_ckpt['state_dict_cs_enc']	
			self.global_step = prev_train_ckpt['global_step'] + 1

		self.pSp_net = pSp(self.opts).to(self.device).eval()
		self.cs_mlp_net = MappingNetwork_c_s1_s2(self.opts).to(self.device)	

		if cs_ckpt is not None:
			print('Loading cs encoder from previous checkpoint...')
			self.cs_mlp_net.load_state_dict(cs_ckpt)
			print(f'Resuming training from step {self.global_step}')

	def	init_additional_params(self, prev_train_ckpt):
		if self.opts.save_training_data and prev_train_ckpt is not None:
			self.best_val_loss = prev_train_ckpt['best_val_loss']
			if self.opts.keep_optimizer :
				self.optimizer.load_state_dict(prev_train_ckpt['optimizer'])
	
	def inference_special(self, X, Y, name):

		with torch.no_grad():

			rec_pSp_X, w_X_pSp = self.pSp_net.forward(X)
			rec_pSp_Y, w_Y_pSp = self.pSp_net.forward(Y) 

			latent_X_c, latent_X_s1, latent_X_s2 = self.cs_mlp_net(w_X_pSp)
			latent_Y_c, latent_Y_s1, latent_Y_s2 = self.cs_mlp_net(w_Y_pSp) 

			rec_X = self.pSp_net.forward(latent_X_c + latent_X_s1, input_code=True, recon_model=True)
			rec_Y = self.pSp_net.forward(latent_Y_c + latent_Y_s2, input_code=True, recon_model=True)		
			swap_cs_X = self.pSp_net.forward(latent_X_c + latent_Y_s2, input_code=True, recon_model=True)
			swap_cs_Y = self.pSp_net.forward(latent_Y_c + latent_X_s1, input_code=True, recon_model=True)	

		# -----------------------------
		# select index (fixed or random)
		# -----------------------------
		if self.opts.special_idx >= 0:
			idx = self.opts.special_idx
		else:
			idx = random.randint(0, X.size(0) - 1)
			
		row1 = torch.stack([X[idx], rec_pSp_X[idx], rec_X[idx], swap_cs_X[idx]], dim=0)
		row2 = torch.stack([Y[idx], rec_pSp_Y[idx], rec_Y[idx], swap_cs_Y[idx]], dim=0)
		# Assuming row1 and row2 are [5, C, H, W]
		columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

		visualize_batch_grid(
			image_batches=columns,  # list of [2, C, H, W]
			titles=["Input", "Recon w", "Recon cs", "Swap cs"],
			save_path=f"{self.log_dir}/images/{name}/train_step_{self.global_step}.png"
		)              


	def train(self):
		#self.pSp_net.eval()
		self.cs_mlp_net.train()
		while self.global_step < self.opts.max_steps:

			for batch_idx, (batch_x, batch_y) in enumerate(zip(self.train_X_dataloader, self.train_Y_dataloader)):
				
				self.optimizer.zero_grad()

				# The code you provided is not complete, as it only shows the variable `X` and a comment `
				X = batch_x
				Y = batch_y

				X, Y = X.to(self.device).float(), Y.to(self.device).float()

				w_X_pSp = self.pSp_net.forward(X, encode_model=True)
				w_Y_pSp = self.pSp_net.forward(Y, encode_model=True) 

				# print('w_X_pSp shape', w_X_pSp.shape)
				latent_X_c, latent_X_s1, latent_X_s2 = self.cs_mlp_net(w_X_pSp)
				latent_Y_c, latent_Y_s1, latent_Y_s2 = self.cs_mlp_net(w_Y_pSp) 
    
				# print('latent_X_c shape', latent_X_c.shape)
    
				rec_X = self.pSp_net.forward(latent_X_c + latent_X_s1, input_code=True, recon_model=True)
				rec_Y = self.pSp_net.forward(latent_Y_c + latent_Y_s2, input_code=True, recon_model=True)	

				# Calculate loss
				loss_lat, loss_lat_dict = self.calc_latent_loss(latent_X_c, latent_X_s1, latent_X_s2, latent_Y_c, latent_Y_s1, latent_Y_s2, w_X_pSp, w_Y_pSp)
				loss_img_X, loss_img_dict_X, _ = self.calc_image_loss(X, rec_X)
				loss_img_Y, loss_img_dict_Y, _ = self.calc_image_loss(Y, rec_Y)

				#train_loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

				loss = loss_lat + loss_img_X + loss_img_Y

				loss.backward()
				self.optimizer.step()

				# 12) Collect all sub‐losses into a single dict for logging
				train_loss_dict = {}
				train_loss_dict.update({f"{k}": v for k, v in loss_lat_dict.items()})
				train_loss_dict.update({f"{k}": v for k, v in loss_img_dict_X.items()})
				train_loss_dict.update({f"{k}": v for k, v in loss_img_dict_Y.items()})
				train_loss_dict["total_loss"] = loss.item()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 :
					self.cs_mlp_net.eval()
					self.inference_special(self.fixed_batch_X, self.fixed_batch_Y, name="val")
					self.inference_special(X, Y, name="train")
					self.cs_mlp_net.train()					

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
		self.cs_mlp_net.eval()

		agg_loss_lat_dict = []
		agg_loss_img_dict_bg = []
		agg_loss_img_dict_t = []

		# for batch_idx, batch in enumerate(self.test_dataloader):
		# 	x, y = batch
		for batch_idx, (batch_x, batch_y) in enumerate(zip(self.test_X_dataloader, self.test_Y_dataloader)):
			
			X = batch_x
			Y = batch_y

			with torch.no_grad():

				X, Y = X.to(self.device).float(), Y.to(self.device).float()

				w_X_pSp = self.pSp_net.forward(X, encode_model=True)
				w_Y_pSp = self.pSp_net.forward(Y, encode_model=True) 

				# print('w_X_pSp shape', w_X_pSp.shape)
				latent_X_c, latent_X_s1, latent_X_s2 = self.cs_mlp_net(w_X_pSp)
				latent_Y_c, latent_Y_s1, latent_Y_s2 = self.cs_mlp_net(w_Y_pSp) 
    
				# print('latent_X_c shape', latent_X_c.shape)
    
				rec_X = self.pSp_net.forward(latent_X_c + latent_X_s1, input_code=True, recon_model=True)
				rec_Y = self.pSp_net.forward(latent_Y_c + latent_Y_s2, input_code=True, recon_model=True)	

				# Calculate loss
				_, loss_lat_dict = self.calc_latent_loss(latent_X_c, latent_X_s1, latent_X_s2, latent_Y_c, latent_Y_s1, latent_Y_s2, w_X_pSp, w_Y_pSp)
				_, loss_img_dict_X, _ = self.calc_image_loss(X, rec_X)
				_, loss_img_dict_Y, _ = self.calc_image_loss(Y, rec_Y)
    
			agg_loss_lat_dict.append(loss_lat_dict)
			agg_loss_img_dict_bg.append(loss_img_dict_X)
			agg_loss_img_dict_t.append(loss_img_dict_Y)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				#self.pSp_net.train()
				self.cs_mlp_net.train()
				return None  # Do not log, inaccurate in first batch
			if batch_idx >= 10:
				break

		loss_lat_dict = train_utils.aggregate_loss_dict(agg_loss_lat_dict)
		loss_img_dict_bg = train_utils.aggregate_loss_dict(agg_loss_img_dict_bg)
		loss_img_dict_t = train_utils.aggregate_loss_dict(agg_loss_img_dict_t)

		loss_dict = self.merge_loss_dict(loss_lat_dict, loss_img_dict_bg, loss_img_dict_t)

		#self.log_metrics(loss_dict, prefix='test')
		# self.print_metrics(loss_dict, prefix='test')

		self.cs_mlp_net.train()
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
		params = list(self.cs_mlp_net.parameters())
		# if self.opts.train_pSp_encoder:
		# 	params += list(self.pSp_net.encoder.parameters())
		# if self.opts.train_stylegan_decoder:
		# 	params += list(self.pSp_net.decoder.parameters())
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
	

	def calc_image_loss(self, real, rec):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# ----------------------------------------------------
		# Identity Loss (not used for medical, but kept here)
		# ----------------------------------------------------
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(rec, real, real)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_id * self.opts.id_lambda

		# ----------------------------------------------------
		# L1 Loss
		# ----------------------------------------------------
		if self.opts.l1_lambda > 0:
			loss_l1 = F.l1_loss(rec, real)
			loss_dict['loss_l1'] = float(loss_l1)
			loss += loss_l1 * self.opts.l1_lambda

		# ----------------------------------------------------
		# L2 Loss
		# ----------------------------------------------------
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(rec, real)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda

		# ----------------------------------------------------
		# LPIPS Loss
		# ----------------------------------------------------
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(rec, real)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda

		# ----------------------------------------------------
		# SSIM Loss
		# ----------------------------------------------------
		if self.opts.ssim_lambda > 0:
			loss_ssim = 1 - ssim(rec, real, data_range=1.0)
			loss_dict['loss_ssim'] = float(loss_ssim)
			loss += loss_ssim * self.opts.ssim_lambda

		# ----------------------------------------------------
		# MS-SSIM Loss
		# ----------------------------------------------------
		if self.opts.msssim_lambda > 0:
			loss_msssim = 1 - ms_ssim(rec, real, data_range=1.0)
			loss_dict['loss_msssim'] = float(loss_msssim)
			loss += loss_msssim * self.opts.msssim_lambda

		# ----------------------------------------------------
		# Total loss
		# ----------------------------------------------------
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs


	def calc_latent_loss(self, latent_cx, latent_sx1, latent_sx2, latent_cy, latent_sy1, latent_sy2, w_x_pSp, w_y_pSp):

		loss_dict = {}
		loss = 0.0

		if self.opts.sbg_lambda > 0:

			loss_silent_bg = F.mse_loss(latent_sx2, torch.zeros_like(latent_sx2))
			loss_silent_bg = F.mse_loss(latent_sy1, torch.zeros_like(latent_sy1))
		
			loss_dict['loss_silent_bg'] = float(loss_silent_bg)
			loss += loss_silent_bg * self.opts.sbg_lambda
			
		if self.opts.lat_recon_lambda > 0:
			loss_lat_recon_x = F.mse_loss(latent_cx + latent_sx1, w_x_pSp)
			loss_lat_recon_y = F.mse_loss(latent_cy + latent_sy2, w_y_pSp)
			loss_dict['loss_lat_recon_x'] = float(loss_lat_recon_x)
			loss_dict['loss_lat_recon_y'] = float(loss_lat_recon_y)
			loss += (loss_lat_recon_x + loss_lat_recon_y) * self.opts.lat_recon_lambda	

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
			#'state_dict_pSp': self.pSp_net.state_dict(),
			'state_dict_cs_enc': self.cs_mlp_net.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizer'] = self.optimizer.state_dict()
			save_dict['best_val_loss'] = self.best_val_loss
				
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.pSp_net.latent_avg
		return save_dict
	
		
