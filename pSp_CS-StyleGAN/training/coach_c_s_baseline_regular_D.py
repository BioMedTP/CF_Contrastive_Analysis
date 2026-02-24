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
from datasets.images_dataset import ImagesDataset, ImagesDataset_mednpy
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from models.mlp3D import MappingNetwork_cs_independent, EqualizedLinear
# from models.mlp2D import MappingNetwork_cs_independent, EqualizedLinear
import math
from utils.common_utils import visualize_batch_grid
# from utils.data_utils import get_special_images
from models.discriminator import CustomLatentClassifier
from argparse import Namespace
import pprint

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
		self.init_networks(self.opts.resume_ckpt_path)
		
		# -----------------------------
		# adversarial training states
		# -----------------------------
		self.last_D_acc = None
		self.last_D_loss = None
		# EMA for discriminator accuracy
		self.D_acc_ema = None
		self.ema_decay = 0.9   # 推荐 0.9 ~ 0.95（batch=4）

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.pSp_net.latent_avg is None:
			self.pSp_net.latent_avg = self.pSp_net.decoder.mean_latent(int(1e5))[0].detach()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:	
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		self.opt_D = torch.optim.Adam(self.domain_D.parameters(), lr=getattr(self.opts, "d_lr", 1e-4))

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
			self.test_bg_dataloader,
			self.test_t_dataloader
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

	def load_opts_from_checkpoint(self, previous_train_ckpt):

		opts = previous_train_ckpt['opts']
		opts = Namespace(**opts)	
		opts_dict = vars(opts)
		pprint.pprint(opts_dict)
		
		return opts

	def init_networks(self, cs_ckpt_path=None):
		
		self.pSp_net = pSp(self.opts).to(self.device).eval()
		self.cs_mlp_net = MappingNetwork_cs_independent(self.opts).to(self.device)	

		if cs_ckpt_path is not None:
			ckpt = torch.load(cs_ckpt_path, map_location='cpu', weights_only=True)
			# opts = self.load_opts_from_checkpoint(ckpt)
			# pSp_ckpt = prev_train_ckpt['state_dict_pSp']
			self.global_step = ckpt['global_step'] + 1
			print(f'Resuming training from step {self.global_step}')
			self.cs_mlp_net.load_state_dict(ckpt['state_dict_cs_enc'])
			
		# -----------------------------
		# Domain Discriminator (for C)
		# -----------------------------
		self.domain_D = CustomLatentClassifier(input_dim=self.opts.style_dim * 512, num_layers=self.opts.num_D_layers).to(self.device)

	def	init_additional_params(self, prev_train_ckpt):
		if self.opts.save_training_data and prev_train_ckpt is not None:
			self.best_val_loss = prev_train_ckpt['best_val_loss']
			if self.opts.keep_optimizer :
				self.optimizer.load_state_dict(prev_train_ckpt['optimizer'])


	def shift_with_avg(self, codes):
			# normalize with respect to the center of an average face
		shifted_codes = codes + self.pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
		return shifted_codes

	def center_with_avg(self, codes):
			# normalize with respect to the center of an average face

		shifted_codes = codes - self.pSp_net.latent_avg.repeat(codes.shape[0], 1, 1)
		return shifted_codes
	
	def inference_special(self, X, Y, name):

		with torch.no_grad():

			rec_pSp_X, w_X_pSp = self.pSp_net.forward(X, return_latents=True)
			rec_pSp_Y, w_Y_pSp = self.pSp_net.forward(Y, return_latents=True) 

			latent_X_c, latent_X_s = self.cs_mlp_net(w_X_pSp, zero_out_silent=self.opts.zero_out_silent_bg)
			latent_Y_c, latent_Y_s = self.cs_mlp_net(w_Y_pSp, zero_out_silent=self.opts.zero_out_silent_t) 

			rec_X = self.pSp_net.forward(latent_X_c, input_code=True, randomize_noise=True, recon_modle=True)
			rec_Y = self.pSp_net.forward(latent_Y_c + latent_Y_s, input_code=True, randomize_noise=True, recon_modle=True)        
			swap_cs_X = self.pSp_net.forward(latent_X_c + latent_Y_s, input_code=True, randomize_noise=True, recon_modle=True)
			swap_cs_Y = self.pSp_net.forward(latent_Y_c, input_code=True, randomize_noise=True, recon_modle=True)    

		# -----------------------------
		# select index (fixed or random)
		# -----------------------------
		if self.opts.special_idx >= 0:
			idx = self.opts.special_idx
		else:
			idx = random.randint(0, X.size(0) - 1)

		# prepare rows
		row1 = torch.stack([X[idx], rec_pSp_X[idx], rec_X[idx], swap_cs_X[idx]], dim=0)
		row2 = torch.stack([Y[idx], rec_pSp_Y[idx], rec_Y[idx], swap_cs_Y[idx]], dim=0)

		columns = [torch.stack([row1[i], row2[i]], dim=0) for i in range(len(row1))]

		visualize_batch_grid(image_batches=columns,
			titles=["Input", "Recon w", "Recon cs", "Swap cs"],
			save_path=f"{self.log_dir}/images/{name}/train_step_{self.global_step}_idx{idx}.png")


	def train_D(self, latent_X_c, latent_Y_c):
		self.domain_D.train()
		self.opt_D.zero_grad()

		logits_x = self.domain_D(latent_X_c.detach())  # [B, 1]
		logits_y = self.domain_D(latent_Y_c.detach())  # [B, 1]

		labels_x = torch.zeros_like(logits_x)
		labels_y = torch.ones_like(logits_y)

		loss_D = (
			F.binary_cross_entropy_with_logits(logits_x, labels_x) +
			F.binary_cross_entropy_with_logits(logits_y, labels_y)
		)

		loss_D.backward()
		self.opt_D.step()

		with torch.no_grad():
			pred_x = (torch.sigmoid(logits_x) < 0.5).float()
			pred_y = (torch.sigmoid(logits_y) > 0.5).float()
			acc_D = 0.5 * (pred_x.mean() + pred_y.mean())

		self.last_D_loss = loss_D.item()
		self.last_D_acc = acc_D.item()


		# update EMA of D accuracy
		if self.D_acc_ema is None:
			# initialize EMA on first valid step
			self.D_acc_ema = self.last_D_acc
		else:
			self.D_acc_ema = (
				self.ema_decay * self.D_acc_ema
				+ (1 - self.ema_decay) * self.last_D_acc
			)

	def update_D_lambda(self):
		if self.D_acc_ema is None:
			return

		# desired discriminator operating range
		target_low = 0.55
		target_high = 0.75

		if self.D_acc_ema > target_high:
			# D too strong → push C harder
			self.opts.D_lambda *= 1.05

		elif self.D_acc_ema < target_low:
			# D too weak → relax pressure
			self.opts.D_lambda *= 0.95

		# clamp for safety
		self.opts.D_lambda = float(
			np.clip(self.opts.D_lambda, 1e-4, 0.2)
		)

	def train(self):
		"""
		Main training loop with adversarial domain discriminator.
		"""

		self.cs_mlp_net.train()

		while self.global_step < self.opts.max_steps:
			for batch_idx, (batch_bg, batch_t) in enumerate(
				zip(self.train_bg_dataloader, self.train_t_dataloader)
			):

				# =====================================================
				# 0. Prepare data
				# =====================================================
				self.optimizer.zero_grad()
				X = batch_bg.to(self.device).float()
				Y = batch_t.to(self.device).float()

				# =====================================================
				# 1. pSp forward
				# =====================================================
				_, w_X_pSp = self.pSp_net.forward(X, return_latents=True)
				_, w_Y_pSp = self.pSp_net.forward(Y, return_latents=True)

				# =====================================================
				# 2. CS decomposition
				# =====================================================
				latent_X_c, latent_X_s = self.cs_mlp_net(
					w_X_pSp, zero_out_silent=self.opts.zero_out_silent_bg
				)
				latent_Y_c, latent_Y_s = self.cs_mlp_net(
					w_Y_pSp, zero_out_silent=self.opts.zero_out_silent_t
				)

				# =====================================================
				# 3. Train discriminator D
				# =====================================================
				if self.opts.D_lambda > 0 and self.global_step >= self.opts.D_start_step:
					self.train_D(latent_X_c, latent_Y_c)

				# =====================================================
				# 4. Reconstruction
				# =====================================================
				rec_X = self.pSp_net.forward(
					latent_X_c, input_code=True, randomize_noise=True, recon_modle=True
				)
				rec_Y = self.pSp_net.forward(
					latent_Y_c + latent_Y_s, input_code=True, randomize_noise=True, recon_modle=True
				)

				# =====================================================
				# 5. Losses
				# =====================================================
				loss_lat, loss_lat_dict = self.calc_latent_loss(
					latent_X_c, latent_X_s,
					latent_Y_c, latent_Y_s,
					w_X_pSp, w_Y_pSp
				)
				loss_img_X, loss_img_dict_X, _ = self.calc_image_loss(X, rec_X)
				loss_img_Y, loss_img_dict_Y, _ = self.calc_image_loss(Y, rec_Y)

				total_loss = loss_lat + loss_img_X + loss_img_Y

				# =====================================================
				# 6. Backprop
				# =====================================================
				total_loss.backward()
				self.optimizer.step()

				if self.global_step % 50 == 0 and self.global_step >= self.opts.adv_start_step:
					self.update_D_lambda()

				# =====================================================
				# 7. -------- LOGGING STRUCTURE (核心修改) --------
				# =====================================================

				# ---- losses (用于优化的) ----
				train_loss_dict = {
					"LOSS_lat": loss_lat_dict,
					"LOSS_img_X": loss_img_dict_X,
					"LOSS_img_Y": loss_img_dict_Y,
					"LOSS_sum": total_loss.item(),
					"Monitor": {
						"D_acc": self.last_D_acc,
						"D_acc_ema": self.D_acc_ema,
					}
				}

				with torch.no_grad():
					mu_x = latent_X_c.mean(dim=0)
					mu_y = latent_Y_c.mean(dim=0)
					var_x = latent_X_c.var(dim=0, unbiased=False)
					var_y = latent_Y_c.var(dim=0, unbiased=False)

					mean_dist = torch.norm(mu_x - mu_y)
					var_dist = torch.norm(var_x - var_y)

					kl_xy = 0.5 * torch.sum(
						torch.log((var_y + 1e-8) / (var_x + 1e-8)) +
						(var_x + (mu_x - mu_y) ** 2) / (var_y + 1e-8) - 1
					)
					kl_yx = 0.5 * torch.sum(
						torch.log((var_x + 1e-8) / (var_y + 1e-8)) +
						(var_y + (mu_x - mu_y) ** 2) / (var_x + 1e-8) - 1
					)

					train_loss_dict["Monitor"].update({
						"c_mean_dist": mean_dist.item(),
						"c_var_dist": var_dist.item(),
						"c_domain_gap": 0.5 * (kl_xy + kl_yx).item(),
					})

				# =====================================================
				# 8. Visualization
				# =====================================================
				if self.global_step % self.opts.image_interval == 0:
					self.cs_mlp_net.eval()
					self.inference_special(self.fixed_batch_X, self.fixed_batch_Y, name="val")
					self.inference_special(X, Y, name="train")
					self.cs_mlp_net.train()

				# =====================================================
				# 9. Text logging
				# =====================================================
				if self.global_step % 25 == 0 and self.n_print < 100:
					self.write_metrics_to_txt(
						train_loss_dict,
						prefix="train",
						filename="metrics_ini.txt"
					)
					self.n_print += 1
     
				if self.global_step % self.opts.log_interval == 0:
					self.write_metrics_to_txt(
						train_loss_dict,
						prefix="train",
						filename="log_metrics.txt"
					)
				# =====================================================
				# 10. Validation & checkpoint
				# =====================================================
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or \
				self.global_step == self.opts.max_steps:

					val_loss_dict = self.validate()
					if val_loss_dict and (
						self.best_val_loss is None or
						val_loss_dict["LOSS_sum"] < self.best_val_loss
					):
						self.best_val_loss = val_loss_dict["LOSS_sum"]
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or \
				self.global_step == self.opts.max_steps:

					self.checkpoint_me(
						val_loss_dict if val_loss_dict is not None else train_loss_dict,
						is_best=False
					)

				# =====================================================
				# 11. Step increment
				# =====================================================
				if self.global_step == self.opts.max_steps:
					print("OMG, finished training!")
					break

				self.global_step += 1


	def validate(self):
		self.cs_mlp_net.eval()
		self.domain_D.eval()

		# --- aggregate containers ---
		agg_loss_lat = []
		agg_loss_img_bg = []
		agg_loss_img_t = []

		agg_D_acc = []
		agg_mean_dist = []
		agg_var_dist = []
		agg_domain_gap = []

		for batch_idx, (batch_bg, batch_t) in enumerate(
			zip(self.test_bg_dataloader, self.test_t_dataloader)
		):
			with torch.no_grad():

				X = batch_bg.to(self.device).float()
				Y = batch_t.to(self.device).float()

				# -----------------------------
				# pSp encoding
				# -----------------------------
				_, w_X = self.pSp_net.forward(X, return_latents=True)
				_, w_Y = self.pSp_net.forward(Y, return_latents=True)

				# -----------------------------
				# CS decomposition
				# -----------------------------
				c_x, s_x = self.cs_mlp_net(
					w_X, zero_out_silent=self.opts.zero_out_silent_bg
				)
				c_y, s_y = self.cs_mlp_net(
					w_Y, zero_out_silent=self.opts.zero_out_silent_t
				)

				# -----------------------------
				# reconstruction
				# -----------------------------
				rec_X = self.pSp_net.forward(
					c_x, input_code=True, randomize_noise=True, recon_modle=True
				)
				rec_Y = self.pSp_net.forward(
					c_y + s_y, input_code=True, randomize_noise=True, recon_modle=True
				)

				# -----------------------------
				# losses (same as train)
				# -----------------------------
				_, loss_lat_dict = self.calc_latent_loss(
					c_x, s_x, c_y, s_y, w_X, w_Y
				)
				_, loss_img_bg_dict, _ = self.calc_image_loss(X, rec_X)
				_, loss_img_t_dict, _ = self.calc_image_loss(Y, rec_Y)

				agg_loss_lat.append(loss_lat_dict)
				agg_loss_img_bg.append(loss_img_bg_dict)
				agg_loss_img_t.append(loss_img_t_dict)

				# =====================================================
				# DOMAIN-INVARIANCE MONITORING (NO TRAINING EFFECT)
				# =====================================================

				# --- D accuracy ---
				logits_x = self.domain_D(c_x)
				logits_y = self.domain_D(c_y)

				pred_x = (torch.sigmoid(logits_x) < 0.5).float()
				pred_y = (torch.sigmoid(logits_y) > 0.5).float()
				D_acc = 0.5 * (pred_x.mean() + pred_y.mean())
				agg_D_acc.append(D_acc.item())

				# --- mean / second moment ---
				mu_x = c_x.mean(dim=0)
				mu_y = c_y.mean(dim=0)
				var_x = c_x.var(dim=0, unbiased=False)
				var_y = c_y.var(dim=0, unbiased=False)

				mean_dist = torch.norm(mu_x - mu_y)
				var_dist = torch.norm(var_x - var_y)

				agg_mean_dist.append(mean_dist.item())
				agg_var_dist.append(var_dist.item())

				# --- symmetric low-order KL (diagnostic only) ---
				kl_xy = 0.5 * torch.sum(
					torch.log((var_y + 1e-8) / (var_x + 1e-8)) +
					(var_x + (mu_x - mu_y) ** 2) / (var_y + 1e-8) - 1
				)
				kl_yx = 0.5 * torch.sum(
					torch.log((var_x + 1e-8) / (var_y + 1e-8)) +
					(var_y + (mu_x - mu_y) ** 2) / (var_x + 1e-8) - 1
				)

				agg_domain_gap.append(0.5 * (kl_xy + kl_yx).item())

			# --- short validation ---
			if batch_idx >= 10:
				break

		# =====================================================
		# aggregate losses
		# =====================================================
		loss_lat = train_utils.aggregate_loss_dict(agg_loss_lat)
		loss_img_X = train_utils.aggregate_loss_dict(agg_loss_img_bg)
		loss_img_Y = train_utils.aggregate_loss_dict(agg_loss_img_t)

		loss_sum = (
			loss_lat["loss"]
			+ loss_img_X["loss"]
			+ loss_img_Y["loss"]
		)

		# =====================================================
		# build unified validation dict (MATCH train schema)
		# =====================================================
		val_loss_dict = {
			"LOSS_lat": loss_lat,
			"LOSS_img_X": loss_img_X,
			"LOSS_img_Y": loss_img_Y,
			"LOSS_sum": float(loss_sum),
			"Monitor": {
				"val_D_acc": float(np.mean(agg_D_acc)),
				"val_c_mean_dist": float(np.mean(agg_mean_dist)),
				"val_c_var_dist": float(np.mean(agg_var_dist)),
				"val_c_domain_gap": float(np.mean(agg_domain_gap)),
			}
		}

		self.cs_mlp_net.train()
		self.domain_D.train()

		return val_loss_dict


	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)

		# log checkpoint event in a clean way
		tag = "best" if is_best else "ckpt"
		self.write_metrics_to_txt(
			loss_dict,
			prefix=tag,
			filename="val_metrics.txt"
		)



	def flatten_loss_dict(self, loss_dict):
		flat = {}
		for k, v in loss_dict.items():
			if isinstance(v, dict):
				for kk, vv in v.items():
					flat[f"{k}.{kk}"] = vv
			else:
				flat[k] = v
		return flat


	def configure_optimizers(self):
		params = list(self.cs_mlp_net.parameters())
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
			loss_dict['id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_id * self.opts.id_lambda

		# ----------------------------------------------------
		# L1 Loss
		# ----------------------------------------------------
		if self.opts.l1_lambda > 0:
			loss_l1 = F.l1_loss(rec, real)
			loss_dict['l1'] = float(loss_l1)
			loss += loss_l1 * self.opts.l1_lambda

		# ----------------------------------------------------
		# L2 Loss
		# ----------------------------------------------------
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(rec, real)
			loss_dict['l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda

		# ----------------------------------------------------
		# LPIPS Loss
		# ----------------------------------------------------
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(rec, real)
			loss_dict['lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda

		# ----------------------------------------------------
		# SSIM Loss
		# ----------------------------------------------------
		if self.opts.ssim_lambda > 0:
			loss_ssim = 1 - ssim(rec, real, data_range=1.0)
			loss_dict['ssim'] = float(loss_ssim)
			loss += loss_ssim * self.opts.ssim_lambda

		# ----------------------------------------------------
		# MS-SSIM Loss
		# ----------------------------------------------------
		if self.opts.msssim_lambda > 0:
			loss_msssim = 1 - ms_ssim(rec, real, data_range=1.0)
			loss_dict['msssim'] = float(loss_msssim)
			loss += loss_msssim * self.opts.msssim_lambda

		# ----------------------------------------------------
		# Total loss
		# ----------------------------------------------------
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs


	def adv_cxcy_loss(self, latent_cx, latent_cy):
		"""
		Adversarial loss on content code C.
		Supports:
			- 'confusion': domain confusion (target = 0.5)
			- 'flip': label-flipping saddle-point loss
			adv_mode = 'confusion':
				Encourages domain-invariant content representations
				by suppressing domain information in C.

			adv_mode = 'flip':
				Implements a saddle-point adversarial objective
				where the content encoder actively fools the discriminator.
				This mode may lead to less stable training.
		"""
		logits_x = self.domain_D(latent_cx)
		logits_y = self.domain_D(latent_cy)

		if self.opts.adv_mode == 'confusion':
			target = 0.5
			loss_x = F.binary_cross_entropy_with_logits(
				logits_x, torch.full_like(logits_x, target)
			)
			loss_y = F.binary_cross_entropy_with_logits(
				logits_y, torch.full_like(logits_y, target)
			)

		elif self.opts.adv_mode == 'flip':
			# generator tries to fool discriminator
			loss_x = F.binary_cross_entropy_with_logits(
				logits_x, torch.ones_like(logits_x)
			)
			loss_y = F.binary_cross_entropy_with_logits(
				logits_y, torch.zeros_like(logits_y)
			)

		else:
			raise ValueError(f"Unknown adv_mode: {self.opts.adv_mode}")

		return loss_x + loss_y
		
	
	def calc_latent_loss(self, latent_cx, latent_sx, latent_cy, latent_sy, w_x_pSp, w_y_pSp):

		loss_dict = {}
		loss = 0.0

		# --------------------------------------------------
		# 1. Silent X loss (background Sx -> 0)
		# --------------------------------------------------
		if self.opts.sbg_lambda > 0:
			loss_silent_x = F.mse_loss(latent_sx, torch.zeros_like(latent_sx))
			loss_dict['silent_x'] = float(loss_silent_x)
			loss += loss_silent_x * self.opts.sbg_lambda

		# --------------------------------------------------
		# 2. Latent reconstruction loss
		# --------------------------------------------------
		if self.opts.lat_recon_lambda > 0:
			loss_lat_recon_x = F.mse_loss(latent_cx, w_x_pSp)
			loss_lat_recon_y = F.mse_loss(latent_cy + latent_sy, w_y_pSp)

			loss_dict['lat_recon_x'] = float(loss_lat_recon_x)
			loss_dict['lat_recon_y'] = float(loss_lat_recon_y)

			loss += (loss_lat_recon_x + loss_lat_recon_y) * self.opts.lat_recon_lambda

		# --------------------------------------------------
		# 3. Adversarial domain loss (on C)
		# --------------------------------------------------
		if self.opts.D_lambda > 0 and self.global_step >= self.opts.adv_start_step:

			loss_adv_raw = self.adv_cxcy_loss(latent_cx, latent_cy)
			loss_dict['adv_c_raw'] = float(loss_adv_raw.detach())

			loss_adv = torch.clamp(loss_adv_raw, max=10.0)
			loss_dict['adv_c'] = float(loss_adv.detach())

			loss += self.opts.D_lambda * loss_adv

		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	# def write_metrics_to_txt(self, metrics_dict, prefix, filename):
	# 	flat = self.flatten_loss_dict(metrics_dict)

	# 	tokens = []
	# 	tokens.append(f"{prefix:5s} | step {self.global_step:6d}")

	# 	# sort keys for stable column order
	# 	for k in sorted(flat.keys()):
	# 		v = flat[k]
	# 		if isinstance(v, float):
	# 			tokens.append(f"{k}={v:6.6f}")
	# 		else:
	# 			tokens.append(f"{k}={v}")

	# 	line = " | ".join(tokens)

	# 	print(line)
	# 	with open(os.path.join(self.checkpoint_dir, filename), "a") as f:
	# 		f.write(line + "\n")
	# def write_metrics_to_txt(self, metrics_dict, prefix, filename):
	# 	with open(os.path.join(self.checkpoint_dir, filename), 'a') as f:
	# 		f.write(f'Metrics for {prefix}, Step - {self.global_step} | {metrics_dict}\n')

	# def write_metrics_to_txt(self, metrics_dict, prefix, filename):

	# 	def round_floats(obj):
	# 		"""
	# 		Recursively format all floats to 6 decimals.
	# 		"""
	# 		if isinstance(obj, dict):
	# 			return {k: round_floats(v) for k, v in obj.items()}
	# 		elif isinstance(obj, float):
	# 			return float(f"{obj:.6f}")
	# 		else:
	# 			return obj

	# 	metrics_fmt = round_floats(metrics_dict)

	# 	with open(os.path.join(self.checkpoint_dir, filename), "a") as f:
	# 		f.write(
	# 			f"Metrics for {prefix}, Step {self.global_step} | "
	# 			f"{metrics_fmt}\n"
	# 		)

	def write_metrics_to_txt(self, metrics_dict, prefix, filename):

		VAL_W =12
		PREC = 6
		STEP_W = 6

		class AlignedFloat(float):
			def __repr__(self):
				return f"{float(self):>{VAL_W}.{PREC}f}"

		def fmt(obj):
			if isinstance(obj, dict):
				return {k: fmt(v) for k, v in obj.items()}
			elif isinstance(obj, float):
				return AlignedFloat(obj)
			else:
				return obj

		metrics_fmt = fmt(metrics_dict)

		with open(os.path.join(self.checkpoint_dir, filename), "a") as f:
			f.write(
				f"Metrics for {prefix}, Step {self.global_step:>{STEP_W}d} | "
				f"{metrics_fmt}\n"
			)


	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)


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
	
		
