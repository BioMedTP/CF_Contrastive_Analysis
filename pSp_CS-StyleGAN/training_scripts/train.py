"""
This file runs the main training/val loop
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import json
import sys
import pprint
import torch
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from configs.paths_config import model_paths
from argparse import Namespace

def main():

	previous_train_path = model_paths['previous_train_ckpt_path']
	previous_train_ckpt = None

	if previous_train_path is not None:
		print('start from previous training checkpoint...')
		previous_train_ckpt = torch.load(previous_train_path, map_location='cpu', weights_only=True)
		opts = load_opts_from_checkpoint(previous_train_ckpt)	
		if hasattr(opts, 'output_size'):
			opts.stylegan_size = opts.output_size
		opts.max_steps = 1000000
	else:
		opts = TrainOptions().parse()
		create_initial_experiment_dir(opts)

	if opts.exp_scheme=='baseline':
		print('using coach_c_s_baseline.py ...')
		from training.coach_c_s_baseline import Coach_csmlp

	elif opts.exp_scheme=='baseline_regular_D':
		print('using coach_c_s_baseline_regular_D.py ...')
		from training.coach_c_s_baseline_regular_D import Coach_csmlp
  
	elif opts.exp_scheme=='baseline_regular_DR':
		print('using coach_c_s_baseline_regular_DR.py ...')
		from training.coach_c_s_baseline_regular_DR import Coach_csmlp
	elif opts.exp_scheme=='baseline_regular_DR_cs1s2':
		print('using coach_c_s_baseline_regular_DR_cs1s2.py ...')
		from training.coach_c_s_baseline_regular_DR_cs1s2 import Coach_csmlp

	elif opts.exp_scheme=='baseline_sx':
		print('using coach_c_s_baseline_sx.py ...')
		from training.coach_c_s_baseline_sx import Coach_csmlp

	elif opts.exp_scheme=='baseline_c_s1s2':
		print('using coach_c_s1s2_baseline.py ...')
		from training.coach_c_s1s2_baseline import Coach_csmlp

	elif opts.exp_scheme=='c_s1_s2':
		print('using coach_csmlp_swap_losses.py ...')
		from training.coach_c_separate_s1_s2 import Coach_csmlp

	elif opts.exp_scheme=='glassesANDsmile':
		print('using coach_c_s_glasses_smile ...')
		from training.coach_c_s_glassesANDsmile import Coach_csmlp

	elif opts.exp_scheme=='glassesORsmile':
		print('using coach_c_s_glassesORsmile ...')
		from training.coach_c_s_glassesORsmile import Coach_csmlp

	elif opts.exp_scheme=='glassesVSsmile':
		print('using coach_c_s_glassesVSsmile.py ...')
		from training.coach_c_s_glassesVSsmile import Coach_csmlp

	elif opts.exp_scheme=='glassesVSsmile_s1s2':
		print('using coach_c_s1s2_glassesVSsmile.py ...')
		from training.coach_c_s1s2_glassesVSsmile import Coach_csmlp

	elif opts.exp_scheme=='glasses_smile':
		print('using coach_c_s_glasses_smile.py ...')
		from training.coach_c_s_glasses_smile import Coach_csmlp
		
	elif opts.exp_scheme=='c_s1s2':
		print('using coach_c_s1s2_baseline.py ...')
		from training.coach_c_s1s2_baseline import Coach_csmlp

	elif opts.exp_scheme=='med_ada':
		print('using coach_c_s_baseline_ada.py ...')
		from training.coach_c_s_baseline_ada import Coach_csmlp

	elif opts.exp_scheme=='med_ada_multi_salients':
		print('using coach_c_s_baseline_ada_multi_salients.py ...')
		from training.coach_c_s_baseline_ada_multi_salients import Coach_csmlp
	else:
		raise Exception('errors in loading coach file')
	
	coach = Coach_csmlp(opts, previous_train_ckpt)
	coach.train()

def load_opts_from_checkpoint(previous_train_ckpt):
	
	opts = previous_train_ckpt['opts']
	opts = Namespace(**opts)	
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	
	return opts


def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		shutil.rmtree(opts.exp_dir)
		#raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

if __name__ == '__main__':
	main()
