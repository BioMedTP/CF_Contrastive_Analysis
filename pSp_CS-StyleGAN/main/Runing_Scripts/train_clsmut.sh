#!/bin/bash

#SBATCH --job-name=trainMI
#SBATCH --nodes=1
#SBATCH --partition=L40S
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_MI_out.txt
#SBATCH --error=./reports/train_MI_err.txt

source ./set_cuda_env.sh


python training_scripts/train.py \
--exp_scheme=cls_mut \
--cmlp_checkpoint_path=results/adversarial_common/1st_alternate/cmlp130k/iteration_130000.pt \
--cls_checkpoint_path=results/adversarial_common/1st_alternate/cmlp130k_classifier/checkpoints/model_epoch_50.pth \
--mut_checkpoint_path=results/adverserial_mutual/2nd_alternate/old/cmlp130k_mut_bce_lr0.001/checkpoints/iteration_130800.pt \
--exp_dir=results/cls_with_mut/ \
--log_interval=200 \
--val_interval=200 \
--save_interval=200 \
--max_steps=140000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--learning_rate=0.001 \
--disc_type=global \
--mi_loss_method=bce \
--n_mut_layers=2 \
--w_cls_lambda=1.0 \
--w_mut_lambda=1.0 \
