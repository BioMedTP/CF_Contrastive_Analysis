#!/bin/bash

#SBATCH --job-name=trainMI
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_MI_out.txt
#SBATCH --error=./reports/train_MI_err.txt

source ./set_cuda_env.sh

# python training_scripts/train.py \
# --exp_scheme=adverserial_mutual \
# --cmlp_checkpoint_path=results/adverserial_mutual/1st_alternate/cmlp130k/iteration_130000.pt \
# --cls_checkpoint_path=results/adverserial_mutual/1st_alternate/cmlp130k_disc/lr1e-5_global_2layers/checkpoints/model_epoch_100.pth \
# --exp_dir=results/adverserial_mutual/2nd_alternate/cmlp130k_mut_relu_lr0.001_100ep \
# --log_interval=100 \
# --val_interval=100 \
# --save_interval=100 \
# --max_steps=135000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --learning_rate=0.001 \
# --disc_type=global \
# --mi_loss_method=relu \
# --w_mi_lambda=1.0 \


python training_scripts/train.py \
--exp_scheme=adverserial_mutual \
--cmlp_checkpoint_path=results/adverserial_mutual/1st_alternate/cmlp130k/iteration_130000.pt \
--cls_checkpoint_path=results/adverserial_mutual/1st_alternate/cmlp130k_disc/lr1e-5_global_2layers/checkpoints/model_epoch_500.pth \
--exp_dir=results/adverserial_mutual/1st_alternate/cmlp_130k_with_cls500ep/lr1e-2 \
--log_interval=200 \
--val_interval=200 \
--save_interval=200 \
--max_steps=140000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--learning_rate=0.01 \
--disc_type=global \
--mi_loss_method=relu \
--w_mi_lambda=1.0 \
