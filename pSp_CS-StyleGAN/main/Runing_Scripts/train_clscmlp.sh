#!/bin/bash

#SBATCH --job-name=clscmlp
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_MI2_out.txt
#SBATCH --error=./reports/train_MI2_err.txt


source ./set_cuda_env.sh

# python training_scripts/train.py \
# --exp_scheme=adverserial_common \
# --cmlp_checkpoint_path=results/adversarial_common/1st_alternate/cmlp130k/iteration_130000.pt \
# --cls_checkpoint_path=results/adversarial_common/1st_alternate/cmlp130k_classifier/checkpoints/model_epoch_50.pth \
# --exp_dir=results/adversarial_common/1st_alternate/cmlp130k_with_cls50ep/lr1e-2 \
# --log_interval=100 \
# --val_interval=100 \
# --save_interval=100 \
# --max_steps=140000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --learning_rate=1e-2 \
# --w_cls_lambda=1.0 \

python training_scripts/train.py \
--exp_scheme=adverserial_common \
--cmlp_checkpoint_path=./results/adversarial_common/1st_alternate/cmlp150_with_cls50ep/lr1e-3/checkpoints/iteration_155000.pt \
--cls_checkpoint_path=./results/adversarial_common/1st_alternate/cmlp155k_classifier/lr1e-3/checkpoints/model_epoch_100.pth \
--exp_dir=./results/adversarial_common/1st_alternate/cmlp155_with_cls50ep/lr1e-3 \
--log_interval=200 \
--val_interval=200 \
--save_interval=200 \
--max_steps=165000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses \
--learning_rate=1e-3 \
--w_cls_lambda=1.0 \


# python training_scripts/train.py \
# --exp_scheme=adverserial_common \
# --cmlp_checkpoint_path=results/adversarial_common/2nd_alternate/cmlp130k_cls_lr1e-2/checkpoints/iteration_130400.pt \
# --cls_checkpoint_path=results/adversarial_common/2nd_alternate/cmlp130k_cls_lr1e-2_classifier/checkpoints/model_epoch_10.pth \
# --exp_dir=results/adversarial_common/3rd_alternate/cmlp130400_cls_lr1e-2 \
# --log_interval=100 \
# --val_interval=100 \
# --save_interval=100 \
# --max_steps=132000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --learning_rate=0.01 \
# --w_cls_lambda=1.0 \


# python training_scripts/train.py \
# --exp_scheme=adverserial_common \
# --cmlp_checkpoint_path=results/adversarial_common/2nd_alternate/cmlp130k_cls_lr1e-2/checkpoints/iteration_130400.pt \
# --cls_checkpoint_path=results/adversarial_common/2nd_alternate/cmlp130k_cls_lr1e-2_classifier/checkpoints/model_epoch_10.pth \
# --exp_dir=results/adversarial_common/3rd_alternate/cmlp130400_cls_lr1e-3_10ep \
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
# --w_cls_lambda=1.0 \



