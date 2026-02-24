#!/bin/bash

#SBATCH --job-name=train_mut
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_mut2_out.txt
#SBATCH --error=./reports/train_mut2_err.txt


source ./set_cuda_arch.sh

# python training_mutual/train.py \
# --model_path=./results/others/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
# --results_dir=./results/adverserial_mutual/1st_alternate/cmlp130k_disc/lr5e-5_global_2layers \
# --latent_hdf5_path=./results/adverserial_mutual/1st_alternate/cmlp130k_latent \
# --max_epochs=1000 \
# --save_interval=100 \
# --disc_type=global \
# --n_layer_disc=2 \
# --lr=5e-5 \
# --no_reproduce_latent \


python training_mutual_R/train.py \
--model_path=./results/adverserial_mutualR/1st_alternate/cmlp130k/iteration_130000.pt \
--results_dir=./results/adverserial_mutualR/1st_alternate/cmlp130k_mutualR \
--latent_hdf5_path=./results/adverserial_mutualR/1st_alternate/cmlp130k_latent \
--max_epochs=500 \
--save_interval=10 \
--n_layer_disc=2 \
--lr=1e-2 \
--n_c2s_layers=12 \
--no_reproduce_latent \
