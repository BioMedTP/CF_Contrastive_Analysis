#!/bin/bash

#SBATCH --job-name=train_cls
#SBATCH --nodes=1
#SBATCH --partition=A40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_clsnew_out.txt
#SBATCH --error=./reports/train_clsnew_err.txt

source ./set_cuda_env.sh

python training_classifier/train.py \
--model_path=./results/adversarial_common/1st_alternate/cmlp150_with_cls50ep/lr1e-3/checkpoints/iteration_155000.pt \
--results_dir=./results/adversarial_common/1st_alternate/cmlp155k_classifier/lr1e-3/ \
--latent_hdf5_path=./results/adversarial_common/1st_alternate/cmlp155k_latent/lr1e-3/ \
--max_epochs=100 \
--save_interval=10 \
--lr=0.0001 \
--no_reproduce_latent

