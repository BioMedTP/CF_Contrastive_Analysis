#!/bin/bash

#SBATCH --job-name=train_D
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=./reports/train_mut2_out.txt
#SBATCH --error=./reports/train_mut2_err.txt

source ~/anaconda3/bin/activate styleGANenv

source ./set_cuda_env.sh

# Remove Torch extensions cache
rm -rf /home/ids/yuhe/.cache/torch_extensions
echo "Torch extensions cache cleared"

# Change directory to project folder
cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_encoder_constructive
echo "Changed directory to $(pwd)"





# # train discriminator 
# python main/train_D.py --config=main/configs_D.json --experiment=CA_with_RD

# ## train reconstructor 
# python main/train_R.py --config=main/configs_R.json --experiment=CA_with_RD

 
# python training_scripts/train.py \
# --dataset_type=ffhq_glasses \
# --exp_scheme=improved_loss \
# --cmlp_checkpoint_path=./results/baseline/ckpt_s0.1/iteration_147200.pt \
# --disc_checkpoint_path=./results/CA_with_RD/discriminators/iter147200/checkpoints/model_epoch_500.pth \
# --c2s_checkpoint_path=./results/CA_with_RD/reconstructors/iter147200/param_v3/checkpoints/model_epoch_250.pth \
# --exp_dir=./results/CA_with_RD/CA_from_147200/param_v3_250ep_c1.0_s0.01  \
# --log_interval=100 \
# --val_interval=100 \
# --save_interval=100 \
# --max_steps=150000 \
# --mlp_norm_type=nodim \
# --n_c2s_layers=6 \
# --c2s_net_type=c2smlp \
# --learning_rate=0.001 \
# --w_cls_lambda=1.0 \
# --w_recon_lambda=1.0 \
# --lambda_s=0.01 \
# --num_D_layers=3 \
# --normalize \


# python training_scripts/train.py \
# --dataset_type=ffhq_glasses \
# --exp_scheme=mult_optims \
# --cmlp_checkpoint_path=./results/CA_with_R/CA_from_130k/config1/checkpoints/iteration_140000.pt \
# --disc_checkpoint_path=./results/baseline/discriminator/checkpoints/model_epoch_200.pth \
# --c2s_checkpoint_path=./results/baseline/reconstructor/strong/lr1e-5/checkpoints/model_epoch_500.pth \
# --exp_dir=./results/CA_with_R/CA_from_140k/config5_strong_lrCA0.001_lrR0.002 \
# --log_interval=100 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=150000 \
# --mlp_norm_type=nodim \
# --n_c2s_layers=12 \
# --c2s_net_type=strong \
# --learning_rate=0.001 \
# --w_cls_lambda=0.0 \
# --w_recon_lambda=1.0 \
# --lr_CA=0.001 \
# --lr_R=0.002



# # for discriminator
# python training_scripts/train.py \
# --dataset_type=ffhq_glasses \
# --exp_scheme=improved_loss \
# --cmlp_checkpoint_path=./results/CA_with_D/CA_from_145k/config1/checkpoints/iteration_145600.pt \
# --disc_checkpoint_path=./results/CA_with_D/discriminators/for_iter145600/checkpoints/model_epoch_500.pth \
# --c2s_checkpoint_path=./results/baseline/reconstructor/c2smlp/checkpoints/model_epoch_140.pth \
# --exp_dir=./results/CA_with_D/CA_from_145600/config1 \
# --log_interval=200 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=160000 \
# --mlp_norm_type=nodim \
# --n_c2s_layers=12 \
# --c2s_net_type=c2smlp \
# --learning_rate=0.001 \
# --w_cls_lambda=1.0 \
# --w_recon_lambda=0.0 \
# --lambda_s=0.2 \