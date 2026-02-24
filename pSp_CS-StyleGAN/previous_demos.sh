#!/bin/bash

#SBATCH --job-name=isic_cs
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --exclude=node[54-56]
#SBATCH --time=24:00:00
#SBATCH --output=./reports/isic_cs.out
#SBATCH --error=./reports/isic_cs.err

# Optional: Manually configure CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set Compute Capability for A100

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN

echo "Changed directory to $(pwd)"

module purge
module load gcc/11
module load cuda/12.5
source ~/anaconda3/bin/activate styleGANenv
# # after activating your env and loading modules
# rm -rf ~/.cache/torch_extensions ~/.cache/torch/inductor
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX
unset _GLIBCXX_USE_CXX11_ABI
export CXXFLAGS="-std=c++17"
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

# python training_scripts/train.py

# ### for TEST
# python training_scripts/train.py \
# --exp_dir=results/ffhq_gender_cs1s2/10layers_lr0.01_s1s2_id0.4 \
# --exp_scheme=c_s1_s2 \
# --dataset_type=ffhq_gender \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=2000 \
# --log_interval=2000 \
# --val_interval=2000 \
# --save_interval=10000 \
# --n_layers_mlp=10 \
# --optim_name=admn \
# --learning_rate=0.01 \
# --special_idx=1 \
# # --id_lambda=0.2 


######################## med_ada_isic csmlp ########################
python training_scripts/train.py \
    --exp_dir results/med_ada_isic/0.001_10layers \
    --exp_scheme med_ada \
    --dataset_type isic \
    --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/isic_ada/checkpoints/iteration_580000.pt \
    --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/ISIC/00002-isic256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-004032.pkl \
    --stylegan_size 256 \
    --data_transform rgb256 \
    --image_interval 2000 \
    --log_interval 2000 \
    --val_interval 2000 \
    --save_interval 10000 \
    --n_layers_mlp 10 \
    --optim_name=admn \
    --learning_rate 0.001 \
    --special_idx 2 \
    --lat_recon_lambda 1.0 \
    --sbg_lambda 1.0 \
    --l1_lambda 0.0 \
    --l2_lambda 1.0 \
    --id_lambda 0.0 \
    --lpips_lambda 0.8 \


######################## med_ada_isic csmlp ########################
python training_scripts/train.py \
    --exp_dir results/med_ada_isic/0.001_10layers \
    --exp_scheme med_ada \
    --dataset_type isic \
    --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/isic_ada/checkpoints/iteration_580000.pt \
    --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/ISIC/00002-isic256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-004032.pkl \
    --stylegan_size 256 \
    --data_transform rgb256 \
    --image_interval 2000 \
    --log_interval 2000 \
    --val_interval 2000 \
    --save_interval 10000 \
    --n_layers_mlp 10 \
    --optim_name=admn \
    --learning_rate 0.001 \
    --special_idx 2 \
    --lat_recon_lambda 1.0 \
    --sbg_lambda 1.0 \
    --l1_lambda 0.0 \
    --l2_lambda 1.0 \
    --id_lambda 0.0 \
    --lpips_lambda 0.8 \

# python training_scripts/train.py \
#     --exp_dir results/med_ada_isic \
#     --exp_scheme med_ada \
#     --dataset_type ffhq \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/pSp_models/psp_ffhq_encode.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_ada/ffhq.pkl \
#     --stylegan_size 1024 \
#     --data_transform rgb256 \
#     --image_interval 50 \
#     --log_interval 50 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 12 \
#     --optim_name=admn \
#     --learning_rate 0.01 \
#     --special_idx 2 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 1.0 \
#     --l2_lambda 1.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8

# ### for TEST
# python training_scripts/train.py \
# --exp_dir=results/ffhq_glasses_flip \
# --exp_scheme=c_s1_s2 \
# --dataset_type=ffhq_glasses \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=2000 \
# --stylegan_size=1024 \
# --log_interval=2000 \
# --val_interval=2000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --learning_rate=0.01 \
# --special_idx=0 \

# #### AFHQ ####
# python training_scripts/train.py \
# --exp_dir=results/AFHQ/dog_cat_flip2 \
# --exp_scheme=baseline_flip \
# --dataset_type=afhqv2 \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_afhqv2.pt \
# --stylegan_size=512 \
# --image_interval=2000 \
# --log_interval=2000 \
# --val_interval=2000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --learning_rate=0.01 \
# --special_idx=0 \

# #### celebaHQ ####
# python training_scripts/train.py \
# --exp_dir=results/celebaHQ/celebaHQ_smile \
# --exp_scheme=baseline \
# --dataset_type=celebaHQ_smile \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_celebahq_styleganffhq.pt \
# --stylegan_size=1024 \
# --image_interval=5000 \
# --log_interval=2000 \
# --val_interval=2000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --learning_rate=0.01 \
# --special_idx=0 \

# # rm -rf /lustre/fswork/projects/rech/hht/usv51hl/.cache/torch_extensions
# ### for TEST
# python training_scripts/train.py \
# --exp_dir=results/cs1s2_glasses_gender \
# --exp_scheme=cs1s2_glasses_gender \
# --dataset_type=ffhq_glasses \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=2000 \
# --log_interval=2000 \
# --val_interval=2000 \
# --save_interval=5000 \
# --transform_size=256 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --batch_size=2 \
# --seed=99 \
# --mlp_norm_type=nodim \


# ### keeep sx
# python training_scripts/train.py \
# --exp_scheme=c_keep_sx \
# --exp_dir=results/cmlp_keep_sx \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses

# s1s2
# python training_scripts/train.py \
# --exp_scheme=c_s1s2 \
# --exp_dir=results/c_s1s2/ffhq_lr0.001 \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --learning_rate=1e-3 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses


# python training_scripts/train.py \
# --exp_scheme=c_disc_s1s2 \
# --dataset_type=afhqv2 \
# --exp_dir=results/c_disc_s1s2/afhqv2/lr_1e-2 \
# --output_size=512 \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/afhqv2/ffhq_params/checkpoints/iteration_300000.pt \
# --stylegan_weights=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2-afhqv2-512x512.pt \
# --image_interval=2000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --learning_rate=1e-2 \
# --id_lambda=0.2 \

## s1s2  dog cat
# python training_scripts/train.py \
# --exp_scheme=c_sep_disc_s1s2 \
# --dataset_type=afhqv2 \
# --exp_dir=results/c_sep_disc_s1s2/afhqv2/confusion_mse \
# --output_size=512 \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/afhqv2/ffhq_params/checkpoints/iteration_300000.pt \
# --stylegan_weights=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2-afhqv2-512x512.pt \
# --image_interval=2000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=10 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --learning_rate=1e-3 \
# --id_lambda=0.2 \
# --disc_n_layers=2 \
# --disc_lr=0.0001 \
# --adv_lambda=0.1 \
# --train_disc_interval=1000 \
# --log_adv_interval=100 \
# --max_disc_epochs=2 \
# --adv_loss_type=confusion_mse \



#confusion_mse  non_satu   JS_confusion
# --lat_recon_lambda=1.0 \
# c_sep-s1s2
# python training_scripts/train.py \
# --exp_scheme=w_wpace_2D \
# --dataset_type=ffhq_glasses \
# --exp_dir=results/W_space_2Dlayers \
# --output_size=1024 \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pixel2style2pixel_orn/results/W_space/checkpoints/iteration_360000.pt \
# --stylegan_weights=../pretrained_models/pSp_models/stylegan2-ffhq1024.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=10 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=dim1 \
# --learn_in_w=True \
# --encoder_type=BackboneEncoderUsingLastLayerIntoW

# python training_scripts/train.py \
# --dataset_type=BraTS_tumor \
# --exp_scheme=baseline \
# --output_size=256 \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pixel2style2pixel_orn/results/BraTS_resume_880k/base/checkpoints/iteration_540000.pt \
# --stylegan_weights=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/stylegan2-rosinality/results/3gpus_bs16/checkpoints/880000.pt \
# --exp_dir=results/csmlp_BraTS/540k \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --id_lambda=0.0 \
# --ssim_lambda=0.3 \
# --pix_lambda=1.0 \
# --lpips_lambda=1.0 \
# --w_dist_lambda=0.8


# python training_scripts/train.py \
# --dataset_type=test \
# --exp_scheme=e4e_lsun \
# --stylegan_size=256 \
# --e4e_checkpoint_path=../pretrained_models/e4e_models/e4e_church_encode.pt \
# --exp_dir=results/lsun_church/weak_w_dist \
# --image_interval=5000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim  \
# --id_lambda=0.2 \
# --ssim_lambda=0.0 \
# --pix_lambda=2.0 \
# --lpips_lambda=1.5 \
# --w_dist_lambda=0.5

