#!/bin/bash

#SBATCH --job-name=OCTM
#SBATCH --nodes=1
#SBATCH --partition=A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --exclude=node03
#SBATCH --time=24:00:00
#SBATCH --output=./reports/oct.out
#SBATCH --error=./reports/oct.err

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN
echo "Changed directory to $(pwd)"

# rm -rf /home/ids/yuhe/.cache/torch_extensions
## --exclude=node[54-56]
# -----------------------------
# LOAD MODULES & ENV
# -----------------------------
module purge
module load gcc/11
HOSTNAME=$(hostname)
echo "Running on $HOSTNAME"

if [[ "$HOSTNAME" =~ node5[4-6] ]]; then
    echo "Detected node 54–56 → loading cuda/12.9"
    module load cuda/12.9
else
    echo "Other node → loading cuda/12.5"
    module load cuda/12.5
fi

echo "[Before conda] nvcc = $(which nvcc)"
nvcc --version || true

source ~/anaconda3/bin/activate styleGANenv

echo "[After conda] nvcc = $(which nvcc)"
nvcc --version || true

# -----------------------------
# FORCE CUDA PATH (关键)
# -----------------------------
unset CUDA_HOME CUDA_PATH
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "CUDA_HOME=$CUDA_HOME"
ls -l "$CUDA_HOME/bin/nvcc"

# -----------------------------
# PROJECT PATH
# -----------------------------
export PYTHONPATH=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN:$PYTHONPATH

# -----------------------------
# COMPILERS
# -----------------------------
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX
unset _GLIBCXX_USE_CXX11_ABI
export CXXFLAGS="-std=c++17"
export TORCH_EXTENSIONS_DIR=/home/ids/yuhe/.cache/torch_extensions

# arch list
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"


# ######################## brats HT ########################
# python training_scripts/train.py \
#     --exp_dir results/Regularization/TEST \
#     --exp_scheme baseline_regular_DR \
#     --dataset_type bratsHT_new \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_ht_Ros_resume/checkpoints/iteration_20000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 1 \
#     --log_interval 100 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 12 \
#     --optim_name admn \
#     --learning_rate 0.001 \
#     --special_idx 3 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 0.0 \
#     --l2_lambda 1.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8 \
#     --max_steps 200000 \
#     --num_D_layers 1 \
#     --D_lambda 0.05 \
#     --D_start_step 2000 \
#     --D_mode confusion \
#     --num_R_layers 2 \
#     --R_lambda 0.05 \
#     --R_start_step 2000 \
#     --R_mode RegrR \



#     #--no_train_Rx
#     #\
#     #DiscMI  RegrR --resume_ckpt  /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsHT_new_/checkpoints/iteration_10000.pt \


    # --pairing unpaired \
    # --XY_mods t1n t2f \
    # --train_shuffule \
    # --random_flip 


# ######################## bratsMultiMod cs1s2 with Reg. ########################
# python training_scripts/train.py \
#     --exp_dir results/BRATSMultiMod/t1n_t2f_Reg_cs1s2 \
#     --exp_scheme baseline_regular_DR_cs1s2 \
#     --dataset_type bratsMultiMod \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 10 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx -1 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 0.0 \
#     --l2_lambda 1.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8 \
#     --max_steps 200000 \
#     --num_D_layers 1 \
#     --D_lambda 0.05 \
#     --D_start_step 2000 \
#     --D_mode confusion \
#     --num_R_layers 2 \
#     --R_lambda 0.05 \
#     --R_start_step 2000 \
#     --R_mode RegrR \
#     --pairing paired \
#     --XY_mods t1n t2f \

    # --no_train_Rnull


#################################################################
#                      bloodmnist training
####################################################################
python training_scripts/train.py \
    --exp_dir results/bloodmnist/bloodmnist_x1y6_Reg_cs1s2_lr0.01 \
    --exp_scheme baseline_regular_DR_cs1s2 \
    --dataset_type bloodmnist_x1y6 \
    --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/bloodmnist_Ros/checkpoints/iteration_460000.pt \
    --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pt \
    --stylegan_size 256 \
    --data_transform rgb256 \
    --image_interval 2000 \
    --log_interval 2000 \
    --val_interval 2000 \
    --save_interval 10000 \
    --n_layers_mlp 10 \
    --optim_name admn \
    --learning_rate 0.01 \
    --special_idx 2 \
    --lat_recon_lambda 1.0 \
    --sbg_lambda 1.0 \
    --l1_lambda 0.0 \
    --l2_lambda 1.0 \
    --id_lambda 0.0 \
    --lpips_lambda 0.8 \
    --max_steps 200000 \
    --num_D_layers 1 \
    --D_lambda 0.05 \
    --D_start_step 2000 \
    --D_mode confusion \
    --num_R_layers 2 \
    --R_lambda 0.05 \
    --R_start_step 2000 \
    --R_mode RegrR \


# ##################################################################
# ##                   octmnist csmlp training                    ##
# ##################################################################
# python training_scripts/train.py \
#     --exp_dir results/OCTMNIST/octmnist_x3y0_reg_lr001 \
#     --exp_scheme baseline_regular_DR \
#     --dataset_type octmnist_x3y0 \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 10 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx -1 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 0.0 \
#     --l2_lambda 1.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8 \
#     --max_steps 200000 \
#     --num_D_layers 1 \
#     --D_lambda 0.05 \
#     --D_start_step 2000 \
#     --D_mode confusion \
#     --num_R_layers 2 \
#     --R_lambda 0.05 \
#     --R_start_step 2000 \
#     --R_mode RegrR \



# ######################## Camelyon16 csmlp ########################
# python training_scripts/train.py \
#     --exp_dir results/Camelyon16 \
#     --exp_scheme baseline \
#     --dataset_type Camelyon16 \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/Camelyon_Ros_resume/checkpoints/iteration_540000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/Camelyon_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 12 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx 2 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 1.0 \
#     --l2_lambda 0.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8

    
# # ######################## isic ########################
# python training_scripts/train.py \
#     --exp_dir results/Camelyon16 \
#     --exp_scheme baseline \
#     --dataset_type Camelyon16 \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/isic_Ros_resume/checkpoints/iteration_540000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/Camelyon_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 12 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx 2 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 1.0 \
#     --l2_lambda 0.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8


######################## bratsMultiMod csmlp ########################
# python training_scripts/train.py \
#     --exp_dir results/bratsMultiMod/t1n_t2w_cs \
#     --exp_scheme baseline_regular_DR \
#     --dataset_type bratsMultiMod \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 10 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx -1 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 0.0 \
#     --l2_lambda 1.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8 \
#     --pairing paired \
#     --XY_mods t1n t2w \
    # --train_shuffule \
    # --random_flip 



# ######################## bratsMultiMod csmlp ########################
# python training_scripts/train.py \
#     --exp_dir results/brats_ht/t1n_t2f_unpaired \
#     --exp_scheme baseline_c_s1s2 \
#     --dataset_type brats_ht \
#     --pSp_checkpoint_path /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt \
#     --stylegan_weights /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_rosinality.pt \
#     --stylegan_size 256 \
#     --data_transform rgb256 \
#     --image_interval 2000 \
#     --log_interval 2000 \
#     --val_interval 2000 \
#     --save_interval 10000 \
#     --n_layers_mlp 10 \
#     --optim_name admn \
#     --learning_rate 0.01 \
#     --special_idx -1 \
#     --lat_recon_lambda 1.0 \
#     --sbg_lambda 1.0 \
#     --l1_lambda 1.0 \
#     --l2_lambda 0.0 \
#     --id_lambda 0.0 \
#     --lpips_lambda 0.8 \
#     --pairing unpaired \
#     --XY_mods t1n t2f \
#     --random_flip

