#!/bin/bash

#SBATCH --job-name=multimod
#SBATCH --nodes=1
#SBATCH --partition=A40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --exclude=node[03,05]
#SBATCH --output=./reports/multimod.out
#SBATCH --error=./reports/multimod.err
# Optional: Manually configure CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set Compute Capability for A100

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS
echo "Changed directory to $(pwd)"

#rm -rf /home/ids/yuhe/.cache/torch_extensions

# -----------------------------+

# LOAD MODULES & ENV
# -----------------------------
module purge
module load gcc/11
module load cuda/12.5

# 关键：先确认 module 提供的 nvcc 在哪
echo "[Before conda] nvcc = $(which nvcc)"
nvcc --version || true

# conda
source ~/anaconda3/bin/activate styleGANenv

# 关键：conda 可能改 PATH，所以激活后再确认一次
echo "[After conda] nvcc = $(which nvcc)"
nvcc --version || true

# -----------------------------
# FORCE CUDA PATH (最关键的修改)
# -----------------------------
unset CUDA_HOME CUDA_PATH
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "CUDA_HOME=$CUDA_HOME"
ls -l "$CUDA_HOME/bin/nvcc"

# -----------------------------
# YOUR ORIGINAL EXPORTS
# -----------------------------
export PYTHONPATH=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS:$PYTHONPATH
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$CXX
unset _GLIBCXX_USE_CXX11_ABI
export CXXFLAGS="-std=c++17"

export TORCH_EXTENSIONS_DIR=/home/ids/yuhe/.cache/torch_extensions

export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"


python inference_ipynb/inf_segmentation.py

#rm -rf /home/ids/yuhe/.cache/torch_extensions

# # python brats_ht_current.py
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train_new/OCTMNIST/octmnist_x1y2_reg_cs \
#     train.train_runner=fse_editor_cs \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=256 \
#     model.channel_multiplier=1 \
#     data.special_idx=-1 \
#     data.dataset=bloodmnist_x1y6 \
#     data.transform=face_256 \
#     model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x1y2_reg_cs/checkpoints/iteration_50000.pt \
    # model.checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train_new/bloodmnist/bloodmnist_x1y3_Reg_000/iteration_320000.pt


# /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x1y2_reg_cs/checkpoints/iteration_70000.pt

#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_cs_Reg/checkpoints/iteration_30000.pt

#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x1y2_reg_cs/checkpoints/iteration_50000.pt

#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x1y2_reg_cs1s2/checkpoints/iteration_50000.pt
#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/BRATSMultiMod/t1n_t2w_Reg_cs1s2/checkpoints/iteration_60000.pt
# model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/BRATSMultiMod/t1n_t2f_Reg_cs1s2/checkpoints/iteration_30000.pt
#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y3_Reg_cs1s2_lr0.01/checkpoints/iteration_60000.pt
#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y3_Reg/checkpoints/iteration_20000.pt


#model.pSp_cs_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2w_cs_Reg/checkpoints/iteration_30000.pt


# /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_D_start_step1000/checkpoints/iteration_100000.pt  
# /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/adv_DR_DiscMI/Exp4_startstep2000_Rlr0.05_Rlayer2_MLPlr0.001/checkpoints/iteration_110000.pt 
# /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Regularization/wo_regular/checkpoints/iteration_110000.pt
#/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y3_Reg_cs1s2_lr0.01/checkpoints/iteration_60000.pt

# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/octmnist/octmnist_x3y012_3k/ \
#     train.train_runner=fse_editor_cs \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=256 \
#     model.channel_multiplier=1 \
#     data.special_idx=-1 \
#     data.dataset=octmnist_x3y012 \
#     data.transform=face_256 \
#     #model.checkpoint_path=./experiments/fse_cs_editor_train/octmnist/octmnist_x3y2_000/iteration_370000.pt \

# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/octmnist/octmnist_x3y012_3k/ \
#     train.train_runner=fse_editor_cs \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=256 \
#     model.channel_multiplier=1 \
#     data.special_idx=-1 \
#     data.dataset=octmnist_x3y012 \
#     data.transform=face_256 \
#     #model.checkpoint_path=./experiments/fse_cs_editor_train/octmnist/octmnist_x3y2_000/iteration_370000.pt \



# ############## CelebAHQ smile ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=celebahq_smile \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/celebaHQ/celebahq_smile \
#     methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
#     train.start_step=330000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=2 \
#     model.w_space_encoder=pSp \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/celebaHQ/celebahq_smile_000/iteration_330000.pt \
    #SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_gender_001/iteration_340000.pt

#     ############## church ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=lsun_church \
#     data.transform=face_256 \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/lsun_church_9k/ \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/sfe_inverter_church_165k.pt \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=3 \
#     model.w_space_encoder=e4e \
#     model.stylegan_size=256 \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_gender_001/iteration_340000.pt \


#     ############## Brats ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=brats_edit \
#     data.transform=face_256 \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/brats/edit/ \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/brats_inverter.pt \
#     train.start_step=300000 \
#     train.direction=two_directions \
#     train.log_step=20 \
#     train.val_step=20 \
#     train.checkpoint_step=20 \
#     data.special_idx=2 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=256 \
#     model.checkpoint_path=../../pretrained_models/sfe/refined_sfe_brats_170k.pt \

#     ############## ffhq_glassesvssmile editor ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=ffhq_glassesvssmile \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k \
#     methods_args.fse_full.inverter_pth=./pretrained_models/sfe_inverter_light.pt \
#     train.train_runner=fse_editor_cs1s2 \
#     train.start_step=360000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=0 \
#     model.w_space_encoder=pSp \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k_001/iteration_360000.pt \


#     ############## AFHQ editor ##############
# python scripts/train.py \
#     exp.exp_dir=./experiments/ \
#     data.dataset=afhq_cat_dog \
#     exp.config_dir=configs \
#     exp.config=fse_cs_editor_train.yaml \
#     exp.name=fse_cs_editor_train/pSp_encoder/AFHQ/S1S2 \
#     methods_args.fse_full.inverter_pth=../../pretrained_models/sfe/afhq_inverter.pt \
#     train.train_runner=fse_editor_cs1s2 \
#     train.start_step=320000 \
#     train.direction=two_directions \
#     train.log_step=2000 \
#     train.val_step=2000 \
#     train.checkpoint_step=10000 \
#     data.special_idx=0 \
#     model.w_space_encoder=pSp \
#     model.stylegan_size=512 \
#     data.transform=face_512 \
#     model.checkpoint_path=./experiments/fse_cs_editor_train/pSp_encoder/ffhq_other_attri/ffhq_glassesVSsmile_40k_000/iteration_320000.pt \