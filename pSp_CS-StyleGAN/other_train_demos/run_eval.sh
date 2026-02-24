#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --partition=V100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=23:00:00
#SBATCH --output=reports/out_eval_age_.txt
#SBATCH --error=reports/err_eval_age_.txt

source ~/anaconda3/bin/activate styleGANenv

# Detect the GPU type and set the appropriate TORCH_CUDA_ARCH_LIST
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)

case $gpu_name in
    *P100*)
        export TORCH_CUDA_ARCH_LIST="6.0"
        ;;
    *V100*)
        export TORCH_CUDA_ARCH_LIST="7.0"
        ;;
    *A100*)
        export TORCH_CUDA_ARCH_LIST="8.0"
        ;;
    *)
        echo "Unknown GPU: $gpu_name"
        exit 1
        ;;
esac

export CUDA_HOME=/usr/local/cuda-12.5 # Set CUDA 12.5 environment variables
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

rm -rf /home/ids/yuhe/.cache/torch_extensions

cd /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_encoder_constructive

# python evaluation/inference.py \
#     --model_ckpt_path ./results/ffhq/iteration_149900.pt \
#     --image_folder ./eval_images/FFHQtest/background \
#     --label_type background \
#     --output_dir ./eval_images/FFHQtest/background

python evaluate_metrics.py \
    --input_dir_real "eval_images/BraTs/recon_t" \
    --input_dir_eval "eval_images/current/references/real_t" \
    --output_dir "results/Quanti/BraTS/results_t" \
    #--num_test 100


# python evaluate_metrics.py \
#     --input_dir_real "eval_images/glasses/references/real_t" \
#     --input_dir_eval "eval_images/glasses/mlp2Dnodim/recon_t" \
#     --output_dir "results/Quanti/2Dnodim/results_t" \

# ###########  reproduce_eval_images ##############  
# python evaluation/reproduce_eval_images.py \
#     --model_ckpt_path ./results/csmlp_BraTS/540k/checkpoints/iteration_140000.pt \
#     --recon_swap_path ./eval_images/BraTs/ \
#     --references_path ./eval_images/current/references \
# #     --eval_ref

# python evaluation/reproduce_eval_e4e.py \
#     --model_ckpt_path ./results/ffhq/iteration_149900.pt \
#     --recon_swap_path ./eval_images/e4e/reference \
#     --references_path ./eval_images/e4e_all/ \
#     --eval_ref

# ############## FID score ############# 
# python evaluation/image_fid_score.py \
#     --eval_images_dir ./eval_images/current_with_time/recon_swap\
#     --real_images_dir ./eval_images/current/references \
#     --save_results_dir ./results/Quanti/fid_scores/fid.txt \
   ## --eval_pSp 

# # ############  Image identity ############# 
# python evaluation/image_identity.py \
#     --eval_images_dir ./eval_images/age/3Dmlp/recon_swap \
#     --real_images_dir ./eval_images/age/references \
#     --save_results_dir ./evaluation/results_fid_age-gender-smile/identity \
#     --max_images -1 --dist_metric cosine

    #--eval_ref
    
# ############# calculate glasses detection #############
# python evaluation/image_glasses_detection.py \
#     --recon_swap_path ./eval_images/sparsity_3e-3_sbg1.5 \
#     --save_results_path ./evaluation/eval_results/glasses_detection/SwinB_sparsity_3e-3_sbg1.5 \
#     --max_images -1 --box_threshold 0.0 --model_type SwinB \

    #--eval_ref --references_path ./eval_images/references
    #--reproduce_images


# ######### latent glasses classification/separately (Logistic regression score) ############
# python evaluation/latent_classification.py \
#     --model_ckpt_path ./results/csmlp_BraTS/540k/checkpoints/iteration_140000.pt \
#     --label_csv_path ./brain_mri_labels.csv \
#     --results_dir ./evaluation/BraTS \
#     --cls_type Label \
#     --reduced_dim None


    # parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to the real images directory')
    # parser.add_argument('--reduced_dim', type=str, default="50", help='Directory where output reconstruction images will be saved')
    # parser.add_argument('--results_dir', type=str, default='./evaluation/separately.txt', help='Directory where fid txt results will be saved')
    # parser.add_argument('--random_seed', type=int, default=42, help='set a random seed for different results')
    # parser.add_argument('--label_csv_path', type=str, default='/labeled_gender_age.csv', help='path to label attributes')
    # parser.add_argument('--cls_type', type=str, default='male-female', help='gender or age')

# ########### calculate gender separately (Logistic regression score) ############
# python evaluation/latent_gender_classification.py \
#     --model_ckpt_path ./results/network_architectures/cnn/checkpoints/iteration_100000.pt \
#     --save_results_dir ./evaluation/eval_results/latent_gender_cls.txt \
#     --reduced_dim 50


# ########### calculate ages separately (Logistic regression score) ############
# python evaluation/latent_age_regression.py \
#     --model_ckpt_path ./results/csmlp_sparsity/mlp3D/effect_l1reg_types/thresholded/element_zeroOut1e-4_lasso3.6e-3_threshold1e-3/checkpoints/iteration_100000.pt \
#     --save_results_dir ./evaluation/eval_results/latent_age_reg_lasso.txt \
#     --reduced_dim None --l1_ratio=0.5