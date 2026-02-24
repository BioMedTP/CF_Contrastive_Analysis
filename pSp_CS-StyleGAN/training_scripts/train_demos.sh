
python training_scripts/train.py \
--exp_scheme=baseline \
--exp_dir=results/cmlp_baseline/3Dmlp \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses


python training_scripts/train.py \
--exp_scheme=baseline \
--exp_dir=results/TEST2 \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glasses


python training_scripts/train.py \
--exp_scheme=Swap_Lat \
--exp_dir=results/cmlp_Swap_Lat/3Dmlp \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--image_interval=10000 \
--print_interval=10000 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--dataset_type=ffhq_glass \
--w_lat_swap_lambda=1.0



python training_scripts/train.py \
--exp_dir=results/train_c2smlp/ \
--exp_scheme=train_c2smlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim


scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/2_data/ uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code
nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code

nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Projects/CA_with_GAN/3_code/diffusion-AE uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/3_code

nohup scp -3 -r yuhe@gpu-gw.enst.fr:/home/ids/yuhe/Shared/ uri15na@jean-zay.idris.fr:/lustre/fswork/projects/rech/ggs/uri15na/2_data/MRI_dataset



python training_scripts/train.py \
--exp_dir=results/train_c2smlp/12layers_l1_lr0.01 \
--exp_scheme=train_c2smlp \
--dataset_type=ffhq_glasses \
--pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
--csmlp_checkpoint_path=results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
--log_interval=500 \
--val_interval=10000 \
--save_interval=10000 \
--n_layers_mlp=12 \
--optim_name=admn \
--seed=99 \
--mlp_norm_type=nodim \
--learning_rate=0.01 \
--c2s_loss_type=l1




# python training_scripts/train.py \
# --exp_scheme=alternative_train \
# --exp_dir=results/alternative_train/131k \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --cmlp_checkpoint_path=./results/alternative_cls/cls100ep_lr0.01_lamb1.0/checkpoints/iteration_130800.pt \
# --cls_checkpoint_path=./training_cls/results/sample_cls/2layer_lr0.001_bs4/checkpoints/model_epoch_50.pth \
# --log_interval=200 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=140000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --learning_rate=0.01 \
# --w_cls_lambda=0.0 \

# python training_scripts/train.py \
# --exp_scheme=baseline \
# --exp_dir=results/cmlp_baseline/3Dmlp \
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

# python training_scripts/train.py \
# --exp_scheme=alternative_cls \
# --exp_dir=results/alternative_cls/cls50ep_lr0.01_lamb0.1 \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --cmlp_checkpoint_path=./results/csmlp_ffhq_glasses/mlp3D/nodim/checkpoints/iteration_130000.pt \
# --cls_checkpoint_path=./training_cls/results/sample_cls/2layer_lr0.0001_bs4/checkpoints/model_epoch_50.pth \
# --log_interval=200 \
# --val_interval=200 \
# --save_interval=200 \
# --max_steps=200000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_glasses \
# --learning_rate=0.01 \
# --w_cls_lambda=0.1 \
# --batch_size=1.0 \
# --test_batch_size=0.0

# --adv_c2s_lambda=100  
# python training_scripts/train.py \
# --exp_dir=results/csmlp_afhqv2/3Dmlp_layers10_test \
# --dataset_type=afhqv2 \
# --output_size=512 \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pixel2style2pixel_orn/results/afhqv2/ffhq_params/checkpoints/iteration_300000.pt \
# --stylegan_weights=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=10 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=3D-mlp \
# --mlp_norm_type=nodim


#/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pixel2style2pixel_orn/results/celebahq/init-pSp/stylegan2-ffhq1024/checkpoints/iteration_260000.pt
# python training_scripts/train.py \
# --exp_dir=results/csmlp_ffhq_smile/2Dmlp \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=2D-mlp \
# --mlp_norm_type=nodim \
# --dataset_type=ffhq_smile

# python training_scripts/train.py \
# --exp_dir=results/celebaHQv2/Gender \
# --dataset_type=celebaHQ_gender \
# --pSp_checkpoint_path=/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pixel2style2pixel_orn/results/celebahq/ffhq_pSp/stylegan2-ffhq/checkpoints/iteration_260000.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=3D-mlp \
# --mlp_norm_type=nodim \



# ####### Compare different architectures: ############
# python train_scripts/train.py \
# --exp_dir=results/network_architectures/simple_cnn/ \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=compare_architectures \
# --mlp_norm_type=nodim \
# --zero_out_type=hard \
# --zero_out_threshold=0.0 \
# --lasso_sbg_lambda=0.0 \
# --lasso_st_lambda=0.0 \
# --lasso_direction=element \
# --lasso_threshold=0.0 \
# --lasso_reg_type=traditional \
# --net_type=simple_cnn 
# --resblock_type=standard
# --trans_act_fn=gelu
# --spatial_encoding=spatial-aware

# --sbg_type=fro \
# --id_lambda=0.0 \
# --pix_lambda=0.0 \
# --lpips_lambda=0.0 \
# --n_shared_layers=2 \
# --n_branch_layers=12 \

## factorization shared independent


# ############### Apply Lasso sparsity ################
# python train_scripts/train.py \
# --exp_dir=results/csmlp_sparsity/mlp3D/effect_l1reg_types/thresholded/element_zeroOut1e-4_lasso3e-3_threshold1e-3_sbg2.0 \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=20000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=3Dsparsity \
# --mlp_norm_type=nodim \
# --zero_out_silent_bg \
# --zero_out_silent_t \
# --zero_out_type=hard \
# --zero_out_threshold=1e-4 \
# --lasso_sbg_lambda=3e-3 \
# --lasso_st_lambda=3e-3 \
# --lasso_direction=element \
# --lasso_threshold=1e-3 \
# --lasso_reg_type=thresholded \
# --net_type=independent \
# --sbg_lambda=2.0

# # ################ csmlp_with_classifier: ################
# python train_scripts/train.py \
# --exp_dir=results/mlp_with_classifier_recon_imgs/ \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --image_interval=10000 \
# --print_interval=10000 \
# --val_interval=10000 \
# --save_interval=10000 \
# --n_layers_mlp=12 \
# --optim_name=admn \
# --seed=99 \
# --exp_scheme=csmlp_with_classifier \
# --mlp_norm_type=nodim \
# --zero_out_type=hard \
# --zero_out_threshold=0.0 \
# --lasso_sbg_lambda=0.0 \
# --lasso_st_lambda=0.0 \
# --lasso_direction=element \
# --learning_rate_class=1e-3 \
# --lasso_threshold=0.0 \
# --net_type=independent \
# --class_lambda=1.0 \
# --use_scheduler \


# python train_scripts/train.py \
# --exp_dir=results/knowledge_distillation/mult_gpus_ranger0.001/ \
# --pSp_checkpoint_path=../pretrained_models/pSp_models/psp_ffhq_encode.pt \
# --exp_scheme=cskd_multi_gpus \
# --optim_name=ranger \
# --learning_rate=0.001 \
# --batch_size=8 \
# --workers=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --print_interval=100 \
# --image_interval=100 \
# --val_interval=10000 \
# --save_interval=20000 \

# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread -C h100