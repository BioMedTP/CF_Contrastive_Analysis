from dataclasses import dataclass, field, fields
import os

models_dir = "pretrained_models/"

# 1) define overrides for *any* field you want to change per‐dataset

specific_paths = {
    "ffhq_glasses": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        # "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        # "e4e_path": models_dir + "e4e_ffhq_encode.pt",
        # "e4e_cs_path": models_dir + "100000_iter_e4e_cs.pt",
        "pSp_cs_path": models_dir + "pSp_cs_149900_iter.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/CA-with-stylegan2-pSp/pretrained_MINE_knn/pretrained_Mine_02/checkpoints/iteration_134000.pt"
        # "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/CA-with-stylegan2-pSp/pretrained_Disc_MI/149_update_R_log/checkpoints/iteration_151000.pt"
        # #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_cs1s2/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_140000.pt"
    },
    # "ffhq_glasses_sx": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_glasses_sx/checkpoints/iteration_130000.pt"
    # },
    # "ffhq_gender_sx": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_sx/checkpoints/iteration_140000.pt",

    # },


    # "afhq_cat_dog": {
    #     "stylegan_weights":     "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pkl",
    #     "psp_path":             "../../pretrained_models/pSp_models/psp_afhqv2.pt",
    #     #"pSp_cs_path":       "../../pretrained_models/pSp_cs_models/pSp_cs_dogcat_baseline.pt",
    #     #"pSp_cs_path":       "../../pretrained_models/pSp_cs_models/pSp_cs_dogcat_sx.pt", # pSp_cs_dogcat_baseline.pt
    #     "pSp_cs_path":       "../../pSp_CS-StyleGAN/results/AFHQ/dog_cat_c_s1_s2/checkpoints/iteration_100000.pt",
    #     # "e4e_cs_path":          models_dir + "100000_iter_e4e_cs_afhq.pt"
    # },

    "brats_edit": {
        "stylegan_weights":     "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/pSp_models/stylegan2-brats_880k.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/pSp_models/stylegan2-brats_880k.pt",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/pSp_models/brats/psp_brats_220k.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/cs_models/brats/iteration_150000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/sfe/brats_inverter.pt",
        "sfe_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/sfe/refined_sfe_brats_170k.pt", 
        "config_yaml_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/sfe/config.yaml", 
    },

    # "ffhq_gender": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender/checkpoints/iteration_140000.pt",
    #     #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_sx/checkpoints/iteration_320000.pt"
    # },

    # "ffhq_age": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/other_attributes/ffhq_age/12layers_lr0.01/checkpoints/iteration_200000.pt"
    # },
    # "ffhq_pose": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/other_attributes/ffhq_pose/12layers_lr0.01/checkpoints/iteration_210000.pt"
    # },
    # "ffhq_smile": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_smile_v2/checkpoints/iteration_200000.pt"
    # },

    # "celebahq_smile": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ_smile/checkpoints/iteration_120000.pt"
    # },
    # "celebahq_gender": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_celebahq_styleganffhq.pt",
    #     #"psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ/celebaHQ_gender/checkpoints/iteration_120000.pt",
    #     #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ/celebaHQ_ffhq_gender/checkpoints/iteration_90000.pt"
    # },
    # "ffhq_glasses_smile": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/glasses_smile/lr0.01/checkpoints/iteration_220000.pt"
    # },
    # "ffhq_glassesvssmile": {
    #     "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
    #     "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/glassesVSsmile/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_40000.pt"
    # },
    # "afhq": {
    #     "stylegan_weights":     "../../pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt",
    #     "stylegan_weights_pkl": "../../pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pkl",
    # },
    
    # "isic2020": {
    #     "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/stylegan2_isic256_rosinality.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/ISIC/00002-isic256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-004032.pkl",
    #     "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/isic_Ros_resume/checkpoints/iteration_540000.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/isic2020/0.01_10layers/checkpoints/iteration_130000.pt"
    # },

    # "camelyon": {
    #     "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/Camelyon_rosinality.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/Camelyon/00001-Camelyon16256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-006451.pkl",
    #     "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/Camelyon_Ros_resume/checkpoints/iteration_540000.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/Camelyon16/checkpoints/iteration_150000.pt"
    # },

    "brats_ht": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_rosinality.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_resume/00000-brats256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-002419.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_ht_Ros_resume/checkpoints/iteration_20000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsHT/checkpoints/iteration_90000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_ht/fse_inverter_train_000/iteration_140000.pt"
    },
    
    "brats_ht_new": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_rosinality.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_resume/00000-brats256x256-mirror-paper256-ada-blit-resumecustom/network-snapshot-002419.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_ht_Ros_resume/checkpoints/iteration_20000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsHT_new/checkpoints/iteration_50000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_ht/fse_inverter_train_000/iteration_140000.pt"
    },
    
    "brats_multimod_t2f": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_cs1s2_10layers/checkpoints/iteration_30000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_cs1s2_Reg/checkpoints/iteration_120000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    },
    
    "brats_multimod_t2w": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2w_cs1s2_10layers/checkpoints/iteration_120000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2w_cs1s2_Reg/checkpoints/iteration_110000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    },    
    
    # "brats_multimod_t1c": {
    #     "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
    #     "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t1c_cs1s2_10layers/checkpoints/iteration_30000.pt",
    #     #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_cs1s2_10layers/checkpoints/iteration_30000.pt",
    #     "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    # },

    "brats_multimod_t2w_cs": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
        # "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2w_cs1s2_10layers/checkpoints/iteration_120000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    },    
    
    "brats_multimod_t2f_unpaired": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_cs1s2_unpaired/checkpoints/iteration_60000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    },
    "brats_multimod_t2f_cs": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_medical_images/brats_multiMod_Ros.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/styleGAN_ada/output/brats_multiMod_1gpu/00000-bratsMultiMod-mirror-paper256-ada-blit-resumecustom/network-snapshot-005241.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/brats_multiMod_Ros_resume/checkpoints/iteration_540000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bratsMultiMod/t1n_t2f_10layers/checkpoints/iteration_60000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/brats_multimod/fse_inverter_train_000/iteration_140000.pt"
    },
    
    ########################################################################################################
    "bloodmnist_x1y3": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/bloodmnist_Ros/checkpoints/iteration_460000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y3_Reg_cs1s2/checkpoints/iteration_30000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_002/iteration_100000.pt"
    },

    # "bloodmnist_x1y3_cs1s2": {
    #     "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pkl",
    #     "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/bloodmnist_Ros/checkpoints/iteration_460000.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y3_cs1s2/checkpoints/iteration_40000.pt",
    #     "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_002/iteration_100000.pt"
    # },

    "bloodmnist_x1y6": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/bloodmnist_Ros/checkpoints/iteration_460000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y6/checkpoints/iteration_100000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_002/iteration_100000.pt"
    },

    # "bloodmnist_x1y6_cs1s2": {
    #     "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pt",
    #     "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/bloodmnist-009676_fid11.7079.pkl",
    #     "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/bloodmnist_Ros/checkpoints/iteration_460000.pt",
    #     "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/bloodmnist_x1y6_cs1s2/checkpoints/iteration_40000.pt",
    #     "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_002/iteration_100000.pt"
    # },



    

    ########################## Octmnist splits ##########################
    "octmnist_x3y0": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x3y0_Reg/checkpoints/iteration_110000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y0/checkpoints/iteration_40000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    "octmnist_x0y1": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x3y0_Reg/checkpoints/iteration_110000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y0/checkpoints/iteration_40000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    "octmnist_x1y2": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/OCTMNIST/octmnist_x3y0_Reg/checkpoints/iteration_110000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y0/checkpoints/iteration_40000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    "octmnist_x3y1": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y1/checkpoints/iteration_60000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    "octmnist_x3y2": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y2/checkpoints/iteration_60000.pt", # 
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    "octmnist_x3y012": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/octmnist-009676_fid5.499.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/octmnist_Ros/checkpoints/iteration_1000000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/octmnist_12layers/octmnist_x3y012/checkpoints/iteration_30000.pt", # 
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/octmnist/fse_inverter_train_000/iteration_100000.pt"
    },


    ########################################################################
    
    ########################## tissuemnist splits ##########################
    "tissuemnist_x0y2": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/tissuemnist_Ros/checkpoints/iteration_440000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/checkpoints/iteration_110000.pt",
        #"inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_000/iteration_0.pt"
    },
    "tissuemnist_x0y3": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/tissuemnist_Ros/checkpoints/iteration_440000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/checkpoints/iteration_110000.pt",
        #"inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_000/iteration_0.pt"
    },
    "tissuemnist_x1y2": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/tissuemnist_Ros/checkpoints/iteration_440000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/checkpoints/iteration_110000.pt",
        #"inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_000/iteration_0.pt"
    },
    "tissuemnist_x1y3": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/tissuemnist_Ros/checkpoints/iteration_440000.pt",
        #"pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/bloodmnist/checkpoints/iteration_110000.pt",
        #"inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/bloodmnist/fse_inverter_train_000/iteration_0.pt"
    },
    ##############################################################################
    
    "pathmnist": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/pathmnist-010080_fid24.656.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/pathmnist-010080_fid24.656.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/pathmnist_Ros/checkpoints/iteration_660000.pt",
        "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/pathmnist/checkpoints/iteration_130000.pt",
        "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/pathmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    
    "tissuemnist": {
        "stylegan_weights":  "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/Medmnist/pretrained_medminist/tissuemnist-009676_fid2.227.pkl",
        "pSp_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp/results/tissuemnist_Ros/checkpoints/iteration_440000.pt",
        # "pSp_cs_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/pSp_CS-StyleGAN/results/isic2020/0.01_10layers/checkpoints/iteration_130000.pt"
        # "inverter_pth": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/StyleFeatureEditor-CS/experiments/inverter/pathmnist/fse_inverter_train_000/iteration_100000.pt"
    },
    
    
    
}


        # if self.opts.e4e_checkpoint_path is not None:
        #     print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.e4e_checkpoint_path))
        #     ckpt = torch.load(self.opts.e4e_checkpoint_path, map_location='cpu', weights_only=True)
        #     self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        #     self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
# SOTA_encoders_StyleGAN/e4e/results/12nmlp_lr0.01_church_resume/checkpoints/iteration_24000.pt

@dataclass
class DefaultPathsClass:
    dataset: str = os.getenv("DATASET", "ffhq_glasses")

    # Base paths
    farl_path:      str = models_dir + "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    mobile_net_pth: str = models_dir + "mobilenet0.25_Final.pth"
    ir_se50_path:   str = models_dir + "model_ir_se50.pth"
    stylegan_car_weights: str = models_dir + "stylegan2-car-config-f-new.pkl"
    arcface_model_path:   str = models_dir + "iresnet50-7f187506.pth"
    moco:                 str = models_dir + "moco_v2_800ep_pretrain.pt"
    curricular_face_path: str = models_dir + "CurricularFace_Backbone.pth"
    mtcnn:                str = models_dir + "mtcnn"
    landmark:             str = models_dir + "79999_iter.pth"
    # decoder_multiplier:   int = 2
    # Optional — can be filled by dataset-specific config
    stylegan_weights:     str = None
    stylegan_weights_pkl: str = None
    pSp_path:             str = None
    e4e_path:             str = None
    e4e_cs_path:          str = None
    pSp_cs_path:          str = None
    inverter_pth:         str = None
    def __post_init__(self):
        key = self.dataset.lower()
        overrides = specific_paths.get(key, {})

        if not overrides:
            raise ValueError(f"No specific path config found for dataset '{self.dataset}'.")

        for name, val in overrides.items():
            # Append if not already set or explicitly None
            if not hasattr(self, name) or getattr(self, name) is None:
                setattr(self, name, val)
            else:
                print(f"[INFO] Keeping existing value for '{name}', not overridden.")

        # Sanity check for critical fields
        if self.stylegan_weights is None or self.stylegan_weights_pkl is None:
            raise ValueError(f"Missing required stylegan weights for dataset '{self.dataset}'.")

    def __iter__(self):
        for f in fields(self):
            yield f.name, getattr(self, f.name)

# Singleton instance
DefaultPaths = DefaultPathsClass()
