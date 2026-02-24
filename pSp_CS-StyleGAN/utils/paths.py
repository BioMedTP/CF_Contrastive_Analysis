# data_configs.py
data_root = "/lustre/fsn1/projects/rech/hht/usv51hl/Projects/data/"

DATASETS = {
    'ffhq_glasses': {
        "train_bg":   data_root + 'ffhq_glasses/train_bg',
        "train_t":    data_root + 'ffhq_glasses/train_t',
        "val_bg":     data_root + 'ffhq_glasses/test_bg',
        "val_t":      data_root + 'ffhq_glasses/test_t',
        "special_bg": '../special_images/background',
        "special_t":  '../special_images/glasses'            
    },
    'ffhq': {
        "train":   '/lustre/fsn1/projects/rech/hht/usv51hl/Projects/data/ffhq_glasses/train',
        "val":     '/lustre/fsn1/projects/rech/hht/usv51hl/Projects/codes/special_images/val',
        "special_bg": '../special_images/background',
        "special_t":  '../special_images/glasses'            
    },
    'afhq_cat_dog': {
        "train":   data_root + 'afhq-v2/train/cat_dog',
        "val":     data_root + 'afhq-v2/test/cat_dog',
        "train_bg":   data_root + 'afhq-v2/train/cat',
        "train_t":   data_root + 'afhq-v2/train/dog',
        "val_bg":     data_root + 'afhq-v2/test/cat',
        "val_t":     data_root + 'afhq-v2/test/dog',
        "special_bg": data_root + 'afhq-v2/val/cat',
        "special_t":  data_root + 'afhq-v2/val/dog'
    },

    'brats_inversion': {
    'train': data_root + 'BraTS2023/inversion_train',
	'val': data_root + 'BraTS2023/inversion_test',
    "special_bg": '/lustre/fsn1/projects/rech/hht/usv51hl/Projects/codes/StyleFeatureEditor-CS/datasets/special_bg',
    "special_t":  '/lustre/fsn1/projects/rech/hht/usv51hl/Projects/codes/StyleFeatureEditor-CS/datasets/special_t'    
    },
    # …
}

DATASETS_NPZ = {
    'glasses_gender': {
        "train":   "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq-features-dataset-master/labels/ffhq_gender_and_glasses/train_groups.npz",
        "test":   "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq-features-dataset-master/labels/ffhq_gender_and_glasses/test_groups.npz",         
    },
    # …
}



model_paths = {
    'ffhq_glasses': {
        "stylegan_weights":     "../pretrained_models/pSp_models/stylegan2-ffhq-config-f.pt",
        "pSp_checkpoint_path":   "../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "csmlp_checkpoint_path": None,
    },
    "ir_se50_path": "../pretrained_models/pSp_models/model_ir_se50.pth"
    
    # "afhq_cat_dog": {
    #     "stylegan_weights":     models_dir + "stylegan2-afhqv2-512x512.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-afhqv2-512x512.pkl",
    #     "psp_path":             models_dir + "pSp_afhq.pt",
    #     "pSp_cs1s2_path":       models_dir + "pSp_cs1s2_afhq_80000.pt",
    # },

    # "brats_inversion": {
    #     "stylegan_weights":     models_dir + "stylegan2-brats_880k.pt",
    #     "stylegan_weights_pkl": models_dir + "stylegan2-brats_880k.pt",
    #     "psp_path":             models_dir + "psp_brats_encode.pt",
    #     "e4e_path":             models_dir + "e4e_brats_encode.pt",
    #     "e4e_cs_path":          models_dir + "100000_iter_e4e_cs_brats.pt",
    #     "pSp_cs_path":          models_dir + "pSp_brats_cs.pt",
    # },

}

