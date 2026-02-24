# data_configs.py
data_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/"
medminist_path = '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets'
med_path = '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging'
# /home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/Medical_imaging/brats/Preprocessed/train_styleGAN2ada_multiMod
# /home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/brats/Preprocessed/train_styleGAN2ada_multiMod
# /home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN//Medical_imaging/brats/Preprocessed/train_styleGAN2ada_multiMod
DATASETS = {

    'camelyon': {
        'train': med_path +'/Camelyon16/cam16_output',
        'val': med_path +'/Camelyon16/cam16_output_test',
      'train_bg': med_path + '/Camelyon16/X_Y_split/train_X.npy',
      'train_t': med_path + '/Camelyon16/X_Y_split/train_Y.npy',
      'test_bg':  med_path + '/Camelyon16/X_Y_split/test_X.npy',
      'test_t':  med_path + '/Camelyon16/X_Y_split/test_Y.npy',
        "special_bg": med_path + '/Camelyon16/X_Y_split/test_X.npy',
        "special_t":  med_path + '/Camelyon16/X_Y_split/test_Y.npy',
	},
    
   'brats_ht': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsHT/train_X.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsHT/train_Y.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',
	},

   'brats_ht_new': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/train_X.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/train_Y.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_X.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_Y.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_X.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_Y.npy',
	},


    
  'brats_multimod_t1c': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t1c.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t1c.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t1c.npy',
	},
    
  'brats_multimod_t2f': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2f.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2f.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2f.npy',
	},
    
  'brats_multimod_t2w': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2w.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2w.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2w.npy',
	},
  
    
  'brats_multimod_t2w_cs': {
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2w.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2w.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2w.npy',
	},
  
  'brats_multimod_t2f_unpaired':{
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_Y_t2f.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_Y_t2f.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_Y_t2f.npy',
        }, 
  'brats_multimod_t2f_cs':{
        'train': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod',
        'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_multiMod_test',
        'train_bg': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
        'train_t': med_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2f.npy',
        'val_bg':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        'val_t':  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2f.npy',
        "special_bg": med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',
        "special_t":  med_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2f.npy',
        }, 


	'bloodmnist_x1y3': {
    'train': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224',
    'val': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_Y.npy',
    'special_bg':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_X.npy',
    'special_t':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_Y.npy'
	},

	'bloodmnist_x1y3_cs1s2': {
    'train': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224',
    'val': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_Y.npy',
    'special_bg':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_X.npy',
    'special_t':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_Y.npy'
	},


	'bloodmnist_x1y6': {
		'train': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224',
		'val': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224_test',     
		'train_bg': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_Y.npy'
  },


	'bloodmnist_x1y6_cs1s2': {
		'train': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224',
		'val': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224_test',     
		'train_bg': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_Y.npy'
  },

	'octmnist_x3y0': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x3_y0/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x3_y0/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_Y.npy',
    'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_Y.npy',
	},
    
  'octmnist_x3y1': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x3_y1/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x3_y1/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_Y.npy',
  },
  'octmnist_x1y2': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x1_y2/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x1_y2/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_Y.npy',
  },
  'octmnist_x0y1': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
  },
  'octmnist_x3y2': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x3_y2/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x3_y2/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_Y.npy',
  },

  'octmnist_x3y012': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x3_y012/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x3_y012/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_Y.npy',
  },    
    
  'octmnist_x0y1': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
  },  

  'octmnist_x0y2': {
		'train': medminist_path + '/MedMNIST/octmnist/octmnist_224',
		'val': medminist_path + '/MedMNIST/octmnist/octmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/octmnist/split_x0_y2/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/octmnist/split_x0_y2/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_Y.npy',
		'special_bg':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_Y.npy',
  },  

  ###########################################################
	'pathmnist': {
		'train': medminist_path + '/MedMNIST/pathmnist/pathmnist_224',
		'val': medminist_path + '/MedMNIST/pathmnist/pathmnist_224_test',
		'train_bg': medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/train_X.npy',
		'train_t': medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/train_Y.npy',
		'val_bg':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_X.npy',
		'val_t':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_Y.npy',
  	'special_bg':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_X.npy',
		'special_t':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_Y.npy',
	},

    # 'seg_train_on_Y': {
    #     'train': med_path +'/brats/Preprocessed/train_styleGAN2ada',
    #     'val': med_path +'/brats/Preprocessed/train_styleGAN2ada_test',
    #     'train_bg': med_path + '/brats/X_Y_splits/bratsHT/train_X.npy',
    #     'train_t': med_path + '/brats/X_Y_splits/bratsHT/train_Y.npy',
    #     'val_bg':  med_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
    #     'val_t':  med_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',
    #     "special_bg": med_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
    #     "special_t":  med_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',         
    # },
    # 'ffhq_gender': {
    #     "train_bg":   data_dir + 'ffhq_cs_gender_age/train_data_male',
    #     "train_t":    data_dir + 'ffhq_cs_gender_age/train_data_female',
    #     "val_bg":     data_dir + 'ffhq_cs_gender_age/test_data_male',
    #     "val_t":      data_dir + 'ffhq_cs_gender_age/test_data_female',
    #     # "special_bg": data_dir + 'special_images/ffhq_gender/male',
    #     # "special_t":  data_dir + 'special_images/ffhq_gender/female',          
    # },
    # 'ffhq_age': {
    #     "train_bg":   data_dir + 'ffhq_cs_gender_age/train_data_young',
    #     "train_t":    data_dir + 'ffhq_cs_gender_age/train_data_old',
    #     "val_bg":     data_dir + 'ffhq_cs_gender_age/test_data_young',
    #     "val_t":      data_dir + 'ffhq_cs_gender_age/test_data_old',
    #     # "special_bg": data_dir + 'special_images/ffhq_age/young',
    #     # "special_t":  data_dir + 'special_images/ffhq_age/old',          
    # },

    # 'ffhq_pose': {
    #     "train_bg":   data_dir + 'ffhq_cs_headpose/train_pose_frontal',
    #     "train_t":    data_dir + 'ffhq_cs_headpose/train_pose_left_right',
    #     "val_bg":     data_dir + 'ffhq_cs_headpose/test_pose_frontal',
    #     "val_t":      data_dir + 'ffhq_cs_headpose/test_pose_left_right',
    #     # "special_bg": data_dir + 'special_images/ffhq_pose/frontal',
    #     # "special_t":  data_dir + 'special_images/ffhq_pose/left_right',          
    # },

    
    # 'ffhq_smile': {
    #     "train_bg":   data_dir + 'ffhq_cs_smile/train_smile_no',
    #     "train_t":    data_dir + 'ffhq_cs_smile/train_smile_yes',
    #     "val_bg":     data_dir + 'ffhq_cs_smile/test_smile_no',
    #     "val_t":      data_dir + 'ffhq_cs_smile/test_smile_yes',
    #     # "special_bg": data_dir + 'special_images/ffhq_smile/nosmile',
    #     # "special_t":  data_dir + 'special_images/ffhq_smile/smile',          
    # },
    
    # 'ffhq_glasses_smile': {
    #     "train_bg":  data_dir + '/ffhq_attrbutes/glasses_smile/train_neither_GS.npy',
    #     "train_t":   data_dir + '/ffhq_attrbutes/glasses_smile/train_GlassesSmile.npy',
    #     "val_bg":    data_dir + '/ffhq_attrbutes/glasses_smile/test_neither_GS.npy',
    #     "val_t":      data_dir + '/ffhq_attrbutes/glasses_smile/test_GlassesSmile.npy',
    #     # "special_bg": data_dir + '/ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/neither',
    #     # "special_t":  data_dir + '/ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/glasses_smile',      
    # },
    
    # 'ffhq_glassesvssmile': {
    #     "train_bg":   data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_glasses_only.npy',
    #     "train_t":    data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_smile_only.npy',
    #     "val_bg":    data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_glasses_only.npy',
    #     "val_t":      data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_smile_only.npy',
    #     # "special_bg": data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_glasses',
    #     # "special_t":  data_dir + '/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_smile',      
    # },
    
    # 'celebahq_smile': {
    #     "train_bg":   data_dir + '/CelebA-HQ/Smiling/train_smile_no',
    #     "train_t":    data_dir + '/CelebA-HQ/Smiling/train_smile_yes',
    #     "val_bg":     data_dir + '/CelebA-HQ/Smiling/test_smile_no',
    #     "val_t":      data_dir + '/CelebA-HQ/Smiling/test_smile_yes',
    #     # "special_bg": data_dir + '/CelebA-HQ/Smiling/test_smile_no',
    #     # "special_t":  data_dir + '/CelebA-HQ/Smiling/test_smile_yes',          
    # },
    # 'celebahq_gender': {
    #     "train_bg":   data_dir + '/CelebA-HQ/Gender/train_male',
    #     "train_t":    data_dir + '/CelebA-HQ/Gender/train_female',
    #     "val_bg":     data_dir + '/CelebA-HQ/Gender/test_male',
    #     "val_t":      data_dir + '/CelebA-HQ/Gender/test_female',
    #     # "special_bg": data_dir + '/CelebA-HQ/Gender/test_male',
    #     # "special_t":  data_dir + '/CelebA-HQ/Gender/test_female',          
    # },

    # 'lsun_church': {
    #     "train_bg":   data_dir + '/LSUN_church/church_daylight_train',
    #     "train_t":    data_dir + '/LSUN_church/church_night_train',
    #     "val_bg":     data_dir + '/LSUN_church/church_daylight_test',
    #     "val_t":      data_dir + '/LSUN_church/church_night_test',
    #     "special_bg": data_dir + '/LSUN_church/church_daylight_test',
    #     "special_t":  data_dir + '/LSUN_church/church_night_test',          
    # },

    # 'ffhq': {
    #     "train":   data_dir + 'ffhq_glasses/train_bg',
    #     "val":     data_dir + 'ffhq_glasses/test_bg',
    #     "special_bg": '../special_images/background',
    #     "special_t":  '../special_images/glasses'            
    # },
    
    # 'afhq_cat_dog': {
    #     "train":   data_dir + 'AFHQ/afhq-v2/train/cat_dog',
    #     "val":     data_dir + 'AFHQ/afhq-v2/test/cat_dog',
    #     "special_bg": data_dir + 'AFHQ/afhq-v2/val/cat',
    #     "special_t":  data_dir + 'AFHQ/afhq-v2/val/dog',
    #     'train_bg': data_dir +'AFHQ/afhq-v2/train/cat',
    #     'train_t': data_dir +'AFHQ/afhq-v2/train/dog', 
    #     'val_bg': data_dir +'AFHQ/afhq-v2/test/cat',
    #     'val_t': data_dir +'AFHQ/afhq-v2/test/dog', 
    # },

    'BraTS_evaluation': {
      'val_bg': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Result_collections/BraTS_evaluation/real_X', 
      'val_t': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Result_collections/BraTS_evaluation/real_Y',
      },
    'BraTS_evaluation_70_100': {
      'val_bg': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Result_collections/BraTS_evaluation_70_100/real_X', 
      'val_t': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Result_collections/BraTS_evaluation_70_100/real_Y',
    },
    # 'isic2020': {
    # 'train': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/train_G_ada',
    # 'val': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/test',
	# 'train_bg': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/train_X.npy',
	# 'train_t': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/train_Y.npy',
    # 'val_bg': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_X.npy',
    # 'val_t': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_Y.npy', 
    # "special_bg": '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_X.npy',
    # "special_t": '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_Y.npy',
    # },

}
# Sp_CS-StyleGAN/results/glassesVSsmile/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_100000.pt
	# 'train_neither_GS': './ffhq_attrbutes/glasses_smile/train_neither_GS.npy',
    # 'train_GlassesSmile': './ffhq_attrbutes/glasses_smile/train_GlassesSmile.npy',
    # 'test_neither_GS': './ffhq_attrbutes/glasses_smile/test_neither_GS.npy',
    # 'test_GlassesSmile': './ffhq_attrbutes/glasses_smile/test_GlassesSmile.npy',