from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'glassesANDsmile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['train_noglasses_nonsmile'],
		'train_bg_target_root': dataset_paths['train_noglasses_nonsmile'],
		'test_bg_source_root': dataset_paths['test_noglasses_nonsmile'],
		'test_bg_target_root': dataset_paths['test_noglasses_nonsmile'],
                
		'train_t_source_root': dataset_paths['train_glasses_smile'],
		'train_t_target_root': dataset_paths['train_glasses_smile'],
		'test_t_source_root': dataset_paths['test_glasses_smile'],
		'test_t_target_root': dataset_paths['test_glasses_smile'],              
	},

	'glassesORsmile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['train_neither'],
		'train_bg_target_root': dataset_paths['train_neither'],
		'test_bg_source_root': dataset_paths['test_neither'],
		'test_bg_target_root': dataset_paths['test_neither'],
                
		'train_t_source_root': dataset_paths['train_glassesORsmile'],
		'train_t_target_root': dataset_paths['train_glassesORsmile'],
		'test_t_source_root': dataset_paths['test_glassesORsmile'],
		'test_t_target_root': dataset_paths['test_glassesORsmile'],              
	},
    
	'glassesVSsmile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['train_glasses_only'],
		'train_bg_target_root': dataset_paths['train_glasses_only'],
		'test_bg_source_root': dataset_paths['test_glasses_only'],
		'test_bg_target_root': dataset_paths['test_glasses_only'],
                
		'train_t_source_root': dataset_paths['train_smile_only'],
		'train_t_target_root': dataset_paths['train_smile_only'],
		'test_t_source_root': dataset_paths['test_smile_only'],
		'test_t_target_root': dataset_paths['test_smile_only'],              
	},


	'glasses_smile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['train_neither_GS'],
		'train_bg_target_root': dataset_paths['train_neither_GS'],
		'test_bg_source_root': dataset_paths['test_neither_GS'],
		'test_bg_target_root': dataset_paths['test_neither_GS'],
                
		'train_t_source_root': dataset_paths['train_GlassesSmile'],
		'train_t_target_root': dataset_paths['train_GlassesSmile'],
		'test_t_source_root': dataset_paths['test_GlassesSmile'],
		'test_t_target_root': dataset_paths['test_GlassesSmile'],              
	},	 

	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_root': dataset_paths['ffhq_bg_train'],
		'train_t_root': dataset_paths['ffhq_glass_train'],
		'test_bg_root': dataset_paths['ffhq_bg_test'],
		'test_t_root': dataset_paths['ffhq_glass_test'],          
	},
        
	'ffhq_glasses': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_bg_train'],
		'train_bg_target_root': dataset_paths['ffhq_bg_train'],
		'test_bg_source_root': dataset_paths['ffhq_bg_test'],
		'test_bg_target_root': dataset_paths['ffhq_bg_test'],
                
		'train_t_source_root': dataset_paths['ffhq_glass_train'],
		'train_t_target_root': dataset_paths['ffhq_glass_train'],
		'test_t_source_root': dataset_paths['ffhq_glass_test'],
		'test_t_target_root': dataset_paths['ffhq_glass_test'],           
	},
    
	'ffhq_gender': {
		'transforms': transforms_config.EncodeTransforms,
		'train_t_source_root': dataset_paths['ffhq_male_train'],
		'train_t_target_root': dataset_paths['ffhq_male_train'],
		'test_t_source_root': dataset_paths['ffhq_male_test'],
		'test_t_target_root': dataset_paths['ffhq_male_test'],
        
		'train_bg_source_root': dataset_paths['ffhq_female_train'],
		'train_bg_target_root': dataset_paths['ffhq_female_train'],
		'test_bg_source_root': dataset_paths['ffhq_female_test'],
		'test_bg_target_root': dataset_paths['ffhq_female_test'],              
	},
    
	'ffhq_age': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_young_train'],
		'train_bg_target_root': dataset_paths['ffhq_young_train'],
		'test_bg_source_root': dataset_paths['ffhq_young_test'],
		'test_bg_target_root': dataset_paths['ffhq_young_test'],
        
		'train_t_source_root': dataset_paths['ffhq_old_train'],
		'train_t_target_root': dataset_paths['ffhq_old_train'],
		'test_t_source_root': dataset_paths['ffhq_old_test'],
		'test_t_target_root': dataset_paths['ffhq_old_test'],            
	},
    
	'ffhq_smile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_nosmile_train'],
		'train_bg_target_root': dataset_paths['ffhq_nosmile_train'],
		'test_bg_source_root': dataset_paths['ffhq_nosmile_test'],
		'test_bg_target_root': dataset_paths['ffhq_nosmile_test'],
        
		'train_t_source_root': dataset_paths['ffhq_smile_train'],
		'train_t_target_root': dataset_paths['ffhq_smile_train'],
		'test_t_source_root': dataset_paths['ffhq_smile_test'],
		'test_t_target_root': dataset_paths['ffhq_smile_test'],            
	},

	'ffhq_pose': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_frontal_train'],
		'train_bg_target_root': dataset_paths['ffhq_frontal_train'],
		'test_bg_source_root': dataset_paths['ffhq_frontal_test'],
		'test_bg_target_root': dataset_paths['ffhq_frontal_test'],
        
		'train_t_source_root': dataset_paths['ffhq_leftright_train'],
		'train_t_target_root': dataset_paths['ffhq_leftright_train'],
		'test_t_source_root': dataset_paths['ffhq_leftright_test'],
		'test_t_target_root': dataset_paths['ffhq_leftright_test'],            
	},
        
	'afhqv2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['afhqcat_train'],
		'train_bg_target_root': dataset_paths['afhqcat_train'],
		'test_bg_source_root': dataset_paths['afhqcat_test'],
		'test_bg_target_root': dataset_paths['afhqcat_test'],
        
		'train_t_source_root': dataset_paths['afhqdog_train'],
		'train_t_target_root': dataset_paths['afhqdog_train'],
		'test_t_source_root': dataset_paths['afhqdog_test'],
		'test_t_target_root': dataset_paths['afhqdog_test'], 

	},
    
	'celebaHQ_gender': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_male_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_male_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_male_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_male_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_female_train'],
		'train_t_target_root': dataset_paths['celebaHQ_female_train'],
		'test_t_source_root': dataset_paths['celebaHQ_female_test'],
		'test_t_target_root': dataset_paths['celebaHQ_female_test'],              
	},
    
	'celebaHQ_smile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_nosmile_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_nosmile_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_nosmile_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_nosmile_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_smile_train'],
		'train_t_target_root': dataset_paths['celebaHQ_smile_train'],
		'test_t_source_root': dataset_paths['celebaHQ_smile_test'],
		'test_t_target_root': dataset_paths['celebaHQ_smile_test'],              
	},

	'celebaHQ_lipstick': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_noLipstick_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_noLipstick_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_noLipstick_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_noLipstick_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_Lipstick_train'],
		'train_t_target_root': dataset_paths['celebaHQ_Lipstick_train'],
		'test_t_source_root': dataset_paths['celebaHQ_Lipstick_test'],
		'test_t_target_root': dataset_paths['celebaHQ_Lipstick_test'],              
	},
	'BraTS_tumor': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['brats_healthy_train'],
		'train_bg_target_root': dataset_paths['brats_healthy_train'],
		'test_bg_source_root': dataset_paths['brats_healthy_test'],
		'test_bg_target_root': dataset_paths['brats_healthy_test'],
        
		'train_t_source_root': dataset_paths['brats_tumor_train'],
		'train_t_target_root': dataset_paths['brats_tumor_train'],
		'test_t_source_root': dataset_paths['brats_tumor_test'],
		'test_t_target_root': dataset_paths['brats_tumor_test'],              
	},
	'lsun_horse': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['horse_only_train'],
		'train_bg_target_root': dataset_paths['horse_only_train'],
		'test_bg_source_root': dataset_paths['horse_only_test'],
		'test_bg_target_root': dataset_paths['horse_only_test'],
        
		'train_t_source_root': dataset_paths['horse_person_train'],
		'train_t_target_root': dataset_paths['horse_person_train'],
		'test_t_source_root': dataset_paths['horse_person_test'],
		'test_t_target_root': dataset_paths['horse_person_test'],              
	},
	'lsun_church': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['church_daylight_train'],
		'train_bg_target_root': dataset_paths['church_daylight_train'],
		'test_bg_source_root': dataset_paths['church_daylight_test'],
		'test_bg_target_root': dataset_paths['church_daylight_test'],
		'train_t_source_root': dataset_paths['church_night_train'],
		'train_t_target_root': dataset_paths['church_night_train'],
		'test_t_source_root': dataset_paths['church_night_test'],
		'test_t_target_root': dataset_paths['church_night_test'],              
	},
}



medminist_path = '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets'
# /home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/octmnist/octmnist_224/train_X.npy
# /home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/bloodmnist/bloodmnist_224
# bratsMultiMod_root = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/brats/X_Y_splits/bratsMultiMod_{paired}"
# bratsMultiMod_root = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/brats/X_Y_splits/bratsMultiMod_{unpaired}"
med_data_path = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging"

Medical_DATASETS = {
	'chestExp1_xrays': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': dataset_paths['chestExp1_X_train'],
		'train_Y_root': dataset_paths['chestExp1_Y_train'],
		'test_X_root': dataset_paths['chestExp1_X_test'],
		'test_Y_root': dataset_paths['chestExp1_Y_test'],
	},

	'chestExp2_xrays': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': dataset_paths['chestExp2_X_train'],
		'train_Y_root': dataset_paths['chestExp2_Y_train'],
		'test_X_root': dataset_paths['chestExp2_X_test'],
		'test_Y_root': dataset_paths['chestExp2_Y_test'],
	},



	'Camelyon16': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': dataset_paths['Camelyon16_X_train'],
		'train_Y_root': dataset_paths['Camelyon16_Y_train'],
		'test_X_root': dataset_paths['Camelyon16_X_test'],
		'test_Y_root': dataset_paths['Camelyon16_Y_test'],
	}, 

	'isic': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': dataset_paths['isic_X_train'],
		'train_Y_root': dataset_paths['isic_Y_train'],
		'test_X_root': dataset_paths['isic_X_test'],
		'test_Y_root': dataset_paths['isic_Y_test'],
	},

	'ffhq': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': dataset_paths['ffhq_bg_train'],
		'train_Y_root': dataset_paths['ffhq_glass_train'],
		'test_X_root': dataset_paths['ffhq_bg_test'],
		'test_Y_root': dataset_paths['ffhq_glass_test'],
	},

	# ======================
	# MedMNIST datasets
	# ======================
	'bratsHT': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',
	}, 
 
	'bratsHTv1': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v1/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v1/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v1/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v1/test_Y.npy',
	}, 

	'bratsHTv2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v2/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v2/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v2/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v2/test_Y.npy',
	}, 


	'bratsHTv3': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v3/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v3/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v3/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v3/test_Y.npy',
	}, 

	'bratsHTv4': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v4/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v4/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v4/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_balanced_v4/test_Y.npy',
	}, 

	'bratsHT_new': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/train_X.npy',
		'train_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/train_Y.npy',
		'test_X_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_X.npy',
		'test_Y_root': med_data_path + '/brats/X_Y_splits/bratsHT_TUMOR_new/test_Y.npy',
	}, 


	'bloodmnist_x1y6': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/bloodmnist/bloodmnist_224/test_Y.npy',
	},
	'bloodmnist_x1y3': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/bloodmnist/split_x1_y3/test_Y.npy',
	},
	#############################################
	# 				Octmnist splits				#
	#############################################
	'octmnist_x0y1': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
	},

	'octmnist_x1y2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x1_y2/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x1_y2/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x1_y2/test_Y.npy',
	},

	'octmnist_x3y0': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x3_y0/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x3_y0/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y0/test_Y.npy',
	},

	'octmnist_x3y1': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x3_y1/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x3_y1/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y1/test_Y.npy',
	},

	'octmnist_x3y2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x3_y2/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x3_y2/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y2/test_Y.npy',
	},
	'octmnist_x3y012': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x3_y012/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x3_y012/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x3_y012/test_Y.npy',
	},	
	'octmnist_x0y1': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x0_y1/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y1/test_Y.npy',
	},	
	'octmnist_x0y2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/octmnist/split_x0_y2/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/octmnist/split_x0_y2/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/octmnist/split_x0_y2/test_Y.npy',
	},	
 
 
	'pathmnist': {
		'transforms': transforms_config.EncodeTransforms,
		'train_X_root': medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/train_X.npy',
		'train_Y_root': medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/train_Y.npy',
		'test_X_root':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_X.npy',
		'test_Y_root':  medminist_path + '/MedMNIST/pathmnist/pathmnist_224_class6_class8_npy/test_Y.npy',
	},


}


med_data_path = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging"
Medical_DATASETS['bratsMultiMod'] = {
    'transforms': transforms_config.EncodeTransforms,

    # PAIRED
    'train_X_t1n_paired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_X_t1n.npy',
    'test_X_t1n_paired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_X_t1n.npy',

    'train_Y_t1c_paired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t1c.npy',
    'test_Y_t1c_paired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t1c.npy',

    'train_Y_t2f_paired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2f.npy',
    'test_Y_t2f_paired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2f.npy',

    'train_Y_t2w_paired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/train_Y_t2w.npy',
    'test_Y_t2w_paired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_paired/test_Y_t2w.npy',


    # UNPAIRED
    'train_X_t1n_unpaired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_X_t1n.npy',
    'test_X_t1n_unpaired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_X_t1n.npy',

    'train_Y_t1c_unpaired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_Y_t1c.npy',
    'test_Y_t1c_unpaired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_Y_t1c.npy',

    'train_Y_t2f_unpaired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_Y_t2f.npy',
    'test_Y_t2f_unpaired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_Y_t2f.npy',

    'train_Y_t2w_unpaired': med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/train_Y_t2w.npy',
    'test_Y_t2w_unpaired':  med_data_path + '/brats/X_Y_splits/bratsMultiMod_unpaired/test_Y_t2w.npy',

	# /home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/octmnist/octmnist_224
    # 'bloodmnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/bloodmnist/bloodmnist_224',
    # 'bloodmnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/bloodmnist/bloodmnist_224_test',

    # 'chestmnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/chestmnist/chestmnist_224',
    # 'chestmnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/chestmnist/chestmnist_224_test',

    # 'octmnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/octmnist/octmnist_224',
    # 'octmnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/octmnist/octmnist_224_test',

    # 'tissuemnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/tissuemnist/tissuemnist_224',
    # 'tissuemnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/tissuemnist/tissuemnist_224_test',

    # 'pathmnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/pathmnist/pathmnist_224',
    # 'pathmnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/pathmnist/pathmnist_224_test',
    
    # 'organamnist_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/organamnist/organamnist_224',
    # 'organamnist_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/medical_datasets/MedMNIST/organamnist/organamnist_224_test',
}
