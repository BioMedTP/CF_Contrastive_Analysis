
root_path = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN"
med_data_path = "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging"

dataset_paths = {
	'chestExp1_X_train': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_diseases/train_X.npy',
	'chestExp1_Y_train': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_diseases/train_Y.npy',
    'chestExp1_X_test': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_diseases/test_X.npy',
    'chestExp1_Y_test': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_diseases/test_Y.npy',    

	'chestExp2_X_train': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_5diseases/train_X.npy',
	'chestExp2_Y_train': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_5diseases/train_Y.npy',
    'chestExp2_X_test': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_5diseases/test_X.npy',
    'chestExp2_Y_test': med_data_path + '/chest_xrays/X_Y_splits/healthy_vs_5diseases/test_Y.npy', 

	'bratsHT_X_train': med_data_path + '/brats/X_Y_splits/bratsHT/train_X.npy',
	'bratsHT_Y_train': med_data_path + '/brats/X_Y_splits/bratsHT/train_Y.npy',
    'bratsHT_X_test': med_data_path + '/brats/X_Y_splits/bratsHT/test_X.npy',
    'bratsHT_Y_test': med_data_path + '/brats/X_Y_splits/bratsHT/test_Y.npy',

	'Camelyon16_X_train': med_data_path + '/Camelyon16/X_Y_split/train_X.npy',
	'Camelyon16_Y_train': med_data_path + '/Camelyon16/X_Y_split/train_Y.npy',
    'Camelyon16_X_test': med_data_path + '/Camelyon16/X_Y_split/test_X.npy',
    'Camelyon16_Y_test': med_data_path + '/Camelyon16/X_Y_split/test_Y.npy',

	'isic_X_train': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/train_X.npy',
	'isic_Y_train': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/train_Y.npy',
    'isic_X_test': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_X.npy',
    'isic_Y_test': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/Medical_imaging/ISIC2020/npy_splits/test_Y.npy',    
    
	'ffhq_male_train': root_path + '/ffhq_cs_gender_age/train_data_male',
	'ffhq_female_train': root_path + '/ffhq_cs_gender_age/train_data_female',
    'ffhq_male_test': root_path + '/ffhq_cs_gender_age/test_data_male',
    'ffhq_female_test': root_path + '/ffhq_cs_gender_age/test_data_female',
    
	'ffhq_young_train': root_path + '/ffhq_cs_gender_age/train_data_young',
	'ffhq_old_train': root_path + '/ffhq_cs_gender_age/train_data_old',
    'ffhq_young_test': root_path + '/ffhq_cs_gender_age/test_data_young',
    'ffhq_old_test': root_path + '/ffhq_cs_gender_age/test_data_old',

	'ffhq_frontal_train': root_path + '/ffhq_cs_headpose/train_pose_frontal',
    'ffhq_leftright_train': root_path + '/ffhq_cs_headpose/train_pose_left_right',
    'ffhq_frontal_test': root_path + '/ffhq_cs_headpose/test_pose_frontal',
	'ffhq_leftright_test': root_path + '/ffhq_cs_headpose/test_pose_left_right',

	'train_noglasses_nonsmile': './ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/train_noglasses_nonsmile.npy',
    'train_glasses_smile': './ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/train_glasses_smile.npy',
    'test_noglasses_nonsmile': './ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/test_noglasses_nonsmile.npy',
    'test_glasses_smile': './ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/test_glasses_smile.npy',

	'train_neither': './ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/train_neither.npy',
    'train_glassesORsmile': './ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/train_glassesORsmile.npy',
    'test_neither': './ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_neither.npy',
    'test_glassesORsmile': './ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_glassesORsmile.npy',

	'train_glasses_only': './ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_glasses_only.npy',
    'train_smile_only': './ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_smile_only.npy',
    'test_glasses_only': './ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_glasses_only.npy',
    'test_smile_only': './ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_smile_only.npy',

	'train_neither_GS': './ffhq_attrbutes/glasses_smile/train_neither_GS.npy',
    'train_GlassesSmile': './ffhq_attrbutes/glasses_smile/train_GlassesSmile.npy',
    'test_neither_GS': './ffhq_attrbutes/glasses_smile/test_neither_GS.npy',
    'test_GlassesSmile': './ffhq_attrbutes/glasses_smile/test_GlassesSmile.npy',

	'ffhq_nosmile_train': root_path + '/ffhq_cs_smile/train_smile_no',
	'ffhq_smile_train': root_path + '/ffhq_cs_smile/train_smile_yes',
    'ffhq_nosmile_test': root_path + '/ffhq_cs_smile/test_smile_no',
    'ffhq_smile_test': root_path + '/ffhq_cs_smile/test_smile_yes',  

	'ffhq_blond_brown_train': '../../../2_data/styleGAN/ffhq_cs_hairColor/train_blond_brown',
	'ffhq_black_train': '../../../2_data/styleGAN/ffhq_cs_hairColor/train_black',
    'ffhq_blond_brown_test': '../../../2_data/styleGAN/ffhq_cs_hairColor/test_blond_brown',
    'ffhq_black_test': '../../../2_data/styleGAN/ffhq_cs_hairColor/test_black',   
    
	'ffhq_bg_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_glasses/train_bg',
	'ffhq_glass_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_glasses/train_t',
    'ffhq_bg_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_glasses/test_bg',
    'ffhq_glass_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_glasses/test_t',

    'ffhq_bg_special': '../special_images/background',
    'ffhq_glass_special':'../special_images/glasses',

	'celebaHQ_male_train': root_path + '/CelebA-HQ/Gender/train_male',
	'celebaHQ_female_train': root_path + '/CelebA-HQ/Gender/train_female',
    'celebaHQ_male_test': root_path + '/CelebA-HQ/Gender/test_male',
    'celebaHQ_female_test': root_path + '/CelebA-HQ/Gender/test_female',
    
	'celebaHQ_smile_train': root_path + '/CelebA-HQ/Smiling/train_smile_no',
	'celebaHQ_nosmile_train': root_path + '/CelebA-HQ/Smiling/train_smile_yes',
    'celebaHQ_smile_test': root_path + '/CelebA-HQ/Smiling/test_smile_no',
    'celebaHQ_nosmile_test': root_path + '/CelebA-HQ/Smiling/test_smile_no',
    
	'celebaHQ_noLipstick_train': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/train_NoLip',
	'celebaHQ_Lipstick_train': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/train_WearLip',
    'celebaHQ_noLipstick_test': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/test_NoLip',
    'celebaHQ_Lipstick_test': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/test_WearLip',	
    
    'afhqcat_train': root_path +'/AFHQ/afhq-v2/train/cat',
	'afhqcat_test': root_path +'/AFHQ/afhq-v2/test/cat', 
    'afhqdog_train': root_path +'/AFHQ/afhq-v2/train/dog',
	'afhqdog_test': root_path +'/AFHQ/afhq-v2/test/dog', 

    'brats_healthy_train': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/train_healthy',
	'brats_healthy_test': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/test_healthy', 
    'brats_tumor_train': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/train_tumor',
	'brats_tumor_test': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/test_tumor', 

    'horse_only_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_horse/horse_only_train',
	'horse_only_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_horse/horse_only_test', 
    'horse_person_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_horse/horse_with_person_train',
	'horse_person_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_horse/horse_with_person_test', 

    'church_daylight_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_church/church_daylight_train',
	'church_daylight_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_church/church_daylight_test', 
    'church_night_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_church/church_night_train',
	'church_night_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/LSUN_church/church_night_test', 

}

model_paths = {
	'stylegan_ffhq': '../pretrained_models/pSp_models/stylegan2-ffhq1024.pt',
    'stylegan_brats': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/stylegan2-rosinality/checkpoint/800000.pt',
    'stylegan_brats2': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/stylegan2-rosinality/results/3gpus_bs16/checkpoints/420000.pt',
    'stylegan_afhqcat': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqcat-512x512.pt',
    'stylegan_afhqdog': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqdog-512x512.pt',
    'stylegan_afhqv2': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt',
    'stylegan_celebahq': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-celebahq-256x256.pt',
    
	'ir_se50': '../pretrained_models/pSp_models/model_ir_se50.pth',
	'circular_face': '../pretrained_models/pSp_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '../pretrained_models/pSp_models/mtcnn/pnet.npy',
	'mtcnn_rnet': '../pretrained_models/pSp_models/mtcnn/rnet.npy',
	'mtcnn_onet': '../pretrained_models/pSp_models/mtcnn/onet.npy',
	'shape_predictor': '../pretrained_models/pSp_models/shape_predictor_68_face_landmarks.dat',
	'moco': '../pretrained_models/pSp_models/moco_v2_800ep_pretrain.pth.tar',
    'previous_train_ckpt_path': None,
    #'previous_train_ckpt_path': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/results/ffhq_glasses/checkpoints/iteration_100000.pt'
}

