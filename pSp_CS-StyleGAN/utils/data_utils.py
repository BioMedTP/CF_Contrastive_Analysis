"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

import torchvision.transforms as transforms
import torch
from base_functions.base_funcs import load_folder_images, transform_images_to_batch, display_and_save_images, load_paths_images

root_path = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN"

Paths = {
    "ffhq_glasses": {"encoder_name": "hyperstyle",
        "code_dir": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/hyperstyle",
        "model_path": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/hyperstyle/results/ffhq_glasses/optimal/iter4_resume/checkpoints/iteration_100000.pt",
        #"restyle_pSp_cs": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/SOTA_encoders_StyleGAN/restyle/results/restyle_pSp/12layers_lr0.001/checkpoints/iteration_50000.pt",
        "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),    
        "images_x_path" : "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/special_images/background", 
        "images_y_path" : "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/special_images/glasses", 
        "device": "cuda"},

    "afhqv2": {
        "images_x_path" : root_path + '/AFHQ/afhq-v2/val/cat',
        "images_y_path" : root_path + '/AFHQ/afhq-v2/val/dog',
        "device": "cuda"},

    "ffhq_gender": {
        "images_x_path" : root_path + '/ffhq_cs_gender_age/test_data_male',
        "images_y_path" : root_path + '/ffhq_cs_gender_age/test_data_female',
        "device": "cuda"},

    "ffhq_age": {
        "images_x_path" : root_path + '/ffhq_cs_gender_age/test_data_young',
        "images_y_path" : root_path + '/ffhq_cs_gender_age/test_data_old',
        "device": "cuda"},
        
    "ffhq_pose": {
        "images_x_path" : root_path + '/ffhq_cs_headpose/test_pose_frontal',
        "images_y_path" : root_path + '/ffhq_cs_headpose/test_pose_left_right',
        "device": "cuda"},
        
    "ffhq_smile": {
        "images_x_path" : root_path + '/ffhq_cs_smile/test_smile_no',
        "images_y_path" : root_path + '/ffhq_cs_smile/test_smile_yes',
        "device": "cuda"},

    "glassesANDsmile": {
        "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),    
        "images_x_path" : "./ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/test_preview/noglasses_nonsmile", 
        "images_y_path" : "./ffhq_attrbutes/glasses_smile_vs_noglasses_nonsmile/test_preview/glasses_smile", 
        "device": "cuda"},

    "glassesVSsmile": {
        "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),    
        "images_x_path" : "./ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_glasses",
        "images_y_path" : "./ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_smile",
        "device": "cuda"},

    "glassesORsmile": {
        "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),    
        "images_x1_path" : "./ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_preview/neither_1", 
        "images_y1_path" : "./ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_preview/glasses_only", 
        
        "images_x2_path" : "./ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_preview/neither_2",
        "images_y2_path" : "./ffhq_attrbutes/glasses_or_smile_balanced_vs_neither/test_preview/smile_only", 
        "device": "cuda"},

    "glassesANDORsmile": {
        "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),    
        "images_x1_path" : "./ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/neither", 
        "images_y1_path" : "./ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/glasses_only", 
        "images_y2_path" : "./ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/smile_only",
        "images_y3_path" : "./ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/glasses_smile",
        "device": "cuda"},
    "celebaHQ_gender": {  
        "images_x_path" : root_path + "/CelebA-HQ/celebaHQ_special/gender/male", 
        "images_y_path" : root_path + "/CelebA-HQ/celebaHQ_special/gender/female", 
        "device": "cuda"},
    "celebaHQ_smile": {  
        "images_x_path" : root_path + "/CelebA-HQ/celebaHQ_special/smile/no_smile", 
        "images_y_path" : root_path + "/CelebA-HQ/celebaHQ_special/smile/smile", 
        "device": "cuda"},
}

def get_special_images(device, size, datasets="ffhq_glasses"):
    image_x_paths = Paths[datasets]["images_x_path"]
    image_y_paths = Paths[datasets]["images_y_path"]

    images_x, selected_paths_x = load_folder_images(image_x_paths, num_images=4, seed=None)
    images_y, selected_paths_y= load_folder_images(image_y_paths, num_images=4, seed=None)

    transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  

    input_x = transform_images_to_batch(images_x, transform).to(device).float()
    input_y = transform_images_to_batch(images_y, transform).to(device).float()

    return input_x, input_y

def get_special_paired_images(device, size, datasets="glassesORsmile"):
    
    image_x1_paths = Paths[datasets]["images_x1_path"]
    image_y1_paths = Paths[datasets]["images_y1_path"]
    image_x2_paths = Paths[datasets]["images_x2_path"]
    image_y2_paths = Paths[datasets]["images_y2_path"]

    images_x1, _ = load_folder_images(image_x1_paths, num_images=4, seed=None)
    images_y1, _= load_folder_images(image_y1_paths, num_images=4, seed=None)
    images_x2, _ = load_folder_images(image_x2_paths, num_images=4, seed=None)
    images_y2, _= load_folder_images(image_y2_paths, num_images=4, seed=None)

    transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  

    input_x1 = transform_images_to_batch(images_x1, transform).to(device).float()
    input_y1 = transform_images_to_batch(images_y1, transform).to(device).float()
    input_x2 = transform_images_to_batch(images_x2, transform).to(device).float()
    input_y2 = transform_images_to_batch(images_y2, transform).to(device).float()

    return input_x1, input_y1, input_x2, input_y2

def get_special_glasses_smile_images(device, size, datasets="glassesANDORsmile"):
    
    image_x1_paths = Paths[datasets]["images_x1_path"]
    image_y1_paths = Paths[datasets]["images_y1_path"]
    image_y2_paths = Paths[datasets]["images_y2_path"]
    image_y3_paths = Paths[datasets]["images_y3_path"]

    images_x1, _ = load_folder_images(image_x1_paths, num_images=4, seed=None)
    images_y1, _ = load_folder_images(image_y1_paths, num_images=4, seed=None)
    images_y2, _ = load_folder_images(image_y2_paths, num_images=4, seed=None)
    images_y3, _ = load_folder_images(image_y3_paths, num_images=4, seed=None)

    transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  

    input_x1 = transform_images_to_batch(images_x1, transform).to(device).float()
    input_y1 = transform_images_to_batch(images_y1, transform).to(device).float()
    input_y2 = transform_images_to_batch(images_y2, transform).to(device).float()
    input_y3 = transform_images_to_batch(images_y3, transform).to(device).float()

    return input_x1, input_y1, input_y2, input_y3