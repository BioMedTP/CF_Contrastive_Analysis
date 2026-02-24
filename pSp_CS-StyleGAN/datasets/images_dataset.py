from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import os
import torch


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im




class ImagesDataset_folder(Dataset):

    def __init__(self, npy_path, opts, transform):
        # Load array of absolute file paths
        # self.paths = np.load(npy_path, allow_pickle=True).tolist()
        self.paths = sorted(data_utils.make_dataset(npy_path))
        self.opts = opts
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]

        # Load image using PIL only
        try:
            img = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}\nError: {e}")

        # Convert based on selected transform
        if "rgb" in self.opts.data_transform.lower():
            img = img.convert("RGB")
        else:
            img = img.convert("L")  # grayscale

        # Apply transforms
        img = self.transform(img)
        return img



class PairedNpyDataset(Dataset):
    def __init__(self, X_npy_path, Y_npy_path, transform=None, opts=None):
        self.X_paths = np.load(X_npy_path, allow_pickle=True)
        self.Y_paths = np.load(Y_npy_path, allow_pickle=True)
        assert len(self.X_paths) == len(self.Y_paths), "Lengths mismatch!"
        self.transform = transform
        self.opts = opts

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        X = np.load(self.X_paths[idx], allow_pickle=True)
        Y = np.load(self.Y_paths[idx], allow_pickle=True)

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return X, Y


class ImagesDataset_mednpy(Dataset):

    def __init__(self, npy_path, opts, transform):
        # Load array of absolute file paths
        self.paths = np.load(npy_path, allow_pickle=True).tolist()
        self.opts = opts
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]

        # Load image using PIL only
        try:
            img = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}\nError: {e}")

        # Convert based on selected transform
        if "rgb" in self.opts.data_transform.lower():
            img = img.convert("RGB")
        else:
            img = img.convert("L")  # grayscale

        # Apply transforms
        img = self.transform(img)
        return img



class ImagesDataset_npy(Dataset):
    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        self.image_root = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/ffhq_1024"

        # Load .npy files (each contains list of relative image filenames)
        self.source_paths = np.load(source_root, allow_pickle=True).tolist()
        self.target_paths = np.load(target_root, allow_pickle=True).tolist()

        assert len(self.source_paths) == len(self.target_paths), \
            f"Source and target lengths mismatch: {len(self.source_paths)} vs {len(self.target_paths)}"
        
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        # Join with image_root to get full path
        from_path = os.path.join(self.image_root, self.source_paths[index])
        to_path   = os.path.join(self.image_root, self.target_paths[index])

        # Load and convert images
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        to_im = Image.open(to_path).convert('RGB')

        # Apply transforms
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im

     

class ImagePathsDataset(Dataset):
    def __init__(self, npy_path, transform=None):
        """
        Args:
            npy_path (str): Path to the .npy or .npz file containing image paths.
            transform (callable, optional): Optional transform to apply to PIL images.
            color_mode (str): 'RGB' or 'L' (grayscale)
        """
        # Load paths (handles both .npy and .npz)
        if npy_path.endswith(".npz"):
            # load all keys and flatten the image paths
            data = np.load(npy_path)
            self.paths = []
            for k in data.files:
                self.paths.extend(data[k].tolist())
        else:
            self.paths = np.load(npy_path).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        img = img.convert('RGB') if self.opts.label_nc == 0 else img.convert('L')

        if self.transform:
            img = self.transform(img)

        return img

class ImagesDatasets2(Dataset):

    def __init__(self, image_dirs_file, labels_file, transform=None):
        """
        Args:
            image_dirs_file (str): Path to the .npy file containing image paths.
            labels_file (str): Path to the .npy file containing corresponding labels.
            opts (dict): Additional options for the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load image paths and labels from .npy files
        self.image_paths = np.load(image_dirs_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Tuple: (image, label) where image is the transformed image tensor and label is the corresponding age.
        """
        # Retrieve the image path
        image_path = self.image_paths[index]

        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and its label
        return image, self.labels[index]
	
class ImagesDatasets_cls(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_dirs_file (str): Path to the .npy file containing image paths.
            labels_file (str): Path to the .npy file containing corresponding labels.
            opts (dict): Additional options for the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load image paths and labels from .npy files
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Tuple: (image, label) where image is the transformed image tensor and label is the corresponding age.
        """
        # Retrieve the image path
        image_path = self.image_paths[index]

        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and its label
        return image, self.labels[index]
    

class ImagesDatasets_cls_fnames(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_dirs_file (str): Path to the .npy file containing image paths.
            labels_file (str): Path to the .npy file containing corresponding labels.
            opts (dict): Additional options for the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load image paths and labels from .npy files
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            Tuple: (image, label) where image is the transformed image tensor and label is the corresponding age.
        """
        # Retrieve the image path
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path) 

        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and its label
        return image, self.labels[index], image_name

