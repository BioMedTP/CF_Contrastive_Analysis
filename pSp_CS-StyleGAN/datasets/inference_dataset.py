import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import data_utils
import os

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im
      

class LatentClassifierDataset_Buffer(Dataset):
    def __init__(self, latents, labels):
        """
        Args:
            latents (torch.Tensor): Tensor of latent representations, shape (N, 18, 512).
            labels (torch.Tensor): Corresponding labels (0 for bg, 1 for target).
        """
        self.latents = latents  # Shape: (N, 18, 512)
        self.labels = labels    # Shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        latent_c = self.latents[index].view(-1)  # Flatten from (18, 512) → (9216)
        label = self.labels[index]
        return latent_c, torch.tensor(label, dtype=torch.long)

    
def create_dataloader_from_buffer(latent_bg_list, latent_t_list, label_list, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader from stored latent representations.

    Args:
        latent_bg_list (list of Tensors): Background latents.
        latent_t_list (list of Tensors): Target latents.
        label_list (list of Tensors): Corresponding labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of workers.

    Returns:
        DataLoader: DataLoader for training the classifier.
    """
    # Convert lists to tensors
    latents = torch.cat(latent_bg_list + latent_t_list, dim=0)  # Shape: (N, 18, 512)
    labels = torch.cat(label_list, dim=0)  # Shape: (N,)

    # Create dataset
    dataset = LatentClassifierDataset_Buffer(latents, labels)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader