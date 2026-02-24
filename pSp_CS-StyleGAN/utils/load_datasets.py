
from utils.transforms import transforms_registry
#from datasets.loaders import InfiniteLoader
# from datasets import data_utils
# from training_cs.paths import DATASETS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

class InfiniteLoader(DataLoader):
    def __init__(
        self,
        *args,
        num_workers=0,
        pin_memory=True,
        is_infinite = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            multiprocessing_context="fork" if num_workers > 0 else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.dataset_iterator = super().__iter__()
        self.is_infinite = is_infinite

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            if self.is_infinite:
                x = next(self.dataset_iterator)
            else:
                raise StopIteration

        return x

# A simple Dataset that wraps a list of image paths
class ListDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def build_group_datasets(npz_path: str, transform_size: int, train: bool = True) -> dict:
    """
    Loads a .npz of precomputed image-path groups and returns a dict of four ListDatasets:
    'mg','mn','fg','fn' (male/female glasses/no-glasses).

    Args:
        npz_path (str): Path to the .npz file containing arrays:
                        'male_glasses','male_noglasses',
                        'female_glasses','female_noglasses'.
        transform_size (int): Size key for transforms_registry (e.g. 256).
        train (bool): If True, use training transforms and shuffling; else use test transforms.
    """
    data = np.load(npz_path)
    groups = {name: data[name].tolist() for name in data.files}

    # select transforms
    tf = transforms_registry[f"face_{transform_size}"]().get_transforms()
    chosen_tf = tf["train"] if train else tf["test"]

    return {
        "mg": ListDataset(groups["male_glasses"],   transform=chosen_tf),
        "mn": ListDataset(groups["male_noglasses"], transform=chosen_tf),
        "fg": ListDataset(groups["female_glasses"], transform=chosen_tf),
        "fn": ListDataset(groups["female_noglasses"], transform=chosen_tf),
    }


def build_loaders(opts, npz_path: str, train: bool = True) -> dict:
    """
    Returns a dict of four InfiniteLoader instances for each group,
    configured for training or testing.

    Args:
        opts: Namespace with attributes:
              - batch_size (int)
              - workers (int)
              - transform_size (int)
        npz_path (str): Path to the .npz file
        train (bool): If True, loaders are shuffled and infinite; else no shuffle.
    """
    datasets = build_group_datasets(npz_path, opts.transform_size, train=train)
    loaders = {}
    for key, ds in datasets.items():
        loaders[key] = InfiniteLoader(
            ds,
            batch_size=opts.batch_size,
            shuffle=train,
            num_workers=opts.workers,
            drop_last=train,
            is_infinite=train
        )
    return loaders

# def build_loaders(opts, npz_path: str, train: bool = True) -> dict:

#     # build four ListDatasets for mg/mn/fg/fn
#     datasets = build_group_datasets(npz_path, opts.transform_size, train=train)

#     # choose batch size & workers based on train vs test
#     bs     = opts.batch_size     
#     workers= opts.workers           
#     shuffle= True                    if train else False
#     drop   = True                    if train else False

#     loaders = {}
#     for key, ds in datasets.items():
#         loaders[key] = DataLoader(
#             ds,
#             batch_size   = bs,
#             shuffle      = shuffle,
#             num_workers  = int(workers),
#             drop_last    = drop,
#         )
#     return loaders

# Example usage:
# from training_cs.paths import DATASETS_NPZ
# train_npz = DATASETS_NPZ['glasses_gender']['train']
# test_npz  = DATASETS_NPZ['glasses_gender']['test']
#
# train_loaders = build_loaders(opts, train_npz, train=True)
# test_loaders  = build_loaders(opts, test_npz,  train=False)


