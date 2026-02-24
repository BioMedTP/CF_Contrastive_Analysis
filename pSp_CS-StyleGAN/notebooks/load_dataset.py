from configs import data_configs
from datasets.images_dataset import ImagesDataset
from torch.utils.data import DataLoader

def configure_datasets(opts):
    if opts.dataset_type not in data_configs.DATASETS.keys():
        Exception(f'{opts.dataset_type} is not a valid dataset_type')
    print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    train_bg_dataset = ImagesDataset(source_root=dataset_args['train_bg_source_root'],
                                    target_root=dataset_args['train_bg_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_gt_train'],
                                    opts=opts)
    
    train_t_dataset = ImagesDataset(source_root=dataset_args['train_t_source_root'],
                                    target_root=dataset_args['train_t_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_gt_train'],
                                    opts=opts)
            
    test_bg_dataset = ImagesDataset(source_root=dataset_args['test_bg_source_root'],
                                    target_root=dataset_args['test_bg_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_test'],
                                    opts=opts)
    
    test_t_dataset = ImagesDataset(source_root=dataset_args['test_t_source_root'],
                                    target_root=dataset_args['test_t_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_test'],
                                    opts=opts)

    print(f"Number of traing bg samples: {len(train_bg_dataset)}")
    print(f"Number of traing t samples: {len(train_t_dataset)}")
    print(f"Number of test bg samples: {len(test_bg_dataset)}")
    print(f"Number of test t samples: {len(test_t_dataset)}")
    return train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset

def get_dataset_loaders(opts):
    
    train_bg_dataset, train_t_dataset, test_bg_dataset, test_t_dataset = configure_datasets(opts)

    train_bg_dataloader = DataLoader(train_bg_dataset,
                                        batch_size=opts.batch_size,
                                        shuffle=True,
                                        num_workers=int(opts.workers),
                                        drop_last=True)

    train_t_dataloader = DataLoader(train_t_dataset,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                num_workers=int(opts.workers),
                                drop_last=True)

    test_bg_dataloader = DataLoader(test_bg_dataset,
                                        batch_size=opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(opts.test_workers),
                                        drop_last=True)

    test_t_dataloader = DataLoader(test_t_dataset,
                                        batch_size=opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(opts.test_workers),
                                        drop_last=True)
    
    return train_bg_dataloader, train_t_dataloader, test_bg_dataloader, test_t_dataloader