import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import warnings


def get_data_path(data_root, dataname: str):
    if data_root is None:
        return None
    if os.path.isdir(os.path.join(data_root, dataname)):
        return os.path.join(data_root, dataname)
    if os.path.isdir(os.path.join(data_root, dataname.lower())):
        return os.path.join(data_root, dataname.lower())
    if os.path.isdir(os.path.join(data_root, dataname.upper())):
        return os.path.join(data_root, dataname.upper())
    raise ValueError(f'Error - Could not locate data {os.path.join(data_root, dataname)}')


def get_data(data_root, cache_root, dataname, image_size,
             dataset_type='train', regime=None, subset=None,
             use_cache=False, save_cache=False, pin_memory=False, in_memory=False,
             batch_size=None, drop_last=True, num_workers=0, ratio=None, shuffle=True, flip=False, permute=False):
    
    # Load real dataset
    if dataname.startswith('PGM'):
        from data.pgm_dataset import PGMDataset
        dataset = PGMDataset(get_data_path(data_root, dataname), get_data_path(cache_root, dataname),
                             dataset_type=dataset_type, regime=regime, subset=subset,
                             use_cache=use_cache, save_cache=save_cache, in_memory=in_memory,
                             image_size=image_size, transform=None, flip=flip, permute=False)
    

    # Load generated dataset for evaluation
    elif 'Generated' in dataname:
        origin = dataname.partition('#')[-1]
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor()])
        dataset = dsets.ImageFolder(data_root, transform=transform)
    
    # Reduce dataset to a smaller subset, nice for debugging
    if ratio is not None:
        old_len = len(dataset)
        import random
        indices = list(range(old_len))
        random.shuffle(indices)
        dataset = torch.utils.data.Subset(dataset, indices[:int(max(old_len * ratio, 5*batch_size))])
        warnings.warn(f'Reducing dataset size from {old_len} to {len(dataset)}')
    
    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)
    
    return dataloader