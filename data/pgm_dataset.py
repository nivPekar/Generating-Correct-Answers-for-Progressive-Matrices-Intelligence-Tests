import os
import random
import glob
import numpy as np
import skimage.transform
import skimage.io

import torch
from torch.utils.data import Dataset

import warnings


class ToTensor(object):
    def __call__(self, sample):
        to_tensor(sample)


def to_tensor(sample):
    return torch.tensor(sample, dtype=torch.float32)


class PGMDataset(Dataset):
    def __init__(self, root, cache_root, dataset_type=None, regime='neutral', image_size=80, transform=None,
                 use_cache=True, save_cache=False, cache_mode='npz', in_memory=False, subset=None, flip=False, permute=False):
        self.root = root
        self.cache_root = cache_root if cache_root is not None else root
        self.dataset_type = dataset_type
        self.regime = regime if regime is not None else 'neutral'
        self.image_size = image_size
        self.transform = transform
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_mode = cache_mode
        self.flip = flip
        self.permute = permute
        
        self.data_flat = False  # if self.regime == 'neutral' else True
        self.fake_data = False
        
        if self.fake_data:
            warnings.warn('Using fake_data=True. You will not load any actual data')
        
        assert self.root is not None or self.cache_root is not None
        assert self.root is not None or self.use_cache
        assert not (use_cache and self.data_flat)
        assert not (use_cache and self.fake_data)
        
        def set_paths():
            if self.root is not None:
                if self.data_flat:
                    self.data_dir = os.path.join(self.root, 'data', regime)
                else:
                    self.data_dir = os.path.join(self.root, 'data', regime, self.dataset_type)
            else:
                self.data_dir = None
            if self.use_cache:
                assert cache_mode is not None
                self.cached_dir = os.path.join(self.cache_root, f'cache_{cache_mode}', regime,
                                               f'{self.dataset_type}_{self.image_size}')
        set_paths()
        
        if subset is not None:
            if dataset_type == "train":
                position_file_names_place = os.path.join('files', 'pgm', f'{subset}_train.txt')
            else:
                self.dataset_type = "test"
                set_paths()
                position_file_names_place = os.path.join('files', 'pgm', f'{subset}_test.txt')
            
            with open(position_file_names_place, "r") as file:
                contents = file.read()
                self.file_names = contents.splitlines()
            
            if self.data_flat:
                self.file_names = [os.path.basename(f) for f in self.file_names]
        else:
            data_dir = self.data_dir if self.data_dir is not None else self.cached_dir
            if self.data_flat:
                self.file_names = [f for f in os.listdir(data_dir) if dataset_type in f]
                self.file_names.sort()
            else:
                self.file_names = []
                for i in os.listdir(data_dir):
                    # file_names = [f for f in glob.glob(os.path.join(self.data_dir, i, "*.npz"))]
                    # file_names = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
                    #               for f in file_names]
                    file_names = [os.path.join(i, f) for f in os.listdir(os.path.join(data_dir, i))]
                    file_names.sort()
                    self.file_names += file_names
            
            # Sanity
            assert subset != 'train' or len(self.file_names) == 1200000, f'Train length = {len(self.file_names)}'
            assert subset != 'val' or len(self.file_names) == 20000, f'Validation length = {len(self.file_names)}'
            assert subset != 'test' or len(self.file_names) == 200000, f'Test length = {len(self.file_names)}'
    
        self.memory = None
        if in_memory:
            self.memory = [None] * len(self.file_names)
            from tqdm import tqdm
            for idx in tqdm(range(len(self.file_names)), 'Loading Memory'):
                image, data, _ = self.get_data(idx)
                d = {'target': data["target"],
                     'meta_target': data["meta_target"],
                     'relation_structure': data["relation_structure"],
                     'relation_structure_encoded': data["relation_structure_encoded"]
                     }
                self.memory[idx] = (image, d)
                del data
                
            
    def save_image(self, image, file):
        image = image.numpy()
        os.makedirs(os.path.dirname(file), exist_ok=True)
        image_file = os.path.splitext(file)[0] + '.png'
        skimage.io.imsave(image_file, image.reshape(self.image_size, self.image_size))
    
    def load_image(self, file):
        image_file = os.path.splitext(file)[0] + '.png'
        gen_image = skimage.io.imread(image_file).reshape(1, self.image_size, self.image_size)
        if self.transform:
            gen_image = self.transform(gen_image)
        gen_image = to_tensor(gen_image)
        return gen_image
    
    def load_cached_file(self, file):
        try:
            if self.cache_mode == 'png':
                image_file = os.path.splitext(file)[0] + '.png'
                image = skimage.io.imread(image_file).reshape(16, self.image_size, self.image_size)
                data = np.load(file)
            elif self.cache_mode == 'npz':
                data = np.load(file)
                image = data['image']
            return image, data
        except:
            raise ValueError(f'Error - Could not open existing file {file}')
    
    def save_cached_file(self, file, image, data):
        # print('saving to cache')
        if self.cache_mode == 'png':
            # png cache
            png_file = file.format('cache_png')
            os.makedirs(os.path.dirname(png_file), exist_ok=True)
            image_file = os.path.splitext(png_file)[0] + '.png'
            skimage.io.imsave(image_file, image.reshape(4 * self.image_size, 4 * self.image_size))
            np.savez(png_file, **data)
        
        elif self.cache_mode == 'npz':
            # npz cache
            npz_file = file.format('cache_npz')
            os.makedirs(os.path.dirname(npz_file), exist_ok=True)
            data['image'] = image
            np.savez_compressed(npz_file, **data)
    
    def __len__(self):
        return len(self.file_names)
    
    def get_data(self, idx):
        data_file = self.file_names[idx]
        if self.memory is not None and self.memory[idx] is not None:
            resize_image, data = self.memory[idx]
        else:
            no_cache = True
            # Try to load a cached file for faster fetching
            if self.use_cache:
                cached_path = os.path.join(self.cached_dir, data_file)
                if os.path.isfile(cached_path):
                    resize_image, data = self.load_cached_file(cached_path)
                    no_cache = data is None
                if no_cache and not self.save_cache:
                    raise ValueError('Error - Expected to load cached data but cache was not found')
            # Load original file otherwise
            if no_cache:
                data_path = os.path.join(self.data_dir, data_file)
                data = np.load(data_path)
                if self.fake_data:
                    resize_image = np.zeros([16, self.image_size, self.image_size])
                    data = {'target': 0, 'meta_target': np.array([0] * 12, dtype=np.int64)}
                else:
                    image = data["image"].reshape(16, 160, 160)
                    if self.image_size != 160:
                        resize_image = []
                        for idx in range(0, 16):
                            # resize_image.append(misc.imresize(image[idx,:,:], (self.image_size, self.image_size)))
                            resize_image.append(
                                skimage.transform.resize(image[idx, :, :], (self.image_size, self.image_size),
                                                         order=1, preserve_range=True, anti_aliasing=True))
                        resize_image = np.stack(resize_image, axis=0).astype(np.uint8)
                    else:
                        resize_image = image.astype(np.uint8)
                        
                    # Optional: save a cached file for further use
                    if self.use_cache:
                        if self.save_cache:
                            # print('Saving to cache')
                            os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                            d = {'target': data["target"],
                                 'meta_target': data["meta_target"],
                                 'relation_structure': data["relation_structure"],
                                 'relation_structure_encoded': data["relation_structure_encoded"]
                                 }
                            self.save_cached_file(cached_path, resize_image, d)
                        else:
                            raise ValueError(f'Error cache file {cached_path} not found')
        
        return resize_image, data, data_file
    
    def __getitem__(self, idx):
        resize_image, data, data_file = self.get_data(idx)
        
        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure = data["relation_structure"]

        new_relation_structure = ["None", "None", "None", "None"]
        for i in range(0, len(structure)):
            relation_string = structure[i][0].decode('UTF-8') + " " + structure[i][1].decode(
                'UTF-8') + " " + structure[i][2].decode('UTF-8')
            new_relation_structure[i] = relation_string

        structure_encoded = data["relation_structure_encoded"]
        del data
        
        if self.transform:
            resize_image = self.transform(resize_image)
        resize_image = to_tensor(resize_image)
        
        if self.flip:
                if random.random() > 0.5:
                    resize_image[[0,1,2,3,4,5,6,7]] = resize_image[[0,3,6,1,4,7,2,5]]
             
        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                resize_image[[8+new_target, 8+target]] = resize_image[[8+target, 8+new_target]]
                target = new_target
        
        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure = [structure.tolist()]
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)
        
        return resize_image, target, meta_target, new_relation_structure, structure_encoded, data_file
