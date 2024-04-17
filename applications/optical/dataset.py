import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

IMAGE_SHAPE = (200, 200, 1)


def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x


def load_images(file_path):
    # List all TIFF files in the directory
    file_names = list(Path(file_path).glob('*.TIFF'))

    images = np.zeros((len(file_names), *IMAGE_SHAPE))
    for index, file_name in enumerate(tqdm(file_names)):
        img = Image.open(file_name)

        # A numpy array containing the tiff data
        image = np.array(img)
        image = image.astype(np.float32)
        image = normalize(image)

        # crop image around optic
        image = image[150:350, 270:470]
        image = np.expand_dims(image, axis=-1)
        images[index] = image

    return images


class OpticalDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root='training'):
        base_dataset_dir = '/aul/homes/sgao014/sciml_bench/datasets/optical_damage_ds1'
        training_path_x = os.path.join(base_dataset_dir, root, 'damaged')
        self.images_x = torch.from_numpy(load_images(training_path_x)).float()
        # swap axes
        self.images_x = self.images_x.permute(0, 3, 1, 2)
        self.images_y = self.images_x.clone()

    def __len__(self):
        return len(self.images_x)

    def __getitem__(self, idx):
        return self.images_x[idx], self.images_y[idx]


def get_loader(batch_size=1024, val_only=False):
    train_set = None if val_only else OpticalDamageDataset(root='training')
    test_set = OpticalDamageDataset(root='inference')

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) if not val_only else None

    return train_loader, val_loader
