import os
import h5py
import numpy as np
import torch
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader


class EMDenoiseTrainingDataset(torch.utils.data.Dataset):
    """
    A generic zipped dataset loader for EMDenoiser
    """

    def __init__(self, noisy_file_path, clean_file_path):
        self.noisy_file_path = noisy_file_path
        self.clean_file_path = clean_file_path
        self.dataset_len = 0
        self.noisy_dataset = None
        self.clean_dataset = None

        with h5py.File(self.noisy_file_path, 'r') as hdf5_file:
            len_noisy = len(hdf5_file["images"])
        with h5py.File(self.clean_file_path, 'r') as hdf5_file:
            len_clean = len(hdf5_file["images"])

        with h5py.File(self.noisy_file_path, 'r') as hdf5_file:
            self.noisy_dataset = torch.from_numpy(np.array(hdf5_file["images"]))
        with h5py.File(self.clean_file_path, 'r') as hdf5_file:
            self.clean_dataset = torch.from_numpy(np.array(hdf5_file["images"]))

        # swap axes to get the correct shape
        self.noisy_dataset = torch.swapaxes(self.noisy_dataset, 3, 1)
        self.clean_dataset = torch.swapaxes(self.clean_dataset, 3, 1)

        self.dataset_len = min(len_clean, len_noisy)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.noisy_dataset is None:
            self.noisy_dataset = h5py.File(self.noisy_file_path, 'r')["images"]
        if self.clean_dataset is None:
            self.clean_dataset = h5py.File(self.clean_file_path, 'r')["images"]
        return self.noisy_dataset[index], self.clean_dataset[index]


def get_loader(batch_size=1024, val_only=False):
    base_dataset_dir = '/aul/homes/sgao014/sciml_bench/datasets/em_graphene_sim/train'
    noisy_path = os.path.join(base_dataset_dir, 'graphene_img_noise.h5')
    clean_path = os.path.join(base_dataset_dir, 'graphene_img_clean.h5')
    train_set = EMDenoiseTrainingDataset(noisy_path, clean_path)
    base_dataset_dir = '/aul/homes/sgao014/sciml_bench/datasets/em_graphene_sim/test'
    noisy_path = os.path.join(base_dataset_dir, 'graphene_img_noise.h5')
    clean_path = os.path.join(base_dataset_dir, 'graphene_img_clean.h5')
    test_set = EMDenoiseTrainingDataset(noisy_path, clean_path)

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) if not val_only else None

    return train_loader, val_loader
