import os
import numpy as np
import torch
import h5py
import torch.utils.data.dataset as dataset
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader


class DMSDataset(dataset.Dataset):
    """
    A generic zipped dataset loader for DMS Structure
    """
    def __init__(self, training=True):

        base_dataset_dir = '/aul/homes/sgao014/sciml_bench/datasets/dms_sim'

        dataset_path = os.path.join(base_dataset_dir, 'training/data-binary.h5')
        hf = h5py.File(dataset_path, 'r')
        onehot_encoder = OneHotEncoder(sparse=False)
        if training:
            img = hf['train/images'][:]
            img = np.swapaxes(img, 1, 3)
            self.X = torch.from_numpy(np.atleast_3d(img))
            lab = np.array(hf['train/labels']).reshape(-1, 1)
            lab = onehot_encoder.fit_transform(lab).astype(int)
            self.Y = torch.from_numpy(lab).float()
        else:
            img = hf['test/images'][:]
            img = np.swapaxes(img, 1, 3)
            self.X = torch.from_numpy(np.atleast_3d(img))
            lab = np.array(hf['test/labels']).reshape(-1, 1)
            lab = onehot_encoder.fit_transform(lab).astype(int)
            self.Y = torch.from_numpy(lab).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_loader(batch_size=1024, val_only=False):
    train_set = DMSDataset(training=True)
    test_set = DMSDataset(training=False)

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
