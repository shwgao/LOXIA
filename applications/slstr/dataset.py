import h5py
import numpy as np
import torch
from pathlib import Path
from typing import List, Union
from torch.utils.data import DataLoader, Dataset

N_CHANNELS = 9
IMAGE_H = 1200
IMAGE_W = 1500
PATCH_SIZE = 256


class SLSTRDataset(Dataset):
    def __init__(self, paths: Union[Path, List[Path]], single_image: bool = False, crop_size: int = 20):
        if isinstance(paths, Path):
            self.image_paths = list(Path(paths).glob('**/S3A*.hdf'))
        else:
            self.image_paths = paths

        self.single_image = single_image
        self.crop_size = crop_size
        self.patch_padding = 'valid' if not single_image else 'same'

    def _preprocess_images(self, img, msk, path):
        # Crop & convert to patches
        img = self._transform_image(img)
        msk = self._transform_image(msk)

        if self.single_image:
            return img, path
        else:
            return img, msk

    def _transform_image(self, img):
        # Crop to image which is divisible by the patch size
        # This also removes boarders of image which are all zero
        img = torch.from_numpy(img).float()

        offset_h = (IMAGE_H % PATCH_SIZE) // 2
        offset_w = (IMAGE_W % PATCH_SIZE) // 2
        target_h = IMAGE_H - offset_h * 2
        target_w = IMAGE_W - offset_w * 2
        if not self.single_image:
            img = img[offset_h:offset_h + target_h, offset_w:offset_w + target_w, :]

        # Convert image from IMAGE_H x IMAGE_W to PATCH_SIZE x PATCH_SIZE
        kernel_size = (PATCH_SIZE, PATCH_SIZE)
        stride = kernel_size
        if self.single_image:
            stride = (self.crop_size, self.crop_size)

        img = img.unsqueeze(0)  # Add batch dimension
        # b, h, w, c to b, c, h, w
        img = img.permute(0, 3, 1, 2)

        unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=(0, 0))
        patches = unfold(img)

        if not self.single_image:
            patches = patches.reshape(PATCH_SIZE, PATCH_SIZE, img.shape[1], -1).permute(3, 0, 1, 2)

        return patches

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with h5py.File(path, 'r') as handle:
            refs = handle['refs'][:]
            bts = handle['bts'][:]
            msk = handle['bayes'][:]

        bts = (bts - bts.mean()) / bts.std()
        refs = (refs - refs.mean()) / refs.std()
        img = np.concatenate([refs, bts], axis=-1)

        msk[msk > 0] = 1
        msk[msk <= 0] = 0
        msk = msk.astype(float)

        img, msk = self._preprocess_images(img, msk, path)

        # Convert numpy arrays to PyTorch tensors
        img_tensor = img.permute(0, 3, 2, 1)  # Convert to CxHxW
        msk_tensor = msk.permute(0, 3, 2, 1)  # Add channel dimension

        if self.single_image:
            return img_tensor, path
        else:
            return img_tensor, msk_tensor


def get_loader(batch_size=1024, val_only=False):
    data_paths = list(Path('/aul/homes/sgao014/sciml_bench/datasets/cloud_slstr_ds1/training').glob('**/S3A*.hdf'))

    train_dataset = SLSTRDataset(data_paths) if not val_only else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if not val_only else None

    data_paths = list(Path('/aul/homes/sgao014/sciml_bench/datasets/cloud_slstr_ds1/inference').glob('**/S3A*.hdf'))
    test_dataset = SLSTRDataset(data_paths)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
