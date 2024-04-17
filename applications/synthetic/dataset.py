import numpy as np
import torch
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader


class RegressionDataset(torch.utils.data.Dataset):

    def __init__(self, num_samples, input_size):
        super().__init__()
        X = torch.randn(num_samples, input_size)
        y = X ** 2 + 15 * np.sin(X) **3
        y_t = torch.sum(y, dim=1)
        self._x = X
        self._y = y_t.unsqueeze(1)

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y


def get_loader(batch_size=1024, val_only=False):
    train_set = RegressionDataset(102400, 200)
    test_set = RegressionDataset(1024, 200)

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
