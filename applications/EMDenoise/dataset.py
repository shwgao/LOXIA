import numpy as np
import torch
import torch.utils.data.dataset as dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utils import delete_constant_columns, reshape_vector, Load_Dataset_Surrogate


def read_data(input_file, output_file):
    LengthOfDataset = 2329104
    NUM_ATOMS = 35
    NUM_DIMENSIONS = 1
    with open(input_file, "r") as f:
        xyz = [
            float(f.readline())
            for _ in range(NUM_ATOMS * NUM_DIMENSIONS * LengthOfDataset)
        ]

    # Create X and Y lists using list comprehension
    X = reshape_vector(xyz, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    with open(output_file, "r") as f:
        out = [float(f.readline()) for _ in range(LengthOfDataset * 5)]

    NUM_ATOMS = 5
    NUM_DIMENSIONS = 1

    X = delete_constant_columns(np.array(X))
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    Y = reshape_vector(out, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    prime_X = np.array(X_normalized)
    prime_Y = np.array(Y)
    return prime_X, prime_Y


def get_loader(batch_size=1024, val_only=False):
    input_file = "./Dataset/CFD/input.txt"
    output_file = "./Dataset/CFD/output.txt"

    data_set = Load_Dataset_Surrogate(input_file, output_file, read_data, application='CFD', normalize=False)

    train_set_len = int(0.8 * len(data_set))
    train_set, test_set = dataset.random_split(data_set, [train_set_len, len(data_set) - train_set_len],
                                               generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
