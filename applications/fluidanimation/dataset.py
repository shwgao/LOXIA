import numpy as np
import torch
import torch.utils.data.dataset as dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from utils import delete_constant_columns, reshape_vector, Load_Dataset_Surrogate


def read_data(input_file, output_file):
    LengthOfDataset = 3294120
    NUM_ATOMS = 3
    NUM_DIMENSIONS = 4
    with open('./Dataset/fluidanimation/inputvector_test.txt', "r") as f:
        num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDataset
        x2 = [
            float(value) for _ in range(num_lines) for value in f.readline().split()
        ]
    vector = reshape_vector(x2, 3, 4, LengthOfDataset)

    NUM_ATOMS = 8
    NUM_DIMENSIONS = 1
    # Read the program input
    with open('./Dataset/fluidanimation/inputcoef_test.txt', "r") as f:
        x1 = [
            float(f.readline())
            for _ in range(NUM_ATOMS * NUM_DIMENSIONS * LengthOfDataset)
        ]

    # Create X and Y lists using list comprehension
    coef = reshape_vector(x1, 8, 1, LengthOfDataset)

    NUM_ATOMS = 3
    NUM_DIMENSIONS = 1
    # Read the program input
    with open('./Dataset/fluidanimation/outputvector_test.txt', "r") as f:
        num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDataset
        x5 = [
            float(value) for _ in range(num_lines) for value in f.readline().split()
        ]

    X = [c + v for c, v in zip(coef, vector)]

    X = delete_constant_columns(np.array(X))
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Create X and Y lists using list comprehension
    prime_Y = np.array(reshape_vector(x5, 3, 1, LengthOfDataset))
    prime_X = np.array(X_normalized)
    return prime_X, prime_Y


def get_loader(batch_size=1024, val_only=False):
    input_file = ""
    output_file = ""

    data_set = Load_Dataset_Surrogate(input_file, output_file, read_data, application='fluidanimation', normalize=False)

    train_set_len = int(0.8 * len(data_set))
    train_set, test_set = dataset.random_split(data_set, [train_set_len, len(data_set) - train_set_len],
                                               generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
