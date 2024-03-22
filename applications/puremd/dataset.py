import numpy as np
import torch
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader

from utils import reshape_vector, Load_Dataset_Surrogate


def read_data(input_file, output_file):
    LengthOfDataset = 1040407
    NUM_ATOMS = 4
    NUM_DIMENSIONS = 1
    LengthOfDatast = LengthOfDataset
    # Read the program input
    with open("./Dataset/puremd/input_inner_loop_data3/C2dbo.txt", "r") as f:
        x1 = [
            float(f.readline())
            for _ in range(NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast)
        ]

    # Create X and Y lists using list comprehension
    C3dbo = reshape_vector(x1, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    NUM_ATOMS = 3
    NUM_DIMENSIONS = 1
    # Read the program input
    with open("./Dataset/puremd/input_inner_loop_data3/nbr_k_bo_data2.txt", "r") as f:
        num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
        x3 = [
            float(value) for _ in range(num_lines) for value in f.readline().split()
        ]

    nbr_k = reshape_vector(x3, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    NUM_ATOMS = 3
    NUM_DIMENSIONS = 1
    # Read the program input
    with open("./Dataset/puremd/input_inner_loop_data3/temp.txt", "r") as f:
        num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
        x4 = [
            float(value) for _ in range(num_lines) for value in f.readline().split()
        ]

    temp = reshape_vector(x4, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    NUM_ATOMS = 3
    NUM_DIMENSIONS = 1
    # Read the program output
    with open("./Dataset/puremd/input_inner_loop_data3/temp_output_2.txt", "r") as f:
        num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
        x4 = [float(value) for _ in range(num_lines) for value in f.readline().split()]

    temp_output = reshape_vector(x4, NUM_ATOMS, NUM_DIMENSIONS, LengthOfDataset)

    prime_X = [
        c2dbo_row + nbr_k_row + temp_row
        for c2dbo_row, nbr_k_row, temp_row in zip(C3dbo, temp, nbr_k)
    ]

    prime_X = np.array(prime_X)[:-400000, :]
    prime_Y = np.array(temp_output)[:-400000, :]

    return prime_X, prime_Y


def get_loader(batch_size=100, val_only=False):
    input_file = ""
    output_file = ""

    data_set = Load_Dataset_Surrogate(input_file, output_file, read_data, application='puremd', normalize=False)

    train_set_len = int(0.8 * len(data_set))
    train_set, test_set = dataset.random_split(data_set, [train_set_len, len(data_set) - train_set_len],
                                               generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
