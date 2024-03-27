import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_ds_from_dir_single(path):
    tensor_x = []
    tensor_y = []
    data_file = [name for name in os.listdir(path) if name.endswith("data.npy")]
    for name_file in data_file:
        path_file = os.path.join(path, name_file)
        x = np.load(path_file, allow_pickle=True)
        x = x.reshape(128, 128, 128, -1)
        x = np.rollaxis(x, 3, 0)
        x = x.astype(np.float32) / 500
        y = np.load(path_file.replace("data.npy", "label.npy"), allow_pickle=True)
        y = y.astype(np.float32) / 2 + 0.5
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)

    return tensor_x, tensor_y


class Load_Dataset_Cosmoflow(Dataset):
    def __init__(self, path):
        self.tensor_x, self.tensor_y = load_ds_from_dir_single(path)
        self.tensor_x = self.tensor_x
        self.tensor_y = self.tensor_y

    def __getitem__(self, index):
        return self.tensor_x[index], self.tensor_y[index]

    def __len__(self):
        return len(self.tensor_x)

    def analyse_data(self):
        print(
            f"mean: {np.mean([torch.mean(x) for x in self.tensor_x])}, std: {np.std([torch.std(x) for x in self.tensor_x])}"
        )
        print(
            f"max: {np.max([torch.max(x) for x in self.tensor_x])}, min: {np.min([torch.min(x) for x in self.tensor_x])}"
        )
        print(
            f"mean: {np.mean([torch.mean(x) for x in self.tensor_y])}, std: {np.std([torch.std(x) for x in self.tensor_y])}"
        )
        print(
            f"max: {np.max([torch.max(x) for x in self.tensor_y])}, min: {np.min([torch.min(x) for x in self.tensor_y])}"
        )

        np_x = np.stack([x.numpy() for x in self.tensor_x])

        # Count the elements of a high-dimensional array np_x and draw a histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(
            np_x.flatten(),
            bins=100,
            color="skyblue",
            edgecolor="black",
            range=(0.02, 1),
        )
        # 设置标题和标签
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def get_loader(batch_size=100, val_only=False):
    train_set = Load_Dataset_Cosmoflow("/aul/homes/sgao014/datasets/cosmoflow/train")
    test_set = Load_Dataset_Cosmoflow("/aul/homes/sgao014/datasets/cosmoflow/validation")

    train_loader = None
    if not val_only:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
