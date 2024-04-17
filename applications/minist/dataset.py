import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_loader(batch_size=100, pm=True, val_only=False):
    transf = [transforms.ToTensor()]

    def flatten(x):
        return x.view(784)

    if pm:
        transf.append(transforms.Lambda(flatten))

    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}

    train_loader = None
    if not val_only:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transform_data),
            batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform_data),
                                             batch_size=batch_size, shuffle=False, **kwargs)
    # num_classes = 10

    return train_loader, val_loader
