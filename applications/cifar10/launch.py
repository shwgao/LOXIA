import torch
from utils import accuracy

criteria = torch.nn.CrossEntropyLoss()


def loss_fn(output, target, model, reg=False):
    loss = criteria(output, target)
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    corrects = accuracy(output.data, target, topk=(1,))[0].item()
    quality.update(corrects, input_size)
