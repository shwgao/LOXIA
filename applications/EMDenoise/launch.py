import torch

criteria = torch.nn.MSELoss()


def loss_fn(output, target, model, reg=False):
    loss = criteria(output, target)
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    mse = criteria(output, target)
    quality.update(mse.item(), 1)
