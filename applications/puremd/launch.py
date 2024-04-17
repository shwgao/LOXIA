import torch
from sklearn.metrics import r2_score
import numpy as np

criteria = torch.nn.MSELoss()


def loss_fn(output, target, model, reg=False):
    loss = criteria(output, target)
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    r2_s = r2_score(target.cpu(), output.detach().cpu().numpy(), multioutput='raw_values')
    r2_s = np.clip(r2_s, -1, 1)
    quality.update(np.mean(r2_s), 1)
