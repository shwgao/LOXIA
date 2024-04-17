import torch
import numpy as np
from sklearn.metrics import accuracy_score

criteria = torch.nn.BCELoss()


def loss_fn(output, target, model, reg=False):
    loss = criteria(output, target)
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    threshold_results = np.where(output.detach().cpu().numpy() > 0.5, 1, 0)
    quality.update(accuracy_score(threshold_results, target.detach().cpu().numpy()), 1)
