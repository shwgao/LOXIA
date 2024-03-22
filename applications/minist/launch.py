import torch
from utils import AverageMeter, accuracy

cretira = torch.nn.CrossEntropyLoss()


def loss_fn(output, target, model, reg=False):
    loss = cretira(output, target)
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def train_step(train_loader, model, criterion, optimizer, epoch, writer=None):
    """Train for one epoch on the training set"""

    losses = AverageMeter()
    l0_losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    for i, (data) in enumerate(train_loader):
        input_, target = data
        target = target.to(model.device)
        input_ = input_.to(model.device)

        # compute output
        output = model(input_)

        loss, l0_loss = criterion(output, target, model, reg=True)
        total_loss = loss + l0_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # clamp the parameters
        model.constrain_parameters()

        # measure accuracy and record loss
        corrects = accuracy(output.data, target, topk=(1,))[0].item()
        acc.update(corrects, input_.size(0))

        losses.update(loss.item(), 1)
        l0_losses.update(l0_loss.item(), 1)

        if model.module.beta_ema > 0.:
            model.module.update_ema()

        if i % 100 == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'L0Loss {l0_loss.val:.6f} ({l0_loss.avg:.6f})\t'
                  'Acc {acc.val:.6f} ({acc.avg:.6f})'.format(
                epoch, i, len(train_loader), loss=losses, l0_loss=l0_losses, acc=acc))

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('loss/train_l0', l0_losses.avg, epoch)
        writer.add_scalar('quality/train', acc.avg, epoch)

    return acc.avg


def validate(val_loader, model, criterion, inference=False):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    acc = AverageMeter()
    old_params = None

    # switch to evaluate mode
    model.eval()
    if model.beta_ema > 0 and not inference:
        old_params = model.module.get_params()
        model.module.load_ema_params()

    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            input_, target = data
            target = target.to(model.device)
            input_ = input_.to(model.device)

            # compute output
            output = model(input_)
            loss, _ = criterion(output, target, model, reg=False)

            # measure accuracy and record loss
            corrects = accuracy(output.data, target, topk=(1,))[0].item()
            acc.update(corrects, input_.size(0))

            losses.update(loss.item())

    print('Test: [{0}]\t'
          'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
          'Acc {acc.val:.6f} ({acc.avg:.6f})'.format(len(val_loader), loss=losses, acc=acc))

    if model.beta_ema > 0 and not inference:
        model.module.load_params(old_params)

    return acc.avg
