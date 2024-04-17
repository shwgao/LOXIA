import math

import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from base_layers import L0Conv2d, L0Conv3d, L0Dense


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates=99999):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.avg = None
        self.sum = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Load_Dataset_Surrogate(Dataset):
    def __init__(
            self,
            input_file,
            output_file,
            read_data,
            application="fluid",
            device=None,
            normalize=False,
    ):
        self.prime_X, self.prime_Y = read_data(input_file, output_file)
        self.application = application
        self.mean_Y = None
        self.std_Y = None
        self.indices = None  # indices of the selected features
        self.x = torch.tensor(self.prime_X, dtype=torch.float32)  # input of dataset for dataloader
        self.y = torch.tensor(self.prime_Y, dtype=torch.float32)  # output of dataset for dataloader
        # standardize prime_X and prime_Y, and assign them to x and y
        self.set_mean_std()
        if normalize:
            self.standardize_data()
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

    def standardize_data(self):
        X_standardized = self.prime_X
        Y_standardized = self.prime_Y

        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        X_normalized = scaler.fit_transform(X_standardized)
        Y_normalized = scaler.fit_transform(Y_standardized)

        self.x = torch.tensor(X_normalized, dtype=torch.float32)
        self.y = torch.tensor(Y_normalized, dtype=torch.float32)

    def get_data_from_file(self, x_path, y_path):
        # Read data from file in file_path, each line is a vector
        if isinstance(x_path, str):
            with open(x_path, "r") as f:
                data_lines = f.readlines()
                x = [[float(value) for value in line.split()] for line in data_lines]
            with open(y_path, "r") as f:
                data_lines = f.readlines()
                y = [[float(value) for value in line.split()] for line in data_lines]
            self.prime_X = np.array(x)
            self.prime_Y = np.array(y)
        elif isinstance(x_path, list):
            # if there is multiple files, we need to read them one by one
            for i in range(len(x_path)):
                with open(x_path[i], "r") as f:
                    data_lines = f.readlines()
                    x = [[float(value) for value in line.split()] for line in data_lines]
                with open(y_path[i], "r") as f:
                    data_lines = f.readlines()
                    y = [[float(value) for value in line.split()] for line in data_lines]
                if i == 0:
                    self.prime_X = np.array(x)
                    self.prime_Y = np.array(y)
                else:
                    self.prime_X = np.concatenate((self.prime_X, np.array(x)), axis=0)
                    self.prime_Y = np.concatenate((self.prime_Y, np.array(y)), axis=0)

    def filter_data(self, filters, filter_method):
        pass

    def set_mean_std(self):
        """Set the mean and std of the dataset, used for standardization and recover"""
        if self.application == "fluidanimation":
            self.mean_Y = [0.12768913, 0.05470352, 0.14003364]
            self.std_Y = [2.07563408, 1.59399168, 2.06319435]
        elif self.application == "CFD":
            self.mean_Y = [
                -2.01850154e-08,
                -5.23806580e-11,
                7.29894413e-12,
                5.15219587e-12,
                1.15924407e-11,
            ]
            self.std_Y = [0.38392921, 0.12564681, 0.12619844, 0.21385977, 0.68862844]
        elif self.application == "puremd":
            self.mean_Y = [
                1.8506536384110004e-06,
                -0.003247667874206878,
                0.0007951518742184539,
            ]
            self.std_Y = [0.30559314964628986, 0.44421521232555966, 0.4909015024281119]
        else:
            print("Application error")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def delete_constant_columns(np_array):
    # find out constant columns of X_train
    is_constant = np.all(np_array == np_array[0, :], axis=0)
    # get the indices of the constant columns
    cc = np.where(is_constant == True)
    # get rid of the constant columns in X_train and X_test
    np_array = np.delete(np_array, cc, axis=1)
    return np_array


def reshape_vector(vector, num_atoms, num_dimensions, LengthOfDataset):
    array = [
        [
            vector[(num_atoms * num_dimensions * j) + (k * num_dimensions) + d]
            for k in range(num_atoms)
            for d in range(num_dimensions)
        ]
        for j in range(LengthOfDataset)
    ]
    return array


def to_one_hot(x, n_cats=10):
    y = np.zeros((x.shape[0], n_cats))
    y[np.arange(x.shape[0]), x] = 1
    return y.astype(np.float32)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def relative_error(output, target):
    """Computes the average relative error"""
    return torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))


def relative_error_1(output, target):
    """Computes the average relative error"""
    target_var = torch.zeros(target.size(0), 10).to(output.device).scatter_(1, target.view(-1, 1), 1.)
    return torch.mean(torch.abs(output - target_var) / torch.abs(target_var + 1e-8))


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def adjust_learning_rate(optimizer, epoch, lr=0.1, lr_decay_ratio=0.1, epoch_drop=(), writer=None):
    """Simple learning rate drop according to the provided parameters"""
    optim_factor = 0
    for i, ep in enumerate(epoch_drop):
        if epoch > ep:
            optim_factor = i + 1
    lr = lr * lr_decay_ratio ** optim_factor

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, epoch, is_best, best_score, directory, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_score,
        'beta_ema': model.beta_ema,
    }
    if model.beta_ema > 0:
        state['avg_params'] = model.avg_param
        state['steps_ema'] = model.steps_ema

    if not os.path.exists(directory):
        os.makedirs(directory)

    if is_best:
        filename = directory + '/' + 'model_best.pth.tar'
    else:
        filename = directory + '/' + filename

    print('Epoch: {}:   Saving checkpoint to: {}'.format(epoch, filename))

    torch.save(state, filename)


def record_process(writer, epoch, losses, quality, model):
    # log to TensorBoard
    writer.add_scalar('loss/val', losses.avg, epoch)
    writer.add_scalar('quality/val', quality.avg, epoch)
    layers = model.layers
    alive, total = 0, 0
    for k, layer in enumerate(layers):
        if hasattr(layer, 'qz_loga'):
            mode_z = layer.sample_z(1, sample=0).view(-1)
            alive += torch.sum(mode_z != 0).item()
            total += mode_z.nelement()

            writer.add_scalar('logitsq0/{}'.format(k), layer.qz_loga[0].item(), epoch)
            writer.add_scalar('logitsq3/{}'.format(k), layer.qz_loga[3].item(), epoch)
            writer.add_scalar('logitslast/{}'.format(k), layer.qz_loga[-1].item(), epoch)

            writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
            writer.add_scalar('mode_n/layer{}'.format(k), torch.sum(mode_z != 0).item(), epoch)
            writer.add_scalar('prune_ratio/layer{}'.format(k), torch.sum(mode_z == 0).item() / mode_z.nelement(), epoch)
    prune_ratio = 1 - alive / total
    writer.add_scalar('prune_ratio/global', prune_ratio, epoch)


def estimate_latency(model, example_inputs, repetitions=50):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    for _ in range(5):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


def measure_parameters(model):
    conv2d = 0
    linear = 0

    for name, module in model.named_modules():
        if isinstance(module, L0Conv2d) or isinstance(module, L0Conv3d) or isinstance(module, nn.Conv2d):
            if module.weight is not None:
                conv2d += module.weight.numel()
            else:
                conv2d += module.weights.numel()
            if module.bias is not None:
                conv2d += module.bias.numel()
        elif isinstance(module, nn.Linear) or isinstance(module, L0Dense):
            if module.weight is not None:
                linear += module.weight.numel()
            else:
                linear += module.weights.numel()
            if module.bias is not None:
                linear += module.bias.numel()

    return conv2d, linear


def measure_flops(model):
    conv_flops = 0
    linear_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, (L0Conv2d, L0Conv3d)):
            conv_flops += compute_conv_flops(module.input_shape, module.kernel_size, module.stride, module.padding,
                                             module.in_channels, module.out_channels)
        elif isinstance(module, (nn.Linear, L0Dense)):
            linear_flops += compute_linear_flops(module.in_features, module.out_features)
    return conv_flops, linear_flops


def compute_output_size(input_size, kernel_size, stride, padding):
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    return output_size


def compute_conv_flops(input_shape, kernel_size, stride, padding, in_channels, out_channels):
    n = (kernel_size[0] * kernel_size[1] * in_channels)
    if len(kernel_size) == 3:
        n *= kernel_size[2]
    flops_per_instance = n + (n - 1)

    num_instances_per_filter = ((input_shape[-3] - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
    num_instances_per_filter *= ((input_shape[-2] - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
    if len(kernel_size) == 3:
        num_instances_per_filter *= ((input_shape[-1] - kernel_size[2] + 2 * padding[2]) / stride[2]) + 1
    flops_per_filter = num_instances_per_filter * flops_per_instance
    flops = flops_per_filter * out_channels

    return flops


def compute_linear_flops(in_features, out_features):
    return in_features * out_features * 2


def cosine_annealing(step, total_steps, initial_budget, final_budget):
    cos_inner = (math.pi * (step % total_steps)) / total_steps
    return final_budget + (initial_budget - final_budget) / 2 * (math.cos(cos_inner) + 1)


def update_budget(args, epoch, model):
    budget, temp = 0, 0
    if args.strategy == 'anneal':
        budget = cosine_annealing(epoch, args.stop_epochs, args.initial_budget, args.final_budget)
    elif args.strategy == 'step':
        budget = args.initial_budget if epoch < args.stop_epochs else args.final_budget
    model.update_budget(budget)

    if args.strategy == 'anneal':
        temp = cosine_annealing(epoch, args.stop_epochs, args.initial_temp, args.final_temp)
    elif args.strategy == 'step':
        temp = args.initial_temp if epoch < args.stop_epochs else args.final_temp
    model.update_temperature(temp)

    args.reg = True if epoch >= args.stop_epochs else False
