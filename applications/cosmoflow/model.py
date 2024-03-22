import json
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from copy import deepcopy
from torch.nn import init
from base_layers import L0Dense, L0Conv3d
import torch_pruning as tp
import os


class Conv3DActMP(nn.Module):
    def __init__(
            self,
            conv_kernel: int,
            conv_channel_in: int,
            conv_channel_out: int,
            use_reg,
            local_rep,
            droprate_init,
            device,
            weight_decay,
            temperature,
            budget,
            lambas=1,
            isbn: bool = False,
    ):
        super().__init__()

        self.conv = L0Conv3d(conv_channel_in, conv_channel_out, kernel_size=conv_kernel, stride=1, padding=1, bias=True,
                             use_reg=use_reg, droprate_init=droprate_init, temperature=temperature, local_rep=local_rep,
                             weight_decay=weight_decay, lamba=lambas, device=device, budget=budget)
        self.isbn = isbn
        self.bn = nn.BatchNorm3d(conv_channel_out)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weights)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.isbn:
            return self.mp(self.act(self.bn(self.conv(x))))
        else:
            return self.mp(self.act(self.conv(x)))


class CosmoFlow(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super().__init__()
        script_dir = os.path.dirname(__file__)
        with open(f'{script_dir}/settings.json') as f:
            settings = json.load(f)

        n_conv_layers = settings["n_conv_layers"]
        n_conv_filters = settings["n_conv_filters"]
        conv_kernel = settings["conv_kernel"]
        dropout_rate = settings["dropout_rate"]
        droprate_init = settings["droprate_init"]
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        budget = settings["budget"]
        self.beta_ema = settings["beta_ema"]
        self.N = settings["N"]

        self.budget = budget
        self.device = device
        self.temperature = temperature
        self.local_rep = local_rep

        self.conv_seq = nn.ModuleList()
        input_channel_size = 4

        if inference:
            use_reg = using_reg

        for i in range(n_conv_layers):
            output_channel_size = n_conv_filters * (1 << i)
            self.conv_seq.append(Conv3DActMP(conv_kernel, input_channel_size, output_channel_size, use_reg=use_reg,
                                             device=device, droprate_init=droprate_init, lambas=lambas,
                                             weight_decay=weight_decay, local_rep=local_rep, temperature=temperature,
                                             budget=budget))
            input_channel_size = output_channel_size

        flatten_inputs = 128 // (2 ** n_conv_layers)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        self.dense1 = L0Dense(flatten_inputs, 128, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)
        self.dense2 = L0Dense(128, 64, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)
        self.output = L0Dense(64, 4, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)

        if self.beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(self.beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dr1 = nn.Dropout(p=self.dropout_rate)
            self.dr2 = nn.Dropout(p=self.dropout_rate)

        for layer in [self.dense1, self.dense2, self.output]:
            if hasattr(layer, 'weights'):
                torch.nn.init.xavier_uniform_(layer.weights)
                torch.nn.init.zeros_(layer.bias)

        self.layers = []
        for layer in self.conv_seq:
            self.layers.append(layer.conv)
        self.layers += [self.dense1, self.dense2, self.output]

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                init.xavier_uniform_(layer.weights)

    def constrain_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.constrain_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_seq):
            x = conv_layer(x)

        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        x = nnf.leaky_relu(self.dense1(x.flatten(1)), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr1(x)

        x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr2(x)

        x = nnf.sigmoid(self.output(x)) * 1.2
        return x

    def update_budget(self, budget):
        for layer in self.layers:
            layer.update_budget(budget)

    def update_temperature(self, temperature):
        for layer in self.layers:
            layer.update_temperature(temperature)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def build_dependency_graph(self):
        dependency_dict = {'conv_seq.0.conv': {'in_mask': None, 'out_mask': self.conv_seq[0].conv.mask},
                           'conv_seq.0.bn': {'in_mask': self.conv_seq[0].conv.mask, 'out_mask': None},
                           'conv_seq.1.conv': {'in_mask': self.conv_seq[0].conv.mask,
                                               'out_mask': self.conv_seq[1].conv.mask},
                           'conv_seq.1.bn': {'in_mask': self.conv_seq[1].conv.mask, 'out_mask': None},
                           'conv_seq.2.conv': {'in_mask': self.conv_seq[1].conv.mask,
                                               'out_mask': self.conv_seq[2].conv.mask},
                           'conv_seq.2.bn': {'in_mask': self.conv_seq[2].conv.mask, 'out_mask': None},
                           'conv_seq.3.conv': {'in_mask': self.conv_seq[2].conv.mask,
                                               'out_mask': self.conv_seq[3].conv.mask},
                           'conv_seq.3.bn': {'in_mask': self.conv_seq[3].conv.mask, 'out_mask': None},
                           'conv_seq.4.conv': {'in_mask': self.conv_seq[3].conv.mask,
                                               'out_mask': self.conv_seq[4].conv.mask},
                           'conv_seq.4.bn': {'in_mask': self.conv_seq[4].conv.mask, 'out_mask': None},
                           'dense1': {'in_mask': self.dense1.mask, 'out_mask': self.dense2.mask},
                           'dense2': {'in_mask': self.dense2.mask, 'out_mask': self.output.mask},
                           'output': {'in_mask': self.output.mask, 'out_mask': None}}
        # dependency_dict['dense1'] = {'in_mask': self.dense1.mask, 'out_mask': self.dense2.mask}

        # process the coupling between conv5 and fc1
        self.dense1.set_couple_prune1((1, 128, 4, 4, 4), pre_mask=dependency_dict['conv_seq.4.conv']['out_mask'])
        dependency_dict['dense1']['in_mask'] = self.dense1.mask

        return dependency_dict

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Conv3d) or isinstance(layer, L0Dense):
                layer.prepare_for_inference()
        dependency_dict = self.build_dependency_graph()
        for name, module in self.named_modules():
            if isinstance(module, L0Conv3d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, L0Dense):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_linear_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_linear_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, nn.BatchNorm2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_batchnorm_in_channels(module, idxs=dependency_dict[name]['in_mask'])
