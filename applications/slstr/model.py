import contextlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, MAPConv2d, MAPDense
import torch_pruning as tp
import os


class UNet(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(UNet, self).__init__()
        
        script_dir = os.path.dirname(__file__)
        with open(f"{script_dir}/settings.json") as f:
            settings = json.load(f)
        
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        droprate_init = settings["droprate_init"]
        budget = settings["budget"]
        beta_ema = settings["beta_ema"]

        self.N = settings["N"]
        self.beta_ema = beta_ema
        self.budget = budget
        self.device = device
        self.local_rep = local_rep
        self.temperature = temperature
        
        use_reg = using_reg if inference else settings["use_reg"]

        # Encoder
        self.enc_conv1 = L0Conv2d(9, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.enc_conv2 = L0Conv2d(32, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.enc_conv3 = L0Conv2d(32, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.enc_conv4 = L0Conv2d(64, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.enc_conv5 = L0Conv2d(64, 128, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.enc_conv6 = L0Conv2d(128, 128, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)

        # Bottleneck
        self.bottleneck_conv1 = L0Conv2d(128, 256, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bottleneck_conv2 = L0Conv2d(256, 256, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)

        # Decoder
        self.dec_conv1 = L0Conv2d(256 + 128, 128, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.dec_conv2 = L0Conv2d(128, 128, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.dec_conv3 = L0Conv2d(128 + 64, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.dec_conv4 = L0Conv2d(64, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.dec_conv5 = L0Conv2d(64 + 32, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.dec_conv6 = L0Conv2d(32, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.final_conv = MAPConv2d(32, 1, kernel_size=1)

        self.layers = []
        self.layers.extend(m for m in self.modules() if isinstance(m, L0Conv2d))
        if beta_ema > 0.:
            print(f'Using temporal averaging with beta: {beta_ema}')
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv1(x))
        x1 = F.relu(self.enc_conv2(x1))
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)

        x2 = F.relu(self.enc_conv3(x2))
        x2 = F.relu(self.enc_conv4(x2))
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)

        x3 = F.relu(self.enc_conv5(x3))
        x3 = F.relu(self.enc_conv6(x3))
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)

        # Bottleneck
        x4 = F.relu(self.bottleneck_conv1(x4))
        x4 = F.relu(self.bottleneck_conv2(x4))

        # Decoder
        x4 = F.interpolate(x4, scale_factor=2, mode='nearest')
        x5 = torch.cat([x4, x3], dim=1)
        x5 = F.relu(self.dec_conv1(x5))
        x5 = F.relu(self.dec_conv2(x5))

        x5 = F.interpolate(x5, scale_factor=2, mode='nearest')
        x6 = torch.cat([x5, x2], dim=1)
        x6 = F.relu(self.dec_conv3(x6))
        x6 = F.relu(self.dec_conv4(x6))

        x6 = F.interpolate(x6, scale_factor=2, mode='nearest')
        x7 = torch.cat([x6, x1], dim=1)
        x7 = F.relu(self.dec_conv5(x7))
        x7 = F.relu(self.dec_conv6(x7))

        x7 = self.final_conv(x7)
        return torch.sigmoid(x7)

    def constrain_parameters(self):
        for layer in self.layers:
            layer.constrain_parameters()

    def update_budget(self, budget):
        self.budget = budget
        for layer in self.layers:
            layer.update_budget(budget)

    def update_temperature(self, temperature):
        self.temperature = temperature
        for layer in self.layers:
            layer.update_temperature(temperature)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
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
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        return deepcopy([p.data for p in self.parameters()])

    def build_dependency_graph(self):
        dependency_dict = {}
        dependency_dict_skip = {}
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                dependency_dict[name] = {'in_mask': None, 'out_mask': module.mask}
                if pre_module is not None:
                    dependency_dict[name]['in_mask'] = dependency_dict[pre_module]['out_mask']
                pre_module = name

        dependency_dict['final_conv'] = {'in_mask': self.dec_conv6.mask, 'out_mask': None}

        # dependency of skip layers
        offset = 256
        dependency_dict['dec_conv1']['in_mask'] = dependency_dict['dec_conv1']['in_mask']+[x + offset for x in dependency_dict['enc_conv6']['out_mask']]
        offset = 128
        dependency_dict['dec_conv3']['in_mask'] = dependency_dict['dec_conv3']['in_mask']+[x + offset for x in dependency_dict['enc_conv4']['out_mask']]
        offset = 96
        dependency_dict['dec_conv5']['in_mask'] = dependency_dict['dec_conv5']['in_mask']+[x + offset for x in dependency_dict['enc_conv2']['out_mask']]

        return dependency_dict, dependency_dict_skip

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.prepare_for_inference()
        dependency_dict, dependency_dict_skip = self.build_dependency_graph()

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, MAPConv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
            elif isinstance(module, nn.BatchNorm2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_batchnorm_in_channels(module, idxs=dependency_dict[name]['in_mask'])

