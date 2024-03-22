import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d
import torch_pruning as tp
import os


class DMSNet(nn.Module):
    """ Define a CNN """

    def __init__(self, inference=False, using_reg=False):
        super(DMSNet, self).__init__()

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
        self.device = device
        self.conv1 = L0Conv2d(3, 8, 4, droprate_init=0.5, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.conv2 = L0Conv2d(8, 16, 4, droprate_init=0.5, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.conv3 = L0Conv2d(16, 32, 4, droprate_init=0.5, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.conv4 = L0Conv2d(32, 64, 4, droprate_init=0.5, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.conv5 = L0Conv2d(64, 128, 4, droprate_init=0.5, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.a_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.fc1 = L0Dense_with_budget(87584, 512, droprate_init=0.5, weight_decay=self.weight_decay, use_reg=use_reg,
        #                                lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)
        self.fc1 = L0Dense(4608, 512, droprate_init=0.5, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)

        self.fc2 = L0Dense(512, 256, droprate_init=0.5, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)
        self.fc3 = L0Dense(256, 2, droprate_init=0.5, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)
        # self.drop = nn.Dropout(p=0.5)

        self.output_dim = 1

        self.layers = []
        self.layers.extend(
            m
            for m in self.modules()
            if isinstance(m, (L0Dense, L0Conv2d))
        )
        if beta_ema > 0.:
            print(f'Using temporal averaging with beta: {beta_ema}')
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.bn5(self.pool(F.relu(self.conv5(x))))
        # x = self.a_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.drop(F.relu(self.fc1(x))).to(self.device)
        # x = self.drop(F.relu(self.fc2(x))).to(self.device)
        x = F.relu(self.fc1(x))  # don't use drop for the last two layers
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def constrain_parameters(self):
        for layer in self.layers:
            layer.constrain_parameters()

    def update_budget(self, budget):
        self.budget = budget
        for layer in self.layers:
            layer.update_budget(budget)

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
        dependency_dict = {
            'conv1': {'in_mask': None, 'out_mask': self.conv1.mask},
            'bn1': {'in_mask': self.conv1.mask, 'out_mask': None},
            'conv2': {'in_mask': self.conv1.mask, 'out_mask': self.conv2.mask},
            'bn2': {'in_mask': self.conv2.mask, 'out_mask': None},
            'conv3': {'in_mask': self.conv2.mask, 'out_mask': self.conv3.mask},
            'bn3': {'in_mask': self.conv3.mask, 'out_mask': None},
            'conv4': {'in_mask': self.conv3.mask, 'out_mask': self.conv4.mask},
            'bn4': {'in_mask': self.conv4.mask, 'out_mask': None},
            'conv5': {'in_mask': self.conv4.mask, 'out_mask': self.conv5.mask},
            'bn5': {'in_mask': self.conv5.mask, 'out_mask': None},
            'fc1': {'in_mask': self.fc1.mask, 'out_mask': self.fc2.mask},
            'fc2': {'in_mask': self.fc2.mask, 'out_mask': self.fc3.mask},
            'fc3': {'in_mask': self.fc3.mask, 'out_mask': None},
            'couple': {'in_mask': self.conv5.mask, 'out_mask': self.fc1.mask},
        }
        # process the coupling between conv5 and fc1
        self.fc1.set_couple_prune(
            (1, 128, 3, 12), pre_mask=dependency_dict['conv5']['out_mask'])
        dependency_dict['fc1']['in_mask'] = self.fc1.mask

        return dependency_dict

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, (L0Conv2d, L0Dense)):
                layer.prepare_for_inference()
                
        dependency_dict = self.build_dependency_graph()
        
        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
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
