import contextlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, MAPConv2d, MAPDense
import torch_pruning as tp
import os


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, local_rep=False,
                 temperature=2./3., budget=0.49, use_reg=True):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = L0Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, budget=budget,
                              droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3), local_rep=local_rep,
                              lamba=lamba, temperature=temperature, use_reg=use_reg)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None
        # self.convShortcut = (not self.equalInOut) and \
        #                     nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01,
                 local_rep=False, temperature=2./3., budget=0.49, use_reg=True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init, budget=budget,
                                      weight_decay=weight_decay, lamba=lamba, local_rep=local_rep, use_reg=use_reg,
                                      temperature=temperature)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init, budget=1e-6, use_reg=True,
                    weight_decay=0.0005, lamba=0.01, local_rep=False, temperature=2./3.):
        layers = [
            block(
                i == 0 and in_planes or out_planes,
                out_planes,
                i == 0 and stride or 1,
                droprate_init,
                weight_decay,
                lamba,
                local_rep=local_rep,
                temperature=temperature,
                budget=budget,
                use_reg=use_reg,
            )
            for i in range(nb_layers)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(WideResNet, self).__init__()

        script_dir = os.path.dirname(__file__)
        with open(f"{script_dir}/settings.json") as f:
            settings = json.load(f)

        depth = settings["depth"]
        num_classes = settings["num_classes"]
        widen_factor = settings["widen_factor"]

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 128*widen_factor, 256*widen_factor]
        assert((depth - 4) % 6 == 0)

        self.n = (depth - 4) // 6

        block = BasicBlock

        weight_decay = settings["weight_decay"]
        lamba = settings["lambas"]
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
        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=weight_decay, device=device)
        # self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, weight_decay,
                                   lamba, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, weight_decay,
                                   lamba, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, weight_decay,
                                   lamba, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)

        self.block4 = NetworkBlock(self.n, nChannels[3], nChannels[4], block, 2, droprate_init, weight_decay,
                                   lamba, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)

        self.block5 = NetworkBlock(self.n, nChannels[4], nChannels[5], block, 2, droprate_init, weight_decay,
                                   lamba, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[5])
        # self.fcout = MAPDense(nChannels[5], num_classes, weight_decay=self.weight_decay)
        self.fcout = nn.Linear(nChannels[5], num_classes)

        self.layers, self.bn_params = [], []
        for m in self.modules():
            if isinstance(m, (MAPDense, MAPConv2d, L0Conv2d)):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print(f'Using temporal averaging with beta: {beta_ema}')
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

        print(f'Using weight decay: {weight_decay}')

    def update_budget(self, budget):
        self.budget = budget
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.update_budget(budget)

    def update_temperature(self, temperature):
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.update_temperature(temperature)

    def forward(self, x):
        out = self._extracted_from_forward_3(x)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        # out = F.avg_pool2d(out, 2) # for cifar10
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    # TODO Rename this here and in `forward`
    def _extracted_from_forward_3(self, x):
        result = self.conv1(x)
        result = self.block1(result)
        result = self.block2(result)
        result = self.block3(result)
        result = self.block4(result)
        result = self.block5(result)
        return result

    def constrain_parameters(self):
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.constrain_parameters()

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            with contextlib.suppress(Exception):
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
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                dependency_dict[name] = {'in_mask': None, 'out_mask': module.mask}
                pre_module = name
            if isinstance(module, MAPConv2d):
                if pre_module is not None:
                    dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None}
                pre_module = None
            if isinstance(module, nn.BatchNorm2d) and pre_module is not None:
                dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None}

        return dependency_dict
    
    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.prepare_for_inference()

        dependency_dict = self.build_dependency_graph()
        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])

            if isinstance(module, MAPConv2d) and name in dependency_dict.keys():
                tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])

            if isinstance(module, nn.BatchNorm2d) and name in dependency_dict.keys() and len(dependency_dict[name]['in_mask']) < module.num_features:
                tp.prune_batchnorm_out_channels(module, idxs=dependency_dict[name]['in_mask'])

