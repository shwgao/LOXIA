import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d
from utils import get_flat_fts
import torch_pruning as tp
import torch.autograd.profiler as profiler
import os


class L0LeNet5(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(L0LeNet5, self).__init__()
        
        script_dir = os.path.dirname(__file__)
        with open(f'{script_dir}/settings.json') as f:
            settings = json.load(f)
        
        input_size = settings["input_size"]
        num_classes = settings["num_classes"]
        conv_dims = settings["conv_dims"]
        fc_dims = settings["fc_dims"]
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        droprate_init = settings["droprate_init"]
        budget = settings["budget"]
        beta_ema = settings["beta_ema"]

        self.beta_ema = beta_ema
        self.N = settings["N"]
        self.budget = budget
        self.device = device
        
        if inference:
            use_reg = using_reg

        convs = [L0Conv2d(3, conv_dims[0], 5, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2),
                #  nn.Conv2d(conv_dims[0], conv_dims[1], 5),
                 L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)

        fcs = [
               L0Dense(1250, 500, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget), nn.ReLU(),
               L0Dense(500, 128, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget), nn.ReLU(),
               L0Dense(128, num_classes, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
               ]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        with profiler.record_function("CONV"):
            o = self.convs(x)

        o = o.view(o.size(0), -1)

        with profiler.record_function("LINEAR"):
            o = self.fcs(o)
        return o

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
            'convs.0': {'in_mask': None, 'out_mask': self.convs[0].mask},
            'convs.3': {'in_mask': self.convs[0].mask,'out_mask': self.convs[3].mask},
            'fcs.0': {'in_mask': self.fcs[0].mask, 'out_mask': self.fcs[2].mask},
            'fcs.2': {'in_mask': self.fcs[2].mask, 'out_mask': self.fcs[4].mask},
            'fcs.4': {'in_mask': self.fcs[4].mask, 'out_mask': None},
        }
        self.fcs[0].set_couple_prune((1, 50, 5, 5), pre_mask=dependency_dict['convs.3']['out_mask'])
        dependency_dict['fcs.0']['in_mask'] = self.fcs[0].mask

        return dependency_dict
    
    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, (L0Conv2d, L0Dense)):
                layer.prepare_for_inference()
        dependency_dict = self.build_dependency_graph()
        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    module = tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    module = tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, L0Dense):
                if dependency_dict[name]['in_mask'] is not None:
                    module = tp.prune_linear_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    module = tp.prune_linear_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, nn.BatchNorm2d):
                if dependency_dict[name]['in_mask'] is not None:
                    module = tp.prune_batchnorm_in_channels(module, idxs=dependency_dict[name]['in_mask'])


if __name__ == '__main__':
    print(MLP())
