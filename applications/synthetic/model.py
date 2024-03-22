import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Dense
import torch_pruning as tp
import os


class MLP(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(MLP, self).__init__()
        # load settings from json file
        script_dir = os.path.dirname(__file__)
        with open(f'{script_dir}/settings.json') as f:
            settings = json.load(f)

        num_classes = settings["num_classes"]
        layer_dims = settings["layer_dims"]
        input_dim = settings["input_dim"]
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]

        self.beta_ema = settings["beta_ema"]
        self.N = settings["N"]
        self.budget = settings["budget"]
        self.device = device
        if inference:
            use_reg = using_reg

        layers = []
        for i, dimh in enumerate(layer_dims):
            inp_dim = input_dim if i == 0 else layer_dims[i - 1]
            droprate_init = 0.2 if i == 0 else 0.5
            layers += [L0Dense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=weight_decay, use_reg=use_reg,
                               lamba=lambas, local_rep=local_rep, temperature=temperature, device=device,
                               budget=self.budget)]
            layers += [nn.ReLU()]

        layers.append(
            L0Dense(layer_dims[-1], num_classes, droprate_init=0.5, weight_decay=weight_decay, use_reg=use_reg,
                    lamba=lambas, local_rep=local_rep, temperature=temperature, device=device,
                    budget=self.budget))
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if self.beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(self.beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        return self.output(x)

    def update_budget(self, budget):
        self.budget = budget
        for layer in self.layers:
            layer.update_budget(budget)

    def constrain_parameters(self):
        for layer in self.layers:
            layer.constrain_parameters()

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
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
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        return deepcopy([p.data for p in self.parameters()])

    def build_dependency_graph(self):
        dependency_dict = {}
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Dense):
                dependency_dict[name] = {'in_mask': module.mask, 'out_mask': None}
                if pre_module is not None:
                    dependency_dict[pre_module]['out_mask'] = module.mask
                pre_module = name

        dependency_dict['output.0']['in_mask'] = None

        return dependency_dict

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Dense):
                layer.prepare_for_inference()
        dependency_dict = self.build_dependency_graph()
        for name, module in self.named_modules():
            if isinstance(module, L0Dense):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_linear_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_linear_out_channels(module, idxs=dependency_dict[name]['out_mask'])


if __name__ == '__main__':
    print(MLP())
