import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, BaseModel
import torch.autograd.profiler as profiler
import os


class L0LeNet5(BaseModel):
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

        self.flat_shape = settings["flat_shape"]
        self.beta_ema = beta_ema
        self.N = settings["N"]
        self.budget = budget
        self.device = device
        self.temperature = temperature
        
        if inference:
            use_reg = using_reg

        convs = [L0Conv2d(3, conv_dims[0], 5, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2),

                 L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=droprate_init, temperature=temperature, budget=budget,
                          weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        flat_fts = 25 * conv_dims[1]
        fcs = [
               L0Dense(flat_fts, fc_dims, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget), nn.ReLU(),
               L0Dense(fc_dims, 128, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget), nn.ReLU(),
               L0Dense(128, num_classes, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
               ]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, (L0Dense, L0Conv2d)):
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

    def build_dependency_graph(self):
        dependency_dict = {}
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Dense):
                dependency_dict[name] = {'in_mask': module.mask, 'out_mask': None, 'type': 'fc'}
                if pre_module is not None:
                    if dependency_dict[pre_module]['type'] == 'fc':
                        dependency_dict[pre_module]['out_mask'] = module.mask
                    else:  # set couple prune between conv and fc
                        module.set_couple_prune1(self.flat_shape, pre_mask=dependency_dict[pre_module]['out_mask'])
                        dependency_dict[name]['in_mask'] = module.mask
                pre_module = name
            elif isinstance(module, L0Conv2d):
                dependency_dict[name] = {'in_mask': None, 'out_mask': module.mask}
                if pre_module is not None:
                    dependency_dict[name]['in_mask'] = dependency_dict[pre_module]['out_mask']
                pre_module = name
            elif isinstance(module, nn.BatchNorm2d):
                dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None}
            else:
                continue

        return dependency_dict
