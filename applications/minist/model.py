import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Dense, BaseModel
import os


class MLP(BaseModel):
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
