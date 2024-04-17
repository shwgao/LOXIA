import contextlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, BaseModel
import torch_pruning as tp
import os


class VGG11(BaseModel):
    def __init__(self, inference=False, using_reg=False):
        super(VGG11, self).__init__()

        script_dir = os.path.dirname(__file__)
        with open(f"{script_dir}/settings.json") as f:
            settings = json.load(f)

        num_classes = settings["num_classes"]

        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        droprate_init = settings["droprate_init"]
        budget = settings["initial_budget"]
        beta_ema = settings["beta_ema"]
        input_size = settings["input_size"]
        conv_dims = settings["conv_dims"]

        self.conv_dims = settings["conv_dims"]
        self.fc_dims = settings["fc_dims"]
        self.N = settings["N"]
        self.beta_ema = beta_ema
        self.budget = budget
        self.device = device
        self.local_rep = local_rep
        self.temperature = temperature

        use_reg = using_reg if inference else settings["use_reg"]

        convs = [  # block 1
            L0Conv2d(input_size[0], input_size[1], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            L0Conv2d(input_size[1], input_size[2], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            L0Conv2d(input_size[2], input_size[3], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            L0Conv2d(input_size[3], input_size[3], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            L0Conv2d(input_size[3], input_size[4], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            L0Conv2d(input_size[4], input_size[4], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            L0Conv2d(input_size[4], input_size[4], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            L0Conv2d(input_size[4], input_size[4], conv_dims[0], padding=1, droprate_init=droprate_init,
                     temperature=temperature, budget=budget,
                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.convs = nn.Sequential(*convs)

        self.convs = self.convs.to(device)

        flat_fts = 4 * 4 * 512  # get_flat_fts(input_size, self.convs, device=device)
        fcs = [
            L0Dense(flat_fts, 4096, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                    lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget),
            nn.ReLU(),
            nn.Dropout(),

            L0Dense(4096, 4096, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                    lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget),
            nn.ReLU(),
            nn.Dropout(),

            L0Dense(4096, num_classes, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                    lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget),
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
        o = self.convs(x)

        o = o.view(o.size(0), -1)

        o = self.fcs(o)
        return o
