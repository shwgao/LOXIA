import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, BaseModel, L0Conv3d
import os


class DMSNet(BaseModel):
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
        budget = settings["initial_budget"]
        beta_ema = settings["beta_ema"]

        self.N = settings["N"]
        self.beta_ema = beta_ema
        self.budget = budget
        self.device = device
        self.local_rep = local_rep
        self.temperature = temperature

        use_reg = using_reg if inference else settings["use_reg"]
        self.device = device
        self.conv1 = L0Conv2d(3, 8, 4, droprate_init=droprate_init, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = L0Conv2d(8, 16, 4, droprate_init=droprate_init, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = L0Conv2d(16, 32, 4, droprate_init=droprate_init, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = L0Conv2d(32, 64, 4, droprate_init=droprate_init, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = L0Conv2d(64, 128, 4, droprate_init=droprate_init, temperature=temperature, budget=budget,
                              weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = L0Dense(4608, 512, droprate_init=droprate_init, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)
        self.fc2 = L0Dense(512, 256, droprate_init=droprate_init, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)
        self.fc3 = L0Dense(256, 2, droprate_init=droprate_init, weight_decay=weight_decay, use_reg=use_reg,
                           lamba=lambas, local_rep=local_rep, temperature=temperature, device=device)

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
        x = x.view(x.size(0), -1)
        # x = self.drop(F.relu(self.fc1(x))).to(self.device)
        # x = self.drop(F.relu(self.fc2(x))).to(self.device)
        x = F.relu(self.fc1(x))  # don't use drop for the last two layers
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

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
            elif isinstance(module, (L0Conv2d, L0Conv3d)):
                dependency_dict[name] = {'in_mask': None, 'out_mask': module.mask, 'type': 'conv'}
                if pre_module is not None and dependency_dict[pre_module]['type'] == 'conv':
                    dependency_dict[name]['in_mask'] = dependency_dict[pre_module]['out_mask']
                pre_module = name
            elif isinstance(module, nn.BatchNorm2d):
                dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None, 'type': 'bn'}
            else:
                continue

        return dependency_dict
