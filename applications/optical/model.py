import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d
import torch_pruning as tp
import os


class Autoencoder(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(Autoencoder, self).__init__()
        
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
        self.latent_dim = settings["latent_dim"]
        self.input_shape = (200, 200, 3)
        
        use_reg = using_reg if inference else settings["use_reg"]

        # Encoder
        self.conv1 = L0Conv2d(1, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = L0Conv2d(64, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = L0Conv2d(32, 16, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)

        self.bn3 = nn.BatchNorm2d(16)

        # Calculating shape after convolutions
        h, w = self.input_shape[:2]
        h, w = h // 4, w // 4  # Adjusted for 2 MaxPool2D layers
        self.flattened_size = h * w * 16

        # Dense layers for bottleneck
        self.dense1 = L0Dense(self.flattened_size, self.latent_dim, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)
        self.dense2 = L0Dense(self.latent_dim, self.flattened_size, droprate_init=droprate_init, weight_decay=weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg, budget=budget)

        # self.dense1 = nn.Linear(self.flattened_size, self.latent_dim)
        # self.dense2 = nn.Linear(self.latent_dim, self.flattened_size)

        # Decoder
        self.conv4 = L0Conv2d(16, 16, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn4 = nn.BatchNorm2d(16)
        self.deconv1 = L0Conv2d(16, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        # self.deconv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = L0Conv2d(32, 32, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn6 = nn.BatchNorm2d(32)
        self.deconv2 = L0Conv2d(32, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)

        # self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv6 = L0Conv2d(64, 64, kernel_size=3, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        self.bn8 = nn.BatchNorm2d(64)
        # self.output_conv = L0Conv2d(64, 1, kernel_size=1, padding=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
        #                     weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=False)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1, padding=1)

        # initialize weights using kaiming normal
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        self.layers = []
        self.layers.extend(
            m for m in self.modules() if isinstance(m, (L0Conv2d, L0Dense))
        )
        if beta_ema > 0.:
            print(f'Using temporal averaging with beta: {beta_ema}')
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = x.view(-1, 16, self.input_shape[0] // 4, self.input_shape[1] // 4)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn5(self.deconv1(x)))
        
        x = F.relu(self.bn6(self.conv5(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.bn7(self.deconv2(x)))
        
        x = F.relu(self.bn8(self.conv6(x)))
        x = self.output_conv(x)
        return x

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
            if isinstance(layer, Conv2d):
                regularization += - (1000. / self.N) * layer.regularization()
            else:
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
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d):
                dependency_dict[name] = {'in_mask': None, 'out_mask': module.mask}
                if pre_module is not None:
                    dependency_dict[name]['in_mask'] = dependency_dict[pre_module]['out_mask']
                pre_module = name
            elif isinstance(module, nn.BatchNorm2d):
                dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None}
               
        dependency_dict['dense1'] = {'in_mask': self.dense1.mask, 'out_mask': self.dense2.mask}
        dependency_dict['dense2'] = {'in_mask': self.dense2.mask, 'out_mask': None}

        dependency_dict['conv4'] = {'in_mask': None, 'out_mask': self.conv4.mask}

        dependency_dict['output_conv'] = {'in_mask': self.conv6.mask, 'out_mask': None}

        self.dense1.set_couple_prune((1, 16, 50, 50), pre_mask=dependency_dict['conv3']['out_mask'])
        dependency_dict['dense1']['in_mask'] = self.dense1.mask
        
        return dependency_dict

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, (L0Conv2d, L0Dense)):
                layer.prepare_for_inference()
                
        dependency_dict = self.build_dependency_graph()

        for name, module in self.named_modules():
            if name not in dependency_dict:
                continue
            if isinstance(module, L0Conv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, nn.ConvTranspose2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
            elif isinstance(module, nn.BatchNorm2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_batchnorm_out_channels(module, idxs=dependency_dict[name]['in_mask'])
            if isinstance(module, L0Dense):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_linear_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_linear_out_channels(module, idxs=dependency_dict[name]['out_mask'])
            elif isinstance(module, nn.Conv2d):
                if dependency_dict[name]['in_mask'] is not None:
                    tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                if dependency_dict[name]['out_mask'] is not None:
                    tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
                    