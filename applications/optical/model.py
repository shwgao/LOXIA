import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from base_layers import L0Dense, L0Conv2d, BaseModel
import os


class Autoencoder(BaseModel):
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
        budget = settings["initial_budget"]
        beta_ema = settings["beta_ema"]

        self.N = settings["N"]
        self.beta_ema = beta_ema
        self.budget = budget
        self.device = device
        self.local_rep = local_rep
        self.temperature = temperature
        self.latent_dim = settings["latent_dim"]
        self.input_shape = (200, 200, 1)
        
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
        self.output_conv = L0Conv2d(64, 1, kernel_size=1, droprate_init=droprate_init, temperature=temperature, budget=budget,
                            weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=False)

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
