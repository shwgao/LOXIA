import json
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from copy import deepcopy
from torch.nn import init
from base_layers import L0Dense, L0Conv3d, BaseModel, L0Conv2d
import os


class Conv3DActMP(nn.Module):
    def __init__(
            self,
            conv_kernel: int,
            conv_channel_in: int,
            conv_channel_out: int,
            use_reg,
            local_rep,
            droprate_init,
            device,
            weight_decay,
            temperature,
            budget,
            lambas=1,
            isbn: bool = False,
    ):
        super().__init__()

        self.conv = L0Conv3d(conv_channel_in, conv_channel_out, kernel_size=conv_kernel, stride=1, padding=1, bias=True,
                             use_reg=use_reg, droprate_init=droprate_init, temperature=temperature, local_rep=local_rep,
                             weight_decay=weight_decay, lamba=lambas, device=device, budget=budget)
        self.isbn = isbn
        self.bn = nn.BatchNorm3d(conv_channel_out)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weights)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.isbn:
            return self.mp(self.act(self.bn(self.conv(x))))
        else:
            return self.mp(self.act(self.conv(x)))


class CosmoFlow(BaseModel):
    def __init__(self, inference=False, using_reg=False):
        super().__init__()
        script_dir = os.path.dirname(__file__)
        with open(f'{script_dir}/settings.json') as f:
            settings = json.load(f)

        n_conv_layers = settings["n_conv_layers"]
        n_conv_filters = settings["n_conv_filters"]
        conv_kernel = settings["conv_kernel"]
        dropout_rate = settings["dropout_rate"]
        droprate_init = settings["droprate_init"]
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        budget = settings["initial_budget"]
        self.beta_ema = settings["beta_ema"]
        self.N = settings["N"]

        self.flat_shape = settings["flat_shape"]
        self.budget = budget
        self.device = device
        self.temperature = temperature
        self.local_rep = local_rep

        self.conv_seq = nn.ModuleList()
        input_channel_size = 4

        if inference:
            use_reg = using_reg

        for i in range(n_conv_layers):
            output_channel_size = n_conv_filters * (1 << i)
            self.conv_seq.append(Conv3DActMP(conv_kernel, input_channel_size, output_channel_size, use_reg=use_reg,
                                             device=device, droprate_init=droprate_init, lambas=lambas,
                                             weight_decay=weight_decay, local_rep=local_rep, temperature=temperature,
                                             budget=budget))
            input_channel_size = output_channel_size

        flatten_inputs = 128 // (2 ** n_conv_layers)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        self.dense1 = L0Dense(flatten_inputs, 128, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)
        self.dense2 = L0Dense(128, 64, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)
        self.output = L0Dense(64, 4, bias=True, use_reg=use_reg, device=device, lamba=lambas,
                              weight_decay=weight_decay, temperature=temperature, droprate_init=droprate_init,
                              local_rep=local_rep, budget=budget)

        if self.beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(self.beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dr1 = nn.Dropout(p=self.dropout_rate)
            self.dr2 = nn.Dropout(p=self.dropout_rate)

        for layer in [self.dense1, self.dense2, self.output]:
            if hasattr(layer, 'weights'):
                torch.nn.init.xavier_uniform_(layer.weights)
                torch.nn.init.zeros_(layer.bias)

        self.layers = []
        for layer in self.conv_seq:
            self.layers.append(layer.conv)
        self.layers += [self.dense1, self.dense2, self.output]

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                init.xavier_uniform_(layer.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_seq):
            x = conv_layer(x)

        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        x = nnf.leaky_relu(self.dense1(x.flatten(1)), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr1(x)

        x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr2(x)

        x = nnf.sigmoid(self.output(x)) * 1.2
        return x
