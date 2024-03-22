import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Conv2d
import torch_pruning as tp
import os


class EMDenoiseNet(nn.Module):
    def __init__(self, inference=False, using_reg=False):
        super(EMDenoiseNet, self).__init__()

        script_dir = os.path.dirname(__file__)
        with open(f"{script_dir}/settings.json") as f:
            settings = json.load(f)

        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
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

        if inference:
            use_reg = using_reg

        # encoder
        self.block1 = nn.ModuleList()
        self.block1.append(
            L0Conv2d(
                1,
                8,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.block1.append(
            L0Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.block1.append(nn.MaxPool2d(2))

        self.block2 = nn.ModuleList()
        self.block2.append(
            L0Conv2d(
                8,
                16,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        self.block2.append(
            L0Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        self.block2.append(nn.MaxPool2d(2))

        self.block3 = nn.ModuleList()
        self.block3.append(
            L0Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        self.block3.append(
            L0Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        self.block3.append(nn.MaxPool2d(2))

        self.block4 = nn.ModuleList()
        self.block4.append(
            L0Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))
        self.block4.append(
            L0Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))

        # decoder
        self.block5 = nn.ModuleList()
        self.block5.append(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.block5.append(
            L0Conv2d(
                64 + 32,
                32,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))
        self.block5.append(
            L0Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))

        self.block6 = nn.ModuleList()
        self.block6.append(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.block6.append(
            L0Conv2d(
                16 + 32,
                16,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))
        self.block6.append(
            L0Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))

        self.block7 = nn.ModuleList()
        self.block7.append(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.block7.append(
            L0Conv2d(
                16 + 8,
                8,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))
        self.block7.append(
            L0Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=use_reg,
            )
        )
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))

        self.last_layer = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.last_layer = L0Conv2d(
                8,
                1,
                kernel_size=3,
                padding=1,
                droprate_init=droprate_init,
                temperature=temperature,
                budget=budget,
                weight_decay=weight_decay,
                lamba=lambas,
                local_rep=local_rep,
                device=device,
                use_reg=False,
            )

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.0:
            print("Using temporal averaging with beta: {}".format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.to(device) for a in self.avg_param]
            self.steps_ema = 0.0

    def forward(self, x):
        skip_layers = []
        for i in range(len(self.block1) - 1):
            x = self.block1[i](x)
        skip_layers.append(x)
        x = self.block1[-1](x)
        for i in range(len(self.block2) - 1):
            x = self.block2[i](x)
        skip_layers.append(x)
        x = self.block2[-1](x)

        for i in range(len(self.block3) - 1):
            x = self.block3[i](x)
        skip_layers.append(x)
        x = self.block3[-1](x)

        for i in range(len(self.block4)):
            x = self.block4[i](x)

        x = self.block5[0](x)
        x = torch.cat((x, skip_layers[-1]), dim=1)
        for i in range(len(self.block5) - 1):
            x = self.block5[i + 1](x)

        x = self.block6[0](x)
        x = torch.cat((x, skip_layers[-2]), dim=1)
        for i in range(len(self.block6) - 1):
            x = self.block6[i + 1](x)

        x = self.block7[0](x)
        x = torch.cat((x, skip_layers[-3]), dim=1)
        for i in range(len(self.block7) - 1):
            x = self.block7[i + 1](x)

        x = self.last_layer(x)
        return x

    def constrain_parameters(self):
        for layer in self.layers:
            layer.constrain_parameters()

    def update_budget(self, budget):
        self.budget = budget
        for layer in self.layers:
            layer.update_budget(budget)

    def regularization(self):
        regularization = 0.0
        for layer in self.layers:
            regularization += -(1.0 / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0.0, 0.0
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
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def build_dependency_graph(self):
        dependency_dict = {}
        dependency_dict_skip = {}
        pre_module = None

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d) and module.use_reg:
                dependency_dict[name] = {
                    "in_mask": None, "out_mask": module.mask}
                if pre_module is not None:
                    dependency_dict[name]["in_mask"] = dependency_dict[pre_module][
                        "out_mask"
                    ]
                pre_module = name
            elif isinstance(module, nn.BatchNorm2d):
                dependency_dict[name] = {
                    "in_mask": dependency_dict[pre_module]["out_mask"],
                    "out_mask": None,
                }

        dependency_dict["last_layer"] = {
            "in_mask": self.block7[4].mask,
            "out_mask": None,
        }

        # dependency of skip layers
        offset = self.block3[3].m.shape[1]
        dependency_dict["block5.1"]["in_mask"] = dependency_dict["block5.1"]["in_mask"] + \
            [x + offset for x in dependency_dict["block3.3"]["out_mask"]]

        offset = self.block2[3].m.shape[1]
        dependency_dict["block6.1"]["in_mask"] = dependency_dict["block6.1"]["in_mask"] + \
            [x + offset for x in dependency_dict["block2.3"]["out_mask"]]

        offset = self.block1[3].m.shape[1]
        dependency_dict["block7.1"]["in_mask"] = dependency_dict["block7.1"]["in_mask"] + \
            [x + offset for x in dependency_dict["block1.3"]["out_mask"]]

        return dependency_dict, dependency_dict_skip

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Conv2d):
                layer.prepare_for_inference()
        dependency_dict, dependency_dict_skip = self.build_dependency_graph()

        for name, module in self.named_modules():
            if isinstance(module, L0Conv2d) and name in dependency_dict.keys():
                if dependency_dict[name]["in_mask"] is not None:
                    tp.prune_conv_in_channels(
                        module, idxs=dependency_dict[name]["in_mask"]
                    )
                if dependency_dict[name]["out_mask"] is not None:
                    tp.prune_conv_out_channels(
                        module, idxs=dependency_dict[name]["out_mask"]
                    )
            elif isinstance(module, nn.BatchNorm2d):
                if dependency_dict[name]["in_mask"] is not None:
                    tp.prune_batchnorm_in_channels(
                        module, idxs=dependency_dict[name]["in_mask"]
                    )
