import json
import torch
import torch.nn as nn
from copy import deepcopy
from base_layers import L0Conv2d, L0Dense, L0Conv3d, BaseModel
import os


class EMDenoiseNet(BaseModel):
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
        budget = settings["initial_budget"]
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
                dependency_dict[name] = {'in_mask': dependency_dict[pre_module]['out_mask'], 'out_mask': None,
                                         'type': 'bn'}
            else:
                continue

        # dependency of skip layers
        offset = self.block3[3].m.shape[1]
        dependency_dict["block5.1"]["in_mask"] = dependency_dict["block5.1"]["in_mask"] + [x + offset for x in dependency_dict["block3.3"]["out_mask"]]

        offset = self.block2[3].m.shape[1]
        dependency_dict["block6.1"]["in_mask"] = dependency_dict["block6.1"]["in_mask"] + [x + offset for x in dependency_dict["block2.3"]["out_mask"]]

        offset = self.block1[3].m.shape[1]
        dependency_dict["block7.1"]["in_mask"] = dependency_dict["block7.1"]["in_mask"] + [x + offset for x in dependency_dict["block1.3"]["out_mask"]]

        return dependency_dict
