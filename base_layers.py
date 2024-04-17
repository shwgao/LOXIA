from copy import deepcopy
import torch_pruning as tp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn.modules.utils import _triple as triple
from torch.autograd import Variable
from torch.nn import init

limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


class BaseModel(nn.Module):
    def update_budget(self, budget):
        for layer in self.layers:
            layer.update_budget(budget)

    def update_temperature(self, temperature):
        for layer in self.layers:
            layer.update_temperature(temperature)

    def constrain_parameters(self):
        for layer in self.layers:
            if layer.use_reg:
                layer.constrain_parameters()

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            if layer.use_reg:
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
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

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

    def prune_model(self):
        for layer in self.layers:
            if isinstance(layer, L0Dense):
                if layer.use_reg:
                    layer.prepare_for_inference()
        dependency_dict = self.build_dependency_graph()

        for name, module in self.named_modules():
            if name in dependency_dict:
                if isinstance(module, (L0Dense, nn.Linear)):
                    if dependency_dict[name]['in_mask'] is not None:
                        tp.prune_linear_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                    if dependency_dict[name]['out_mask'] is not None:
                        tp.prune_linear_out_channels(module, idxs=dependency_dict[name]['out_mask'])
                elif isinstance(module, (L0Conv2d, nn.Conv2d, L0Conv3d)):
                    if dependency_dict[name]['in_mask'] is not None:
                        tp.prune_conv_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                    if dependency_dict[name]['out_mask'] is not None:
                        tp.prune_conv_out_channels(module, idxs=dependency_dict[name]['out_mask'])
                elif isinstance(module, nn.BatchNorm2d):
                    if dependency_dict[name]['in_mask'] is not None:
                        tp.prune_batchnorm_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                else:
                    print(f"Module {name} is not supported for pruning.")


class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""

    def __init__(
            self,
            in_features,
            out_features,
            device="cuda:0",
            bias=True,
            weight_decay=1.0,
            droprate_init=0.5,
            temperature=2.0 / 3.0,
            lamba=1.0,
            local_rep=False,
            use_reg=True,
            budget=0.49,
    ):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.weight = None
        self.qz_loga = Parameter(torch.Tensor(in_features)) if use_reg else None
        self.mask = None
        self.m = None  # quantile_concrete qz_loga
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        self.device = device
        self.trained_z = None
        self.use_reg = use_reg
        self.budget = budget
        self.relu = nn.ReLU()
        self.flatten_select = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.reset_parameters()
        self.transposed = True
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode="fan_out")

        if self.use_reg:
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self):
        if self.use_reg:
            self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)  # scale
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        x = x.to(self.device)
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        logpw_col = torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        qz = 1 - self.cdf_qz(0)
        logpw = torch.sum(qz * logpw_col)
        logpb = (
            -torch.sum(0.5 * self.prior_prec * self.bias.pow(2)) if self.use_bias else 0
        )
        # print("logpw:",logpw.item(),"logpb:",logpb.item())
        return logpw + logpb

    def update_budget(self, budget):
        self.budget = budget

    def update_temperature(self, temperature):
        self.temperature = temperature

    def regularization(self):
        return self._reg_w() if self.use_reg else 0.

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        # if self.use_reg:
        #     # ppos = torch.sum(1 - self.cdf_qz(0))
        #     z = self.sample_z(1, sample=False)
        #     ppos = torch.sum(z).item()
        # else:
        ppos = self.in_features
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        # return expected_flops.data[0], expected_l0.data[0]
        return expected_flops, expected_l0

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).to(self.device).uniform_(self.budget, 1 - self.budget)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = (
                torch.sigmoid(self.qz_loga)
                .view(1, self.in_features)
                # .expand(batch_size, self.in_features)
            )
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(
            self.get_eps(self.in_features)
        )
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def get_mask(self):
        z = self.quantile_concrete(self.get_eps(self.in_features))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1)

    def update_z(self, sample=False):
        z = self.sample_z(1, sample)
        z[z < 0.1] = 0
        self.trained_z = z

    def forward(self, input_):
        if self.use_reg:
            if self.local_rep or not self.training:
                z = self.sample_z(input_.size(0), sample=self.training)
                xin = input_.mul(z)
                # xin = input.mul(self.trained_z)
                output = torch.matmul(xin, self.weights)  # output = xin.mm(self.weight)
            else:
                weight = self.sample_weights()
                output = torch.matmul(input_, weight)  # output = input_.mm(weight)

        else:
            output = input_.mm(self.weights)

        if self.use_bias:
            output.add_(self.bias)
        return output

    def __repr__(self):
        s = (
            "{name}({in_features} -> {out_features}, droprate_init={droprate_init}, "
            "lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, "
            "local_rep={local_rep}"
        )
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def prepare_for_inference(self):
        if self.use_reg:
            pi = (torch.sigmoid(self.qz_loga).view(1, self.in_features))
            self.m = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1).view(-1, 1)
            self.mask = (self.m.flatten() == 0).nonzero().flatten().tolist()
            self.weights.data = self.m * self.weights.data
        self.weight = self.weights

        self.weights = None
        self.qz_loga = None

        # rewrite the forward function
        def new_forward(input_):
            # output = torch.matmul(input_, self.weight)
            output = input_.mm(self.weight)
            if self.use_bias:
                output.add_(self.bias)
            return output

        self.forward = new_forward

    def set_couple_prune1(self, input_shape, pre_mask):
        # self is not using regualarization
        mask = torch.ones(input_shape)
        mask[:, pre_mask, :, :] = 0
        self.mask = (mask.flatten() == 0).nonzero().flatten().tolist()

    def set_couple_prune(self, input_shape, pre_mask):
        mask = torch.ones(input_shape)
        mask[:, pre_mask, :, :] = 0
        m = self.m

        self.mask = (m.flatten() * mask.flatten().to(m.device) == 0).nonzero().flatten().tolist()
        # self.flatten_select = (m.flatten()*mask.flatten().to(m.device) != 0).nonzero().flatten()
        keep_index = list(set(range(mask.shape[1])) - set(pre_mask))
        if len(input_shape) == 5:
            self.flatten_select = (m.flatten() * mask.flatten().to(m.device)).view(input_shape)[:, keep_index, :, :, :]
        else:
            self.flatten_select = (m.flatten() * mask.flatten().to(m.device)).view(input_shape)[:, keep_index, :, :]
        self.flatten_select = self.flatten_select.flatten().nonzero().flatten()  # .tolist()

        self.m = None

        # rewrite the forward function
        def new_forward(input_):
            # select input with mask
            # s_time = time.time()
            input_ = input_[:, self.flatten_select]
            # input_ = torch.take(input_, self.flatten_select)
            # e_time = time.time()
            # print('scatter time: {}'.format(e_time - s_time))
            # output = torch.matmul(input_, self.weight)
            output = input_.mm(self.weight)
            if self.use_bias:
                output.add_(self.bias)
            return output

        self.forward = new_forward


class L0Conv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        use_reg=True,
        device="cpu",
        budget=0.49,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv2d, self).__init__()
        self.m = None
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.use_reg = use_reg
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.budget = budget
        self.relu = nn.ReLU()
        self.floatTensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.use_bias = False
        self.bias = None
        self.weights = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.weight = None
        self.qz_loga = Parameter(torch.Tensor(out_channels)) if use_reg else None
        self.mask = None
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        self.transposed = False
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode="fan_in")

        if self.use_reg:
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self):
        if self.use_reg:
            self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def update_budget(self, budget):
        self.budget = budget

    def update_temperature(self, temperature):
        self.temperature = temperature

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = 1 - self.cdf_qz(0)
        logpw_col = (
            torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 3)
            .sum(2)
            .sum(1)
        )
        logpw = torch.sum(q0 * logpw_col)
        # logpb = 0 if not self.use_bias else - torch.sum(q0 * (.5 * self.prior_prec * self.bias.pow(2) -
        #                                                             self.lamba))
        logpb = (
            -torch.sum(
                            q0 * (0.5 * self.prior_prec * self.bias.pow(2) - self.lamba)
                        ) if self.use_bias else 0
        )
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # if self.use_reg:
        #     # ppos = torch.sum(1 - self.cdf_qz(0))
        #     ppos = torch.sum(self.sample_z(1, sample=False)).item()
        # else:
        ppos = self.out_channels
        n = (
            self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        )  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0])
            / self.stride[0]
        ) + 1  # for rows
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]
        ) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        # return expected_flops.data[0], expected_l0.data[0]
        return expected_flops, expected_l0
    
    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size, device=self.device).uniform_(self.budget, 1 - self.budget)
        eps = Variable(eps).to(self.qz_loga.device)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def get_mask(self):
        pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
        return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.dim_z)).view(
            self.dim_z, 1, 1, 1
        )
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = self.bias if self.use_bias else None
        if not self.use_reg:
            return F.conv2d(
                input_,
                self.weights,
                b,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        if self.local_rep or not self.training:
            output = F.conv2d(
                input_,
                self.weights,
                b,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            z = self.sample_z(output.size(0), sample=self.training)
            return output.mul(z)
        else:
            weights = self.sample_weights()
            return F.conv2d(
                input_,
                weights,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, "
            "lamba={lamba}, local_rep={local_rep}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def prepare_for_inference(self):
        if not self.use_reg:
            self.weight = self.weights
            return
        pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
        self.m = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        self.mask = (self.m.flatten() == 0).nonzero().flatten().tolist()
        self.weight = self.weights
        self.weight.data = self.weight.data * self.m.view(-1, 1, 1, 1)
        if self.use_bias:
            self.bias.data = self.bias.data * self.m.flatten()
        else:
            self.bias = None

        self.weights = None
        self.qz_loga = None

        keep_idxs = list(set(range(self.out_channels)) - set(self.mask))
        if not keep_idxs:
            # if remove all the channels, replace the forward function with a function that returns zeros
            def new_forward(input_):
                return torch.zeros_like(input_).to(self.weight.device)
        else:
            keep_idxs.sort()
            self.m = torch.index_select(self.m, 1, torch.LongTensor(keep_idxs).to(self.weight.device).contiguous())

            self.weights = None

            # rewrite the forward function
            def new_forward(input_):
                if self.input_shape is None:
                    self.input_shape = input_.size()
                return F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        self.forward = new_forward

    def prepare_for_inference1(self):
        pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
        self.m = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        self.mask = (self.m.flatten() == 0).nonzero().flatten().tolist()
        self.weight = self.weights
        self.bias = self.bias if self.use_bias else None
        keep_idxs = sorted(set(range(self.out_channels)) - set(self.mask))
        self.m = torch.index_select(self.m, 1, torch.LongTensor(keep_idxs).to(self.weight.device).contiguous())

        self.weights = None

        # rewrite the forward function
        def new_forward(input_):
            if self.input_shape is None:
                self.input_shape = input_.size()
            output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = output.mul(self.m)
            return output

        self.forward = new_forward


class L0Conv3d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        use_reg=True,
        device="cpu",
        budget=1e-6,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv3d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = triple(kernel_size)
        self.stride = triple(stride)
        self.padding = triple(padding)
        self.dilation = triple(dilation)
        self.output_padding = triple(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.device = device
        self.transposed = False
        self.floatTensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.use_bias = False
        self.bias = None
        self.weights = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.mask = None
        self.m = None
        self.weight = None
        self.qz_loga = Parameter(torch.Tensor(out_channels)) if use_reg else None
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep
        self.use_reg = use_reg
        self.budget = budget

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        print(self)

    def get_mask(self):
        pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1, 1)
        return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode="fan_in")

        if self.use_reg:
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self):
        if self.use_reg:
            self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = (
            torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 4)
            .sum(3).sum(2).sum(1))
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = (
            -torch.sum(
                            (1 - q0) * (0.5 * self.prior_prec * self.bias.pow(2) - self.lamba)
                        ) if self.use_bias else 0
        )
        return logpw + logpb

    def regularization(self):
        return self._reg_w() if self.use_reg else 0

    def count_expected_flops_and_l0(self):
        # if self.use_reg:
        #     ppos = torch.sum(self.sample_z(1, sample=False))
        # else:
        ppos = self.out_channels
        n = (self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels)
        flops_per_instance = n + (n - 1)
        num_instances_per_filter = ((self.input_shape[-3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1
        num_instances_per_filter *= ((self.input_shape[-2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1
        num_instances_per_filter *= ((self.input_shape[-1] - self.kernel_size[2] + 2 * self.padding[2]) / self.stride[2]) + 1
        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos
        expected_l0 = n * ppos
        if self.use_bias:
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos
        return expected_flops, expected_l0

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size, device=self.device).uniform_(self.budget, 1 - self.budget)
        eps = Variable(eps)
        return eps

    def update_budget(self, budget):
        self.budget = budget

    def update_temperature(self, temperature):
        self.temperature = temperature

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps([self.dim_z])).view(
            self.dim_z, 1, 1, 1, 1
        )
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = self.bias if self.use_bias else None

        if not self.use_reg:
            return F.conv3d(
                input_,
                self.weights,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        if self.local_rep or not self.training:
            output = F.conv3d(
                input_,
                self.weights,
                b,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            z = self.sample_z(output.size(0), sample=self.training)
            return output.mul(z)
        else:
            weights = self.sample_weights()
            return F.conv3d(
                input_,
                weights,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, "
            "lamba={lamba}, local_rep={local_rep}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def prepare_for_inference(self):
        pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1, 1)
        self.m = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        self.m = self.m.detach()
        self.mask = (self.m.squeeze() == 0).nonzero().squeeze().tolist()
        self.weight = self.weights
        self.bias = self.bias if self.use_bias else None

        self.weight.data = self.weight.data * self.m.view(-1, 1, 1, 1, 1)
        if self.use_bias:
            self.bias.data = self.bias.data * self.m.squeeze()

        keep_idxs = sorted(set(range(self.out_channels)) - set(self.mask))
        self.m = torch.index_select(self.m, 1, torch.LongTensor(keep_idxs).to(self.weight.device).contiguous())

        self.weights = None
        self.qz_loga = None

        # rewrite the forward function
        def new_forward(input_):
            if self.input_shape is None:
                self.input_shape = input_.size()

            output = F.conv3d(
                input_,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            # output = output.mul(self.m)
            return output

        self.forward = new_forward


class MAPDense(Module):
    def __init__(
        self, in_features, out_features, bias=True, weight_decay=1.0, **kwargs
    ):
        super(MAPDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.floatTensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode="fan_out")

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, **kwargs):
        pass

    def _reg_w(self, **kwargs):
        logpw = -torch.sum(self.weight_decay * 0.5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = -torch.sum(self.weight_decay * 0.5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        expected_flops = (2 * self.in_features - 1) * self.out_features
        expected_l0 = self.in_features * self.out_features
        if self.bias is not None:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops, expected_l0

    def forward(self, input):
        output = input.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ", weight_decay: "
            + str(self.weight_decay)
            + ")"
        )


class MAPConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        weight_decay=1.0,
        budget=0.49,
        device='cpu',
        **kwargs
    ):
        super(MAPConv2d, self).__init__()
        self.weight_decay = weight_decay
        self.floatTensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.budget = budget
        self.transposed = False
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.input_shape = None
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode="fan_in")

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, thres_std=1.0):
        pass

    def _reg_w(self, **kwargs):
        logpw = -torch.sum(self.weight_decay * 0.5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = -torch.sum(self.weight_decay * 0.5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        ppos = self.out_channels
        n = (
            self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        )  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0])
            / self.stride[0]
        ) + 1  # for rows
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]
        ) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.bias is not None:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops, expected_l0

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        return F.conv2d(
            input_,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size} "
            ", stride={stride}, weight_decay={weight_decay}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
