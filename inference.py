import torch
import argparse
from calflops import calculate_flops
from torch_pruning.utils.benchmark import measure_memory, measure_latency
from utils import measure_parameters, measure_flops


def performance_test(model, device, input_shape, repeat=1):
    model.eval()
    model.to(device)

    with torch.no_grad():
        p_conv, p_lin = measure_parameters(model)
        print('Calculating FLOPs...')
        flops, macs, param_num = calculate_flops(model=model,
                                                 input_shape=input_shape,
                                                 output_as_string=True,
                                                 print_results=False,
                                                 output_precision=6)
        expected_flops, _ = model.get_exp_flops_l0()
        print(f"Pruning FLOPs: {expected_flops} ")
        f_conv, f_lin = measure_flops(model)
        print(f"Conv Params: {p_conv}   Linear Params: {p_lin} ")
        print(f"Conv Flops: {f_conv}   Linear Flops: {f_lin} ")
        print(
            f"Tested by calflops:   FLOPs:{flops}   MACs:{macs}   Params:{param_num} "
        )

        torch.cuda.empty_cache()

        model.to(device)
        example_inputs = torch.rand(input_shape)
        example_inputs = example_inputs.repeat_interleave(repeat, 0).to(device)

        print('Testing input with shape: ', example_inputs.shape)
        memory = measure_memory(model, example_inputs, device=device) if device != 'cpu' else 0
        base_latency, std_time = measure_latency(
            model, example_inputs, 200, 10)
        print("Base Latency: {:.4f}+-({:.4f}) ms, Base MACs: {}, Peak Memory: {:.4f}M\n"
              .format(base_latency, std_time, flops, memory / (1024*1024)))
        
        flops_of_layers(model)
        parameters_of_layers(model)


def flops_of_layers(model):
    flops = {
        name: module.count_expected_flops_and_l0()
        for name, module in model.named_modules()
        if hasattr(module, 'count_expected_flops_and_l0')
    }
    print(flops)


def parameters_of_layers(model):
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'qz_loga'):
            params[name] = measure_parameters(module)

    print(params)


def inference(arg):
    val_loader = None
    using_reg = arg.model != 'original'

    print('\n Creating model and preparing dataset...:')
    if arg.application == "minist":
        from applications.minist import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 111000, 256
    elif arg.application == "cifar10":
        from applications.cifar10 import model, dataset, launch, input_shape
        init_model = model.L0LeNet5(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 1400, 256
    elif arg.application == "puremd":
        from applications.puremd import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 440000, 1024
    elif arg.application == "CFD":
        from applications.CFD import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 167000, 1024
    elif arg.application == "fluidanimation":
        from applications.fluidanimation import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 170000, 1024
    elif arg.application == "cosmoflow":
        from applications.cosmoflow import model, dataset, launch, input_shape
        init_model = model.CosmoFlow(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 5, 64
    elif arg.application == "EMDenoise":
        from applications.EMDenoise import model, dataset, launch, input_shape
        init_model = model.EMDenoiseNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 52, 64
    elif arg.application == "DMS":
        from applications.DMS import model, dataset, launch, input_shape
        init_model = model.DMSNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 150, 64
    elif arg.application == "optical":
        from applications.optical import model, dataset, launch, input_shape
        init_model = model.Autoencoder(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 30, 256
    elif arg.application == "stemdl":
        from applications.stemdl import model, dataset, launch, input_shape
        init_model = model.VGG11(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 95, 256
    elif arg.application == "slstr":
        from applications.slstr import model, dataset, launch, input_shape
        init_model = model.UNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 10, 100
    elif arg.application == "synthetic":
        from applications.synthetic import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 310000, 100
    else:
        print("Application not found")
        return

    if arg.task == 'quality':
        _, val_loader = dataset.get_loader(
            batch_size=batch_size, val_only=True)

    print('Preparation done.')

    model_s = 'original' if arg.model == 'original' else 'pruned'
    model_pth = f"./checkpoints/{arg.application}/{model_s}_model.pth.tar"
    print(f"\nTesting {model_s} model\'s {arg.task} on device {arg.device}: \n")

    state = torch.load(model_pth, map_location='cpu')
    print(f"Current quality of model: {state['curr_prec1']}")
    init_model.load_state_dict(state['state_dict'], strict=False)

    if model_s == 'pruned':
        print('Testing pruned model, pruning...')
        init_model.prune_model()
        print('Pruning done')
    else:
        print('Testing original model...')

    if arg.task == 'quality':
        quality = launch.validate(
            val_loader, init_model, launch.loss_fn, inference=True)
        print('Quality of the model: ', quality, '\n')
    else:
        performance_test(init_model, args.device, input_shape, repeat=intensive_repeat)
        # performance_test(init_model, args.device, input_shape, repeat=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument("--application", type=str, default="cifar10",
                        help="CFD or fluidanimation or puremd or cosmoflow or EMDenoise or minist "
                        "or DMS or optical or stemdl, slstr or synthetic or cifar10")
    parser.add_argument("--state_dir", type=str, default="../checkpoints-v0/")
    parser.add_argument("--model", type=str,
                        default="original", help="original or pruned")
    parser.add_argument("--task", type=str,
                        default="performance", help="performance or quality")
    parser.add_argument("--device", type=str,
                        default='cpu', help="0, 1, ...")
    params = parser.parse_args()

    args = parser.parse_args()
    # args.application = 'fluidanimation'
    # args.task = 'quality'
    args.model = 'pruned'
    inference(args)
