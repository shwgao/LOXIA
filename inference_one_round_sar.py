import os
import numpy as np
import onnxruntime as ort
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
        memory = measure_memory(model, example_inputs, device=device)
        memory /= (1024*1024)
        base_latency, std_time = measure_latency(
            model, example_inputs, 200, 10)
        print("Base Latency: {:.4f}+-({:.4f}) ms, Base MACs: {}, Peak Memory: {:.4f}M\n"
              .format(base_latency, std_time, flops, memory))
        # keep 4 decimal places for latency and memory
        base_latency = round(base_latency, 4)
        memory = round(memory, 4)
        std_time = round(std_time, 4)
        
        performance_dict["ConvParams"] = p_conv
        performance_dict["LinearParams"] = p_lin
        performance_dict["ConvFlops"] = f_conv
        performance_dict["LinearFlops"] = f_lin
        performance_dict["Calflops-Flops"] = flops
        performance_dict["Calflops-Macs"] = macs
        performance_dict["Calflops-Params"] = param_num
        performance_dict["PeakMemory"] = memory
        performance_dict["Latency"] = base_latency
        performance_dict["StdTime"] = std_time

    torch.cuda.empty_cache()
    
    return performance_dict


def transfer_model(arg):
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
        intensive_repeat, batch_size = 8000, 256
    elif arg.application == "puremd":
        from applications.puremd import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 440000, 1024
    elif arg.application == "CFD":
        from applications.CFD import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 180000, 1024
    elif arg.application == "fluidanimation":
        from applications.fluidanimation import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 170000, 1024
    elif arg.application == "cosmoflow":
        from applications.cosmoflow import model, dataset, launch, input_shape
        init_model = model.CosmoFlow(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 6, 64
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
        init_model = model.WideResNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 120, 256
    elif arg.application == "slstr":
        from applications.slstr import model, dataset, launch, input_shape
        init_model = model.UNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 10, 100
    elif arg.application == "synthetic":
        from applications.synthetic import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 340000, 100
    else:
        print("Application not found")
        return

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

    # save the whole model to infer the performance, save it as onnx model
    if not os.path.exists(f"./whole_model/{arg.application}"):
        os.makedirs(f"./whole_model/{arg.application}")
    # save as onnx
    torch.onnx.export(init_model, torch.rand(input_shape),
                      f"./whole_model/{arg.application}/{model_s}_model.onnx", verbose=False)


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
        intensive_repeat, batch_size = 8000, 256
    elif arg.application == "puremd":
        from applications.puremd import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 440000, 1024
    elif arg.application == "CFD":
        from applications.CFD import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 180000, 1024
    elif arg.application == "fluidanimation":
        from applications.fluidanimation import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 170000, 1024
    elif arg.application == "cosmoflow":
        from applications.cosmoflow import model, dataset, launch, input_shape
        init_model = model.CosmoFlow(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 6, 64
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
        init_model = model.WideResNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 120, 256
    elif arg.application == "slstr":
        from applications.slstr import model, dataset, launch, input_shape
        init_model = model.UNet(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 10, 100
    elif arg.application == "synthetic":
        from applications.synthetic import model, dataset, launch, input_shape
        init_model = model.MLP(inference=True, using_reg=using_reg)
        intensive_repeat, batch_size = 340000, 100
    else:
        print("Application not found")
        return

    if arg.task == 'quality':
        _, val_loader = dataset.get_loader(
            batch_size=batch_size, val_only=True)
        
    print('Preparation done.')

    model_s = 'original' if arg.model == 'original' else 'pruned'
    model_pth = f"./whole_model/{arg.application}/{model_s}_model.onnx"
    print(f"\nTesting {model_s} model\'s {arg.task} on device {arg.device}: \n")

    sess = ort.InferenceSession(model_pth)

    batch_inputs = np.random.rand(intensive_repeat, *input_shape).astype(np.float32)
    # batch_inputs = np.random.rand(*input_shape).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # warm up
    for i in range(10):
        sess.run([output_name], {input_name: batch_inputs[i % len(batch_inputs)]})

    s_t = torch.cuda.Event(enable_timing=True)
    e_t = torch.cuda.Event(enable_timing=True)
    s_t.record()
    for i in range(20):
        sess.run([output_name], {input_name: batch_inputs[i % len(batch_inputs)]})
    e_t.record()
    torch.cuda.synchronize()
    latency = s_t.elapsed_time(e_t) / 20
    print(f"Latency: {latency} ms")


    if arg.task == 'quality':
        quality = launch.validate(
            val_loader, init_model, launch.loss_fn, inference=True)
        print('Quality of the model: ', quality, '\n')
    else:
        performance_test(init_model, args.device,
                         input_shape, repeat=batch_size)
    
    # record the performance in 'performance.txt'
    with open('performance.txt', 'a') as f:
        f.write(''.join([str(performance_dict[col]).ljust(max_len) for col in columns]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument("--application", type=str, default="synthetic",
                        help="CFD or fluidanimation or puremd or cosmoflow or EMDenoise or minist "
                        "or DMS or optical or stemdl, slstr or synthetic")
    parser.add_argument("--state_dir", type=str, default="../checkpoints/")
    parser.add_argument("--model", type=str,
                        default="original", help="original or pruned")
    parser.add_argument("--task", type=str,
                        default="performance", help="performance or quality")
    parser.add_argument("--device", type=str,
                        default='cuda:0', help="0, 1, ...")
    params = parser.parse_args()

    args = parser.parse_args()

    applications = ['minist', "cifar10", "puremd", "CFD", "fluidanimation", "cosmoflow", "EMDenoise", "DMS", "optical", "stemdl", "slstr", "synthetic"]
    
    global performance_dict, columns, max_len
    # create performance.txt and write the header, make sure every column has the same length
    columns = ["Application", "Model", "ConvParams", "LinearParams", "ConvFlops", "LinearFlops", "Calflops-Flops", "Calflops-Macs", 
               "Calflops-Params", "PeakMemory", "Latency", "StdTime"]
    max_len = max(len(col) for col in columns)
    with open('performance.txt', 'w') as f:
        f.write(''.join([col.ljust(max_len) for col in columns]) + '\n')
    
    performance_dict = {}

    for app in applications:
        args.application = app
        
        performance_dict["Application"] = app
        print(f"***************************Testing {app} application**************************\n\n")
        
        for model in ["original", "pruned"]:
            args.model = model
            
            performance_dict["Model"] = model
            
            inference(args)
            # transfer_model(args)
