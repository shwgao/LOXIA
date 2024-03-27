import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def convert_units(value, unit='FLOPS'):
    if unit.startswith('K'):
        return float(value) * 1e3
    elif unit.startswith('MF') or unit.startswith('MM'):
        return float(value) * 1e6
    elif unit.startswith('G'):
        return float(value) * 1e9
    else:
        return float(value)


def read_acc(file='accuracy.txt'):
    performances = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.split()
            app = items[0]
            model = items[1]
            if app not in performances:
                performances[app] = {}
            performances[app][model] = {'ACC': float(items[2]), }
    return performances


def read_performance(file='performance.txt'):
    performances = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.split()
            app = items[0]
            model = items[1]
            if app not in performances:
                performances[app] = {}
            performances[app][model] = {
                'ConvParams': int(items[2]),
                'LinearParams': int(items[3]),
                'ConvFlops': float(items[4]),
                'LinearFlops': float(items[5]),
                'Calflops-Flops': float(convert_units(items[6], items[7])),
                'Calflops-Macs': float(convert_units(items[8], items[9])),
                'Calflops-Params': float(convert_units(items[10], items[11])),
                'PeakMemory': float(items[12]),
                'Latency': float(items[13]),
                'StdTime': float(items[14]),
            }
    return performances


def speed_up_separate():
    performance = read_performance(file='./logs/performance.txt')
    apps = performance.keys()

    # apps = [['minist', 'CFD', 'puremd', 'fluidanimation', 'synthetic'], ['cifar10', 'EMDenoise', 'cosmoflow', 'stemdl', 'DMS', 'slstr', 'optical']]

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    bar_width = 0.45
    colors = ['#3AA6B9', 'orange', '#FF9EAA',
              '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']
    threshold = 1.6

    # draw original
    index = np.arange(len(apps)) * 1.2

    original_time = [performance[app]['original']['Latency'] for app in apps]
    original_std = [performance[app]['original']['StdTime'] for app in apps]
    pruned_time = [performance[app]['pruned']['Latency'] for app in apps]
    pruned_std = [performance[app]['pruned']['StdTime'] for app in apps]

    pruned_std = [ps / ot for ps, ot in zip(pruned_std, original_time)]
    original_std = [os / ot for os, ot in zip(original_std, original_time)]
    original_time_n = [1 for _ in original_time]
    pruned_speedup = [ot / pt for ot, pt in zip(original_time, pruned_time)]

    # linear below 5, log above 5
    # ax.bar(index, original_time_n, bar_width,
    #        label='Original', color=colors[0])
    ax.bar(index + bar_width + 0.05, pruned_speedup,
           bar_width, label='Pruned', color=colors[1])
    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(apps, rotation=45, ha='right', fontsize=20)

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=0.5, pad=0, sharex=ax)
    # axlog.bar(index, original_time_n, bar_width,
    #           label='Original', color=colors[0])
    axlog.bar(index + bar_width + 0.05, pruned_speedup,
              bar_width, label='Pruned', color=colors[1])
    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 180)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=14)
    axlog.yaxis.set_tick_params(labelsize=14)

    # put speedup on the top of the bar
    text_in_log, text_in_linear = [], []
    for i in range(len(apps)):
        if pruned_speedup[i] < threshold:
            text_in_linear.append(i)
        else:
            text_in_log.append(i)

    for j in text_in_log:
        axlog.text(index[j] + bar_width + 0.05, pruned_speedup[j],
                   f'{pruned_speedup[j]:.2f}x', ha='center', va='bottom', fontsize=18)

    for j in text_in_linear:
        ax.text(index[j] + bar_width + 0.05, pruned_speedup[j],
                f'{pruned_speedup[j]:.2f}x', ha='center', va='bottom', fontsize=18)

    # plt.subplots_adjust(wspace=0.00001)

    ax.set_ylabel('Speedup x', fontsize=20)
    fig.suptitle('Speedup by Applications', fontsize=20)
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    # ax.legend((rects1[0], rects2[0]), ('Original', 'Pruned'))

    # extend down side to show the x-axis labels
    plt.subplots_adjust(bottom=0.22)

    plt.savefig('./logs/Speedup_separate.png')
    plt.show()


def parameter_breakdown_separate():
    performance = read_performance()
    x = performance.keys()

    apps = [['CFD', 'puremd', 'fluidanimation', 'EMDenoise', 'synthetic'], [
        'minist', 'cifar10', 'cosmoflow', 'stemdl', 'DMS', 'slstr'], ['optical']]

    fig, ax = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [len(apps[0]), len(apps[1]), len(apps[2])],
                                                               'wspace': 0.2})
    bar_width = 0.45
    colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
              '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

    # draw original
    for i in range(3):
        index = np.arange(len(apps[i])) * 1.2
        pruned_conv_params = [performance[app]
                              ['pruned']['ConvParams'] for app in apps[i]]
        pruned_linear_params = [performance[app]
                                ['pruned']['LinearParams'] for app in apps[i]]
        original_conv_params = [performance[app]
                                ['original']['ConvParams'] for app in apps[i]]
        original_linear_params = [
            performance[app]['original']['LinearParams'] for app in apps[i]]

        prund_ratio = [(pruned_conv_params[i]+pruned_linear_params[i]) / (
            original_conv_params[i]+original_linear_params[i]) for i in range(len(apps[i]))]

        rects1 = ax[i].bar(index, original_conv_params, bar_width,
                           label='ConvParams', color='g', hatch='//', edgecolor='black')
        rects2 = ax[i].bar(index, original_linear_params, bar_width, bottom=original_conv_params, label='LinearParams', hatch='\\\\',
                           color='g', edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width+0.05, pruned_conv_params, bar_width, label='ConvParams', color='r',
                           hatch='//', edgecolor='black')
        rects4 = ax[i].bar(index + bar_width+0.05, pruned_linear_params, bar_width, bottom=pruned_conv_params, hatch='\\\\',
                           label='LinearParams', color='r', edgecolor='black')

        ax[i].set_xticks(index + bar_width / 2)
        ax[i].set_xticklabels(apps[i], rotation=45, ha='right')

        # put pruned ratio on the top of the bar
        for j in range(len(apps[i])):
            ax[i].text(index[j] + bar_width + 0.2, pruned_linear_params[j]+pruned_conv_params[j], f'{prund_ratio[j]*100:.2f}%',
                       ha='center', va='bottom', fontsize=8, rotation=50)

    # set ax[0]'s y-axis scientific notation
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # plt.subplots_adjust(wspace=0.00001)

    # set y-axis log scale
    # ax[2].set_yscale('log')

    ax[0].set_ylabel('Parameters', fontsize=14)
    fig.suptitle('Parameters by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    ax[1].legend((rects1[0], rects2[0], rects3[0], rects4[0]),
                 ('Original Conv', 'Original Linear', 'Pruned Conv', 'Pruned Linear'))

    # extend down side to show the x-axis labels
    plt.subplots_adjust(bottom=0.22)

    plt.savefig('./logs/parameter_breakdown_separate.png')
    plt.show()


def flops_breakdown_separate():
    performance = read_performance()
    x = performance.keys()

    apps = [['CFD', 'puremd', 'fluidanimation', 'synthetic',], ['minist', 'cifar10',], [
        'DMS', 'EMDenoise', ], ['stemdl', 'cosmoflow', 'optical'], ['slstr']]

    fig, ax = plt.subplots(1, 5, figsize=(11, 5), gridspec_kw={'width_ratios': [len(apps[0]), len(apps[1]), len(apps[2]), len(apps[3]), len(apps[4])],
                                                               'wspace': 0.2})
    bar_width = 0.45
    colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
              '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

    # draw original
    for i in range(5):
        index = np.arange(len(apps[i])) * 1.2
        pruned_conv_params = [performance[app]
                              ['pruned']['ConvFlops'] for app in apps[i]]
        pruned_linear_params = [performance[app]
                                ['pruned']['LinearFlops'] for app in apps[i]]
        original_conv_params = [performance[app]
                                ['original']['ConvFlops'] for app in apps[i]]
        original_linear_params = [performance[app]
                                  ['original']['LinearFlops'] for app in apps[i]]

        prund_ratio = [(pruned_conv_params[i]+pruned_linear_params[i]) / (
            original_conv_params[i]+original_linear_params[i]) for i in range(len(apps[i]))]

        rects1 = ax[i].bar(index, original_conv_params, bar_width,
                           label='ConvFlops', color='skyblue', hatch='//', edgecolor='black')
        rects2 = ax[i].bar(index, original_linear_params, bar_width, bottom=original_conv_params, label='LinearFlops', hatch='\\\\',
                           color='skyblue', edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width+0.05, pruned_conv_params, bar_width, label='ConvFlops', color='orange',
                           hatch='//', edgecolor='black')
        rects4 = ax[i].bar(index + bar_width+0.05, pruned_linear_params, bar_width, bottom=pruned_conv_params, hatch='\\\\',
                           label='LinearFlops', color='orange', edgecolor='black')

        ax[i].set_xticks(index + bar_width / 2)
        ax[i].set_xticklabels(apps[i], rotation=45, ha='right')

        # put pruned ratio on the top of the bar
        for j in range(len(apps[i])):
            ax[i].text(index[j] + bar_width + 0.2, pruned_linear_params[j]+pruned_conv_params[j], f'{prund_ratio[j]*100:.2f}%',
                       ha='center', va='bottom', fontsize=8, rotation=0)

    # set ax[0]'s y-axis scientific notation
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # plt.subplots_adjust(wspace=0.00001)

    # set y-axis log scale
    # ax[2].set_yscale('log')

    ax[0].set_ylabel('Flops', fontsize=14)
    fig.suptitle('Flops by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    # ax[4].legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Original Conv', 'Original Linear', 'Pruned Conv', 'Pruned Linear'))
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(apps[0]), bbox_to_anchor=(0.655, 0.87))

    plt.subplots_adjust(top=0.85)

    # extend down side to show the x-axis labels
    plt.subplots_adjust(bottom=0.22)

    plt.savefig('./logs/flops_breakdown_separate.png')
    plt.show()


def accuracy():
    performance = read_acc(file='accuracy.txt')
    x = performance.keys()

    apps = [['minist', 'cifar10', 'CFD', 'puremd', 'fluidanimation', 'DMS',
             'stemdl', 'synthetic'], ['cosmoflow', 'EMDenoise', 'slstr', 'optical']]

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={
        'width_ratios': [len(apps[0]), len(apps[1])],
        'wspace': 0.2})
    bar_width = 0.45
    colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
              '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

    for i in range(2):
        index = np.arange(len(apps[i])) * 1.2

        original_acc = [performance[app]['original']['ACC'] for app in apps[i]]
        pruned_acc = [performance[app]['pruned']['ACC'] for app in apps[i]]

        # if acc bigger than 1, then it is percentage, so divide by 100
        original_acc = [oa/100 if oa > 1 else oa for oa in original_acc]
        pruned_acc = [pa/100 if pa > 1 else pa for pa in pruned_acc]

        difference = [pa - oa for oa, pa in zip(original_acc, pruned_acc)]

        rects1 = ax[i].bar(index, original_acc, bar_width, label='Original ACC', color='skyblue',
                           edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width + 0.05, pruned_acc, bar_width, label='Pruned ACC', color='orange',
                           edgecolor='black')

        # put difference on the top of the bar
        # if differrence is positive, color is green, otherwise red
        for j in range(len(apps[i])):
            if difference[j] < 0:
                color = 'g' if i == 0 else 'r'
                text = f'{difference[j]:.4f}'
            else:
                color = 'r' if i == 0 else 'g'
                text = f'+{difference[j]:.4f}'
            ax[i].text(index[j] + bar_width + 0.2, pruned_acc[j],
                       text, ha='center', va='bottom', fontsize=8, color=color)

        ax[i].set_xticks(index + bar_width / 2)
        ax[i].set_xticklabels(apps[i], rotation=45, ha='right')

    ax[0].set_ylabel('higher better', fontsize=14)
    ax[1].set_ylabel('lower better', fontsize=14)
    fig.suptitle('Quality by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set y-axis log scale
    ax[1].set_yscale('log')

    # set legend ConvParams and LinearParams
    # ax[4].legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Original Conv', 'Original Linear', 'Pruned Conv', 'Pruned Linear'))
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(apps[0]), bbox_to_anchor=(0.655, 0.87))

    plt.subplots_adjust(top=0.85, bottom=0.22)
    # left and right side have too much space, so adjust it
    plt.subplots_adjust(left=0.1, right=0.95)

    plt.savefig('./logs/Accuracy.png')
    plt.show()


def peak_memory():
    performance = read_performance(file='./logs/before323-performance.txt')
    apps = performance.keys()

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    bar_width = 0.45
    colors = ['g', 'r']
    threshold = 1.5

    # draw original
    index = np.arange(len(apps)) * 1.2

    original_memory = [performance[app]['original']['PeakMemory'] for app in apps]
    pruned_memory = [performance[app]['pruned']['PeakMemory'] for app in apps]

    pruned_ratio = [pm / om for pm, om in zip(pruned_memory, original_memory)]

    # linear below 5, log above 5
    ax.bar(index, original_memory, bar_width, label='Original', color=colors[0])
    ax.bar(index + bar_width + 0.05, pruned_memory, bar_width, label='Pruned', color=colors[1])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(apps, rotation=45, ha='right', fontsize=20)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=14)

    # put the ratio on the pruned bar
    for j in range(len(apps)):
        ax.text(index[j] + bar_width + 0.05, pruned_memory[j], f'{pruned_ratio[j]*100:.2f}%',
                ha='center', va='bottom', fontsize=16)

    # plt.subplots_adjust(wspace=0.00001)

    ax.set_ylabel('Peak Memory', fontsize=20)
    # fig.suptitle('Speedup by Applications', fontsize=20)
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend
    ax.legend()

    # extend down side to show the x-axis labels
    plt.subplots_adjust(bottom=0.22)

    plt.savefig('./logs/peakmemory.png')
    # plt.show()


def layers_flops_breakdown():
    original_flops = {'conv_seq.0.conv': (3623878656.0, 872), 'conv_seq.1.conv': (1811939328.0, 3472), 'conv_seq.2.conv': (905969664.0, 13856), 'conv_seq.3.conv': (
        452984832.0, 55360), 'conv_seq.4.conv': (226492416.0, 221312), 'dense1': (2097152, 1048704), 'dense2': (16384, 8256), 'output': (512, 260)}
    pruned_flops = {'conv_seq.0.conv': (3623878656.0, 872), 'conv_seq.1.conv': (1811939328.0, 3472), 'conv_seq.2.conv': (905969664.0, 13856), 'conv_seq.3.conv': (
        226492416.0, 27680), 'conv_seq.4.conv': (12386304.0, 12110), 'dense1': (229376, 114816), 'dense2': (16384, 8256), 'output': (512, 260)}
    original_params = {'conv_seq.0.conv': (872, 0), 'conv_seq.1.conv': (3472, 0), 'conv_seq.2.conv': (13856, 0), 'conv_seq.3.conv': (55360, 0),
                       'conv_seq.4.conv': (221312, 0), 'dense1': (0, 1048704), 'dense2': (0, 8256), 'output': (0, 260)}
    pruned_params = {'conv_seq.0.conv': (872, 0), 'conv_seq.1.conv': (3472, 0), 'conv_seq.2.conv': (13856, 0), 'conv_seq.3.conv': (27680, 0),
                     'conv_seq.4.conv': (12110, 0), 'dense1': (0, 114816), 'dense2': (0, 8256), 'output': (0, 260)}

    # x_labels = ['Layer1.conv', 'Layer2.conv', 'Layer3.conv', 'Layer4.conv', 'Layer5.conv', 'Layer6.linear', 'Layer7.linear', 'Layer8.linear']
    x_labels = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Dense1', 'Dense2', 'Dense3']
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    bar_width = 0.2

    original_flops = [original_flops[layer][0] for layer in original_flops.keys()]
    pruned_flops = [pruned_flops[layer][0] for layer in pruned_flops.keys()]
    
    original_ratio = [original_flops[i] / sum(original_flops) for i in range(len(x_labels))]
    pruned_ratio = [pruned_flops[i] / sum(pruned_flops) for i in range(len(x_labels))]
    
    index = np.arange(len(x_labels)) * 1
    ax.bar(index, original_flops, bar_width, label='Original FLOPS', color='skyblue', edgecolor='black')
    ax.bar(index + bar_width + 0.02, pruned_flops, bar_width, label='Pruned FLOPS', color='skyblue', edgecolor='black', hatch='++')
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_labels, ha='left', fontsize=20)

    ax.set_ylabel('FLOPS', fontsize=20)
    ax.yaxis.set_tick_params(labelsize=18)

    # draw the second bar chart of parameters use the same x-axis but different y-axis, and put the second y-axis on the right side
    ax2 = ax.twinx()
    original_params = [sum(original_params[layer]) for layer in original_params.keys()]
    pruned_params = [sum(pruned_params[layer]) for layer in pruned_params.keys()]

    original_ratio_p = [original_params[i] / sum(original_params) for i in range(len(x_labels))]
    pruned_ratio_p = [pruned_params[i] / sum(pruned_params) for i in range(len(x_labels))]

    ax2.bar(index + 2*(bar_width + 0.02), original_params, bar_width, label='Original Params', color='orange', edgecolor='black')
    ax2.bar(index + 3*(bar_width + 0.02), pruned_params, bar_width, label='Pruned Params', color='orange', edgecolor='black', hatch='++')

    # put original and pruned ratio on the top of the bar
    for j in range(len(x_labels)):
        text = '<1%' if pruned_ratio[j] < 0.01 else f'{pruned_ratio[j] * 100:.0f}%'
        ax.text(index[j]+bar_width+0.05, pruned_flops[j], text, ha='center', va='bottom', fontsize=14)
        text = '<1%' if original_ratio[j] < 0.01 else f'{original_ratio[j] * 100:.0f}%'
        ax.text(index[j], original_flops[j], text, ha='center', va='bottom', fontsize=14)

    # # put original and pruned ratio on the top of the bar
    for j in range(len(x_labels)):
        text = '<1%' if original_ratio_p[j] < 0.01 else f'{original_ratio_p[j] * 100:.0f}%'
        ax2.text(index[j] + 2 * (bar_width + 0.05), original_params[j], text, ha='center', va='bottom', fontsize=14)
        text = '<1%' if pruned_ratio_p[j] < 0.01 else f'{pruned_ratio_p[j] * 100:.0f}%'
        ax2.text(index[j] + 3 * (bar_width + 0.05), pruned_params[j], text, ha='center', va='bottom', fontsize=14)

    ax2.set_ylabel('Parameters', fontsize=20)
    ax2.yaxis.set_tick_params(labelsize=18)

    # ax2.set_xticklabels(x_labels, ha='right', fontsize=25)

    # set ax2 y-axis scientific notation
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # set y-axis log scale
    ax2.set_yscale('log')
    # ax.set_yscale('log')

    # fig.suptitle('FLOPS by Layers', fontsize=20)

    # set legend original and pruned for both flops and params
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax.legend(all_handles, all_labels, loc='upper right', fontsize=20)

    # cut the blank space on the right side
    # plt.subplots_adjust(right=0.95)
    # plt.subplots_adjust(left=0.05)
    # tight_layout()
    fig.tight_layout()

    plt.savefig('./logs/layers_flops_breakdown.png')
    plt.show()


if __name__ == '__main__':
    # performances = read_performance()
    # print(performances)

    # parameter_breakdown()
    # flops_breakdown()
    # parameter_breakdown_separate()
    # flops_breakdown_separate()
    # speed_up()
    speed_up_separate()
    # accuracy()
    # layers_flops_breakdown()
    # peak_memory()
