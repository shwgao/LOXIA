import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 26
BIGGER_SIZE = 26

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE )     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE )    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE )    # legend fontsize

font1 = {'color':'black','size':BIG_SIZE, 'weight':'bold'}

colors = ['skyblue','skyblue','orange']

figure_size = (20, 8)

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


def read_performance(file='./logs/before323-performance.txt'):
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
    # convert all first lower str of the keys to upper
    performances = {k[0].upper()+k[1:]: v for k, v in performances.items()}

    return performances


def speed_up_separate():
    performance = read_performance(file='./logs/performance111sc.txt')
    apps = performance.keys()

    # apps = [['minist', 'CFD', 'puremd', 'fluidanimation', 'synthetic'], ['cifar10', 'EMDenoise', 'cosmoflow', 'stemdl', 'DMS', 'slstr', 'optical']]
    apps_shortname = ['Min.','Cif.','Pur.','CFD','Flu.','Cos.','EMD.','DMS','Opt.','Ste.','Sls.','Syn.']

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.7
    # colors = ['#3AA6B9', 'orange', '#FF9EAA',
    #           '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']
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
    ax.bar(index, pruned_speedup,
           bar_width, label='Pruned', color=colors[1])
    ax.set_ylim(0, threshold)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(apps_shortname, ha='right', fontsize=BIG_SIZE, fontweight='bold') #  rotation=45,

    divider = make_axes_locatable(ax)
    axlog = divider.append_axes("top", size=2.0, pad=0, sharex=ax)
    # axlog.bar(index, original_time_n, bar_width,
    #           label='Original', color=colors[0])
    axlog.bar(index, pruned_speedup,
              bar_width, label='Pruned', color=colors[1])
    axlog.set_yscale('log')
    axlog.set_ylim(threshold, 180)

    axlog.spines['bottom'].set_visible(False)
    axlog.xaxis.set_ticks_position('top')
    plt.setp(axlog.get_xticklabels(), visible=False)

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    axlog.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put speedup on the top of the bar
    text_in_log, text_in_linear = [], []
    for i in range(len(apps)):
        if pruned_speedup[i] < threshold:
            text_in_linear.append(i)
        else:
            text_in_log.append(i)

    for j in text_in_linear:
        ax.text(index[j], pruned_speedup[j],
                f'  {pruned_speedup[j]:.2f}x', ha='center', va='bottom',fontsize=BIG_SIZE)  #  rotation=90,

    for j in text_in_log:
        axlog.text(index[j], pruned_speedup[j],
                   f'  {pruned_speedup[j]:.2f}x', ha='center', va='bottom',  fontsize=BIG_SIZE)  # rotation=90,


    # plt.subplots_adjust(wspace=0.00001)

    ax.set_ylabel('Speedup x', fontsize=BIG_SIZE, fontweight='bold')
    # fig.suptitle('Speedup by Applications', fontsize=20)
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    # ax.legend((rects1[0], rects2[0]), ('Original', 'Pruned'))

    # extend down side to show the x-axis labels
    plt.subplots_adjust(left=0.06,
                        bottom=0.07,
                        right=0.99,
                        top=0.95,
                        wspace=0.01,
                        hspace=0.01)

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9)
    axlog.grid(axis='y', linestyle='--', alpha=0.9)

    plt.savefig('./logs/Speedup_separate_sc.png')
    plt.show()


def speed_up_separate_cpu():
    performance = read_performance(file='./logs/performance111sc.txt')
    apps = performance.keys()

    # apps = [['minist', 'CFD', 'puremd', 'fluidanimation', 'synthetic'], ['cifar10', 'EMDenoise', 'cosmoflow', 'stemdl', 'DMS', 'slstr', 'optical']]
    apps_shortname = ['Min.','Cif.','Pur.','CFD','Flu.','Cos.','EMD.','DMS','Opt.','Ste.','Sls.','Syn.']

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.7
    # colors = ['#3AA6B9', 'orange', '#FF9EAA',
    #           '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']
    threshold = 1.3

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
    ax.bar(index, pruned_speedup,
           bar_width, label='Pruned', color=colors[2])
    ax.set_ylim(0, threshold)
    # ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(apps_shortname, ha='right', fontsize=BIG_SIZE, fontweight='bold') #  rotation=45,

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=BIG_SIZE)

    # put speedup on the top of the bar
    text_in_log, text_in_linear = [], []
    for i in range(len(apps)):
        if pruned_speedup[i] < threshold:
            text_in_linear.append(i)
        else:
            text_in_log.append(i)

    for j in text_in_linear:
        ax.text(index[j], pruned_speedup[j],
                f'  {pruned_speedup[j]:.2f}x', ha='center', va='bottom',fontsize=BIG_SIZE)  #  rotation=90,

    # plt.subplots_adjust(wspace=0.00001)

    ax.set_ylabel('Speedup x', fontsize=BIG_SIZE, fontweight='bold')
    # fig.suptitle('Speedup by Applications', fontsize=20)
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # # set legend
    # ax.legend(loc='upper left', frameon=False)

    # extend down side to show the x-axis labels
    plt.subplots_adjust(left=0.06,
                        bottom=0.07,
                        right=0.99,
                        top=0.95,
                        wspace=0.01,
                        hspace=0.01)

    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.9)

    plt.savefig('./logs/Speedup_separate_sc.png')
    # plt.show()



def parameter_breakdown_separate():
    performance = read_performance()
    x = performance.keys()

    apps = [['CFD', 'Puremd', 'Fluidanimation', 'EMDenoise', 'Synthetic'], [
        'Minist', 'Cifar10', 'Cosmoflow', 'DMS', 'Slstr'], ['Optical', 'Stemdl']]
    apps_shortname = [['CFD', 'Pur.', 'Flu.', 'EMD.', 'Syn.'], [
        'Min.', 'Cif.', 'Cos.', 'DMS', 'Sls.'], ['Opt.', 'Ste.']]

    fig, ax = plt.subplots(1, 3, figsize=figure_size, gridspec_kw={'width_ratios': [len(apps[0]), len(apps[1]), len(apps[2])],
                                                               'wspace': 0.12})
    bar_width = 0.45
    # colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
    #           '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

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
                           label='Original Conv', color=colors[0],  edgecolor='black')
        rects2 = ax[i].bar(index, original_linear_params, bar_width, bottom=original_conv_params, label='Original Linear', hatch='++',
                           color=colors[0], edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width+0.05, pruned_conv_params, bar_width, label='Pruned Conv', color=colors[1],
                            edgecolor='black')
        rects4 = ax[i].bar(index + bar_width+0.05, pruned_linear_params, bar_width, bottom=pruned_conv_params, hatch='++',
                           label='Pruned Linear', color=colors[1], edgecolor='black')

        ax[i].set_xticks(index + bar_width / 2)
        # ax[i].set_xticklabels(apps[i], rotation=45, ha='right')
        ax[i].set_xticklabels(apps_shortname[i],  ha='center', fontweight='bold')

        # put pruned ratio on the top of the bar
        for j in range(len(apps[i])):
            ax[i].text(index[j] + bar_width + 0.2, pruned_linear_params[j]+pruned_conv_params[j], f'  {prund_ratio[j]*100:.2f}%',
                       ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)

        # add grid
        ax[i].grid(axis='y', linestyle='--', alpha=0.9)

    # set ax[0]'s y-axis scientific notation
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # plt.subplots_adjust(wspace=0.00001)

    # set y-axis log scale
    # ax[2].set_yscale('log')

    ax[0].set_ylabel('Parameters Amount', font1)
    # fig.suptitle('Parameters by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    # ax[1].legend((rects1[0], rects2[0], rects3[0], rects4[0]),
    #              ('Original Conv', 'Original Linear', 'Pruned Conv', 'Pruned Linear'))
    # ax[1].legend(loc="upper left")
    # put the legend on the top of the figure, and use horizontal layout
    # handles, labels = ax[0].get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper center', ncol=4)
    ax[0].legend(bbox_to_anchor=(0, 1.12), loc='upper left', borderaxespad=0, ncol=4, frameon=False,
                 handlelength=1, handletextpad=0.4)

    # extend down side to show the x-axis labels
    plt.subplots_adjust(    left=0.06,
                            bottom=0.07,
                            right=0.99,
                            top=0.9,
                            wspace=0.01,
                            hspace=0.01)

    plt.savefig('./logs/parameter_breakdown_separate.png')
    plt.show()


def flops_breakdown_separate():
    performance = read_performance()
    x = performance.keys()

    apps = [['CFD', 'Puremd', 'Fluidanimation', 'Synthetic','Minist',], [
        'EMDenoise','Cifar10','DMS',  ], ['Cosmoflow', 'Optical', 'Slstr'], ['Stemdl']]
    apps_shortname = [['CFD', 'Pur.', 'Flu.',  'Syn.','Min.'], [ 'EMD.','Cif.','DMS'],['Cos.','Opt.', 'Sls.'], [ 'Ste.']]

    fig, ax = plt.subplots(1, 4, figsize=figure_size, gridspec_kw={'width_ratios': [len(apps[0]), len(apps[1]), len(apps[2]), len(apps[3])],
                                                               'wspace': 0.17})
    bar_width = 0.45
    # colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
    #           '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

    # draw original
    for i in range(4):
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
                           label='Original Conv', color=colors[0], edgecolor='black')
        rects2 = ax[i].bar(index, original_linear_params, bar_width, bottom=original_conv_params, label='Original Linear', hatch='++',
                           color=colors[0], edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width+0.05, pruned_conv_params, bar_width, label='Pruned Conv', color=colors[1],
                            edgecolor='black')
        rects4 = ax[i].bar(index + bar_width+0.05, pruned_linear_params, bar_width, bottom=pruned_conv_params, hatch='++',
                           label='Pruned Linear', color=colors[1], edgecolor='black')

        ax[i].set_xticks(index + bar_width / 2)
        ax[i].set_xticklabels(apps_shortname[i],  ha='center', fontweight='bold')

        # put pruned ratio on the top of the bar
        for j in range(len(apps[i])):
            ax[i].text(index[j] + bar_width + 0.2, pruned_linear_params[j]+pruned_conv_params[j], f'  {prund_ratio[j]*100:.2f}%',
                       ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)

        ax[i].grid(axis='y', linestyle='--', alpha=0.9)

    # set ax[0]'s y-axis scientific notation
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax[1].set_ylim(0, 8e8)
    ax[2].set_ylim(0, 5e10)

    # plt.subplots_adjust(wspace=0.00001)

    # set y-axis log scale
    # ax[2].set_yscale('log')

    ax[0].set_ylabel('FP operations', fontsize=BIG_SIZE, fontweight='bold')
    # fig.suptitle('Flops by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set legend ConvParams and LinearParams
    # handles, labels = ax[0].get_legend_handles_labels()
    # ax[0].legend(handles, labels)

    ax[0].legend(bbox_to_anchor=(0, 1.12), loc='upper left', borderaxespad=0, ncol=4, frameon=False,
                 handlelength=1, handletextpad=0.4)

    # plt.subplots_adjust(top=0.85)

    # extend down side to show the x-axis labels
    plt.subplots_adjust(left=0.07,
                        bottom=0.07,
                        right=0.99,
                        top=0.9,
                        wspace=0.01,
                        hspace=0.01)

    plt.savefig('./logs/flops_breakdown_separate.png')
    plt.show()


def accuracy():
    performance = read_acc(file='./logs/accuracy.txt')
    x = performance.keys()

    apps = [['Minist', 'Cifar10', 'CFD', 'Puremd', 'Fluidanimation', 'DMS',
             'Stemdl', 'Synthetic'], ['Cosmoflow', 'EMDenoise', 'Slstr', 'Optical']]
    apps_shortname = [['Min.','Cif.','CFD', 'Pur.', 'Flu.',  'DMS','Ste.', 'Syn.',],
                      ['Cos.','EMD.', 'Sls.','Opt.']]

    fig, ax = plt.subplots(1, 2, figsize=figure_size, gridspec_kw={
        'width_ratios': [len(apps[0]), len(apps[1])],
        'wspace': 0.2})

    bar_width = 0.45
    # colors = ['#3AA6B9', '#C1ECE4', '#FF9EAA',
    #           '#2D9596', '#FC7300', '#9AD0C2', '#BFDB38']

    for i in range(2):
        index = np.arange(len(apps[i])) * 1.2

        original_acc = [performance[app]['original']['ACC'] for app in apps[i]]
        pruned_acc = [performance[app]['pruned']['ACC'] for app in apps[i]]

        # if acc bigger than 1, then it is percentage, so divide by 100
        original_acc = [oa/100 if oa > 1 else oa for oa in original_acc]
        pruned_acc = [pa/100 if pa > 1 else pa for pa in pruned_acc]

        difference = [pa - oa for oa, pa in zip(original_acc, pruned_acc)]
        diff = [(pa - oa)/oa for oa, pa in zip(original_acc, pruned_acc)]

        rects1 = ax[i].bar(index, original_acc, bar_width, label='Original ACC', color=colors[0],
                           edgecolor='black')

        # draw pruned
        rects3 = ax[i].bar(index + bar_width + 0.05, pruned_acc, bar_width, label='Pruned ACC', color=colors[1],
                           edgecolor='black')

        # put difference on the top of the bar
        # if differrence is positive, color is green, otherwise red
        for j in range(len(apps[i])):
            if i == 0:
                if difference[j] < 0:
                    color = 'g' if i == 0 else 'r'
                    text = f'{diff[j]*100:.2f}'    # *100
                else:
                    color = 'r' if i == 0 else 'g'
                    text = f'+{diff[j]:.2f}'   # *100
                text = text +"%"
            else:
                if difference[j] < 0:
                    color = 'g' if i == 0 else 'r'
                    text = f'{difference[j]:.4f}'    # *100
                else:
                    color = 'r' if i == 0 else 'g'
                    text = f'+{difference[j]:.4f}'   # *100
            ax[i].text(index[j] + bar_width + 0.2, pruned_acc[j],
                       text, ha='center', va='bottom',rotation=90, fontsize=BIG_SIZE, color=color)

        ax[i].set_xticks(index + bar_width / 2)
        ax[i].set_xticklabels(apps_shortname[i], ha='center', fontweight='bold') #  rotation=45,

        ax[i].grid(axis='y', linestyle='--', alpha=0.9)

    # ax[0].set_ylim(0.0, 1.5)
    ax[0].set_ylim(0.0, 1.5)
    ax[0].set_ylabel('higher better', fontsize=BIG_SIZE, fontweight='bold')
    ax[1].set_ylabel('lower better', fontsize=BIG_SIZE, fontweight='bold')
    # fig.suptitle('Quality by Applications')
    # ax.set_xticks(index + bar_width / 2)
    # set x-axis labels rotation

    # set y-axis log scale
    # ax[1].set_ylim(0.0, 0.5)
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.0001, 10)

    # set legend ConvParams and LinearParams
    # ax[4].legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Original Conv', 'Original Linear', 'Pruned Conv', 'Pruned Linear'))
    handles, labels = ax[0].get_legend_handles_labels()
    # ax[0].legend(handles, labels, loc='lower center',
    #            ncol=len(apps[0]), bbox_to_anchor=(0.655, 0.87))
    ax[0].legend(handles, labels, loc='upper left', frameon=False)

    # plt.subplots_adjust(top=0.85, bottom=0.22)
    # left and right side have too much space, so adjust it
    # plt.subplots_adjust(left=0.1, right=0.95)
    plt.subplots_adjust(    left=0.07,
                            bottom=0.07,
                            right=0.99,
                            top=0.95,
                            wspace=0.01,
                            hspace=0.01)

    plt.savefig('./logs/Accuracy.png')
    plt.show()


def peak_memory():
    performance = read_performance(file='./logs/before323-performance.txt')
    apps = performance.keys()
    apps_shortname = ['Min.', 'Cif.', 'Pur.', 'CFD', 'Flu.', 'Cos.', 'EMD.', 'DMS', 'Opt.', 'Ste.', 'Sls.', 'Syn.']

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    bar_width = 0.45
    colors = ['skyblue', 'orange']
    threshold = 1.5

    # draw original
    index = np.arange(len(apps)) * 1.2

    original_memory = [performance[app]['original']['PeakMemory'] for app in apps]
    pruned_memory = [performance[app]['pruned']['PeakMemory'] for app in apps]

    # pruned_ratio = [pm / om for pm, om in zip(pruned_memory, original_memory)]
    difference = [pm - om for pm, om in zip(pruned_memory, original_memory)]

    # linear below 5, log above 5
    ax.bar(index, original_memory, bar_width, label='Original', color=colors[0])
    ax.bar(index + bar_width + 0.05, pruned_memory, bar_width, label='Pruned', color=colors[1])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(apps_shortname, ha='center', fontsize=BIG_SIZE, fontweight='bold')

    # set y ticks' fontsize
    ax.yaxis.set_tick_params(labelsize=14)

    # put the ratio on the pruned bar
    for j in range(len(apps)):
        ax.text(index[j] + bar_width + 0.05, pruned_memory[j], f'{round(difference[j])}MB', color='g',
                ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)

    # plt.subplots_adjust(wspace=0.00001)

    ax.set_ylim(0, 1300)

    ax.set_ylabel('Peak Memory', fontsize=BIG_SIZE, fontweight='bold')

    # ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.9)

    plt.subplots_adjust(left=0.07,
                        bottom=0.01,
                        right=0.99,
                        top=0.95,
                        wspace=0.01,
                        hspace=0.01)

    # set legend
    ax.legend(frameon=False, loc='upper left')

    fig.tight_layout()

    # extend down side to show the x-axis labels
    plt.subplots_adjust(left=0.07,
                        bottom=0.07,
                        right=0.93,
                        top=0.95,
                        wspace=0.01,
                        hspace=0.01)

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
    
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    bar_width = 0.2

    original_flops = [original_flops[layer][0] for layer in original_flops.keys()]
    pruned_flops = [pruned_flops[layer][0] for layer in pruned_flops.keys()]
    
    original_ratio = [original_flops[i] / sum(original_flops) for i in range(len(x_labels))]
    pruned_ratio = [pruned_flops[i] / sum(pruned_flops) for i in range(len(x_labels))]
    
    index = np.arange(len(x_labels)) * 1
    ax.bar(index, original_flops, bar_width, label='Original FLOPS', color=colors[0], edgecolor='black')
    ax.bar(index + bar_width + 0.02, pruned_flops, bar_width, label='Pruned FLOPS', color=colors[0], edgecolor='black', hatch='++')
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_labels, ha='left', fontsize=BIG_SIZE, fontweight='bold')

    ax.set_ylabel('FP operations', fontweight='bold') # , fontsize=BIG_SIZE
    # ax.yaxis.set_tick_params(labelsize=BIG_SIZE)
    ax.set_ylim(0, 4.25e9)

    # draw the second bar chart of parameters use the same x-axis but different y-axis, and put the second y-axis on the right side
    ax2 = ax.twinx()
    original_params = [sum(original_params[layer]) for layer in original_params.keys()]
    pruned_params = [sum(pruned_params[layer]) for layer in pruned_params.keys()]

    original_ratio_p = [original_params[i] / sum(original_params) for i in range(len(x_labels))]
    pruned_ratio_p = [pruned_params[i] / sum(pruned_params) for i in range(len(x_labels))]

    ax2.bar(index + 2*(bar_width + 0.02), original_params, bar_width, label='Original Params', color=colors[1], edgecolor='black')
    ax2.bar(index + 3*(bar_width + 0.02), pruned_params, bar_width, label='Pruned Params', color=colors[1], edgecolor='black', hatch='++')

    # put original and pruned ratio on the top of the bar
    for j in range(len(x_labels)):
        text = ' <1%' if pruned_ratio[j] < 0.01 else f' {pruned_ratio[j] * 100:.0f}%'
        ax.text(index[j]+bar_width+0.025, pruned_flops[j], text, ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)
        text = ' <1%' if original_ratio[j] < 0.01 else f' {original_ratio[j] * 100:.0f}%'
        ax.text(index[j], original_flops[j], text, ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)

    # # put original and pruned ratio on the top of the bar
    for j in range(len(x_labels)):
        text = ' <1%' if original_ratio_p[j] < 0.01 else f' {original_ratio_p[j] * 100:.0f}%'
        ax2.text(index[j] + 2 * (bar_width + 0.025), original_params[j], text, ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)
        text = ' <1%' if pruned_ratio_p[j] < 0.01 else f' {pruned_ratio_p[j] * 100:.0f}%'
        ax2.text(index[j] + 3 * (bar_width + 0.025), pruned_params[j], text, ha='center', va='bottom', fontsize=BIG_SIZE, rotation=90)

    ax2.set_ylabel('Parameters Amount', fontweight='bold') # , fontsize=BIG_SIZE
    # ax2.yaxis.set_tick_params(labelsize=18)
    ax2.set_ylim(1e2, 4e6)

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
    ax.legend(all_handles, all_labels, loc='upper right', fontsize=BIG_SIZE)

    # cut the blank space on the right side
    # plt.subplots_adjust(right=0.95)
    # plt.subplots_adjust(left=0.05)
    # tight_layout()
    fig.tight_layout()

    plt.subplots_adjust(left=0.07,
                        bottom=0.07,
                        right=0.93,
                        top=0.95,
                        wspace=0.01,
                        hspace=0.01)

    ax.grid(axis='y', linestyle='--', alpha=0.9)

    plt.savefig('./logs/layers_flops_breakdown.png')
    plt.show()


def count():
    performance = read_performance()

    apps = ['CFD', 'Puremd', 'Fluidanimation', 'EMDenoise', 'Synthetic',
        'Minist', 'Cifar10', 'Cosmoflow', 'DMS', 'Slstr', 'Optical', 'Stemdl']

    pruned_conv_params = [performance[app]
                          ['pruned']['ConvFlops'] for app in apps]
    pruned_linear_params = [performance[app]
                            ['pruned']['LinearFlops'] for app in apps]
    original_conv_params = [performance[app]
                            ['original']['ConvFlops'] for app in apps]
    original_linear_params = [performance[app]['original']['LinearFlops'] for app in apps]

    linear_reduction_ratio = [(op - pp) / (op+1) for op, pp in zip(original_linear_params, pruned_linear_params)]
    conv_reduction_ratio = [(op - pp) / (op+1) for op, pp in zip(original_conv_params, pruned_conv_params)]

    total_reduction_ratio = [1 - (plp + pcp) / (olp+ocp) for olp, ocp, plp, pcp in zip(original_linear_params, original_conv_params, pruned_linear_params, pruned_conv_params)]

    # print('linear reduction ratio:', np.mean(linear_reduction_ratio))
    # print('conv reduction ratio:', np.mean(conv_reduction_ratio))
    print('linear reduction ratio:', linear_reduction_ratio)
    print('conv reduction ratio:', conv_reduction_ratio)
    print('total reduction ratio:', total_reduction_ratio)
    print('total reduction ratio:', np.mean(total_reduction_ratio))


def read_attention():
    attention_matrix = []
    with open('./logs/attention', 'r') as f:
        for line in f:
            attention_matrix.append([float(x) for x in line.strip().split()])
    return attention_matrix


def attention():
    # read attention matrix from file
    attention_matrix = read_attention()
    # rotate the matrix
    attention_matrix = np.array(attention_matrix).T

    # draw attention matrix
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    im = ax.imshow(attention_matrix, cmap='hot', interpolation='nearest', vmax=0.2, vmin=0)

    # set x-axis labels like x1, x2, x3, ...
    ax.set_xticks(np.arange(attention_matrix.shape[1]))
    # x_labels = ['x'+str(i) for i in range(attention_matrix.shape[1])]
    # ax.set_xticklabels(np.arange(attention_matrix.shape[1]), fontsize=BIG_SIZE, fontweight='bold')
    # set y-axis labels like y1, y2, y3, ...
    ax.set_yticks(np.arange(attention_matrix.shape[0]))
    # ax.set_yticklabels(np.arange(attention_matrix.shape[0]), fontsize=BIG_SIZE, fontweight='bold')

    # set x-axis labels unvisible
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # add bar on the right side
    plt.colorbar(im)

    plt.savefig('./logs/attention.png')


if __name__ == '__main__':
    # performances = read_performance()
    # print(performances)

    # parameter_breakdown_separate()
    # flops_breakdown_separate()
    # speed_up_separate()
    # accuracy()
    #
    # layers_flops_breakdown()
    # peak_memory()
    count()
    # speed_up_separate_gpu_cpu()
    # speed_up_separate_cpu()
    # attention()
