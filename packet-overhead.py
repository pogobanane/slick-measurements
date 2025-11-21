#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List, Any
from plotting import HATCHES as _hatches
from tqdm import tqdm
from functools import reduce
import scipy.stats as scipyst
import operator


# resource on how to do stackplots:
# https://stackoverflow.com/questions/59038979/stacked-bar-chart-in-seaborn

PLOTTING_NAME="microbenchmarks"
DEFAULT_OUTPUT=f"{PLOTTING_NAME}.pdf"

hatches = _hatches.copy()
hatches[0] = _hatches[6]
hatches[6] = _hatches[2]
hatches[2] = _hatches[0]

COLORS = [ str(i) for i in range(20) ]
# COLORS = mcolors.CSS4_COLORS.keys()
# COLORS = [
#     'blue',
#     'cyan',
#     'green',
#     'yellow',
#     'orange',
#     'red',
#     'magenta',
# ]

# hue_map = {
#     '9_vmux-dpdk-e810_hardware': 'vmux-emu (w/ rte_flow)',
#     '9_vmux-med_hardware': 'vmux-med (w/ rte_flow)',
#     '9_vmux-dpdk-e810_software': 'vmux-emu',
#     '9_vmux-med_software': 'vmux-med',
#     '1_vfio_software': 'qemu-pt',
#     '1_vmux-pt_software': 'vmux-pt',
#     '1_vmux-pt_hardware': 'vmux-pt (w/ rte_flow)',
#     '1_vfio_hardware': 'qemu-pt (w/ rte_flow)',
# }

system_map = {
        'ebpf-click-unikraftvm': 'UniBPF',
        'click-unikraftvm': 'Unikraft/Click',
        'click-linuxvm': 'Linux/Click',
        'ebpf-unikraftvm': 'MorphOS',
        'ebpf-linuxvm': 'XDP',
        }

YLABEL = 'Reconfiguration [ms]     '
XLABEL = 'System'

def map_hue(df_hue, hue_map):
    return df_hue.apply(lambda row: hue_map.get(str(row), row))

def log(s: str):
    print(s, flush=True)

def setup_parser():
    parser = argparse.ArgumentParser(
        description=f'Plot {PLOTTING_NAME} graph'
    )

    parser.add_argument('-t',
                        '--title',
                        type=str,
                        help='Title of the plot',
                        )
    parser.add_argument('-W', '--width',
                        type=float,
                        default=12,
                        help='Width of the plot in inches'
                        )
    parser.add_argument('-H', '--height',
                        type=float,
                        default=6,
                        help='Height of the plot in inches'
                        )
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help=f'''Path to the output plot
                             (default: {DEFAULT_OUTPUT})''',
                        default=DEFAULT_OUTPUT
                        )
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )
    parser.add_argument('-c', '--cached',
                        action='store_true',
                        help='Use cached version of parsed data',
                        )
    parser.add_argument('-s', '--slides',
                        action='store_true',
                        help='Use Other setting to plot for presentation slides',
                        )
    for color in COLORS:
        parser.add_argument(f'--{color}',
                            type=argparse.FileType('r'),
                            nargs='+',
                            help=f'''Paths to MoonGen measurement logs for
                                  the {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            help=f'''Name of {color} plot''',
                            )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    if not any([args.__dict__[color] for color in COLORS]):
        parser.error('At least one set of moongen log paths must be ' +
                     'provided')

    return args

# hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O']
hatches_used = 0

# Define a custom function to add hatches to the bar plots
def barplot_with_hatches(*args, **kwargs):
    global hatches_used
    sns.barplot(*args, **kwargs)
    for i, bar in enumerate(plt.gca().patches):
        hatch = hatches[hatches_used % len(hatches)]
        print(hatch)
        bar.set_hatch(hatch)
        hatches_used += 1

def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    # print("\n".join(df.to_string().splitlines()[:20]))
    def parse(df, set_spec, set_calculator, supplementary_df=None):
        rows = []
        ignore_labels = [ "nsec", "label", "Unnamed: 0" ]

        def get_supplementary_data(supplementary_df_, repeating_row_):
            if supplementary_df_ is None:
                return None

            # choose supplementary data for this set
            columns = [ column for column in supplementary_df_.columns if column not in ignore_labels ]
            column_equalities = [ supplementary_df_[column] == repeating_row_[column] for column in columns ]
            supplementary_data = supplementary_df_[reduce(operator.and_, column_equalities)]
            return supplementary_data

        ignore_columns = None
        repeating_row = None
        set_supplement = None
        this_set = set_spec.copy()
        set_nr = 0
        for (i, row) in tqdm(df.iterrows(), total=df.shape[0]):
            # ensure that test inputs of a set are the same
            if repeating_row is None:
                ignore_columns = np.array([ key in ignore_labels for key in row.keys() ])
                repeating_row = row
                set_supplement = get_supplementary_data(supplementary_df, repeating_row)
            elif all(value is not None for value in this_set.values()):
                # arrived at next set
                set_nr += 1
                this_set = set_spec.copy()
                if not all((repeating_row == row) | ignore_columns):
                    # new row is not the same
                    set_supplement = get_supplementary_data(supplementary_df, repeating_row)
                repeating_row = row
            elif not np.all((repeating_row.values == row.values) | ignore_columns):
                raise Exception(f"Different parameters within one set (expected {repeating_row} but got {row})")

            # fill values for this set
            if row['label'] in this_set.keys():
                # if it is already filled, we messed up our set
                if this_set[row['label']] is not None:
                    missing = [ key for key, value in this_set.items() if value is None ]
                    print(f"Duplicate label '{row['label']}' at {i}:\n{df.loc[i-5:i+5].to_string()}")
                    raise Exception(f"Duplicate label {row['label']} in one set. Missing keys for current set: {missing}")
                this_set[row['label']] = row['nsec']
            if all(value is not None for value in this_set.values()):
                # set done, calculate this_set and append to output
                data = set_calculator(set_nr, this_set, set_supplement)
                for key, value in data.items():
                    new_row = repeating_row.copy()
                    new_row['label'] = key
                    new_row['nsec'] = value
                    rows += [ new_row ]

        ret = pd.DataFrame(rows)
        del ret['Unnamed: 0']
        return ret


    log("Parsing linux")
    linux_raw = df[df['system'] == 'linux']
    set_spec = {
        'main': None,
        'init done': None,
        'first packet': None,
        'total startup time': None,
    }
    def calculate_linux(set_nr, i, supp): # takes a set_spec filled with values
        out = dict()
        out['Click init'] = i['init done'] - i['main']
        out['First packet'] = i['first packet'] - i['init done']
        out['Other'] = i['total startup time'] - out['Click init'] + out['First packet']
        out['total'] = i['total startup time']
        return out
    linux = parse(linux_raw, set_spec, calculate_linux)

    log("Parsing uktrace")
    uktrace_raw = df[df['system'] == 'uktrace']
    set_spec = {
        # we collect the same metrics as with uk, but they are inaccurate here because of expensive tracing
        # 'click main()': None,
        # 'print config': None,
        # 'print config done': None,
        # 'initialize elements': None,
        # 'initialize elements done': None,
        # 'first packet': None,
        # 'total startup time': None,
        'qemu start': None,
        'qemu kvm entry': None,
        'qemu kvm port 255': None,
        'qemu kvm port 254': None,
        'qemu kvm port 253': None,
    }
    def calculate_uktrace(set_nr, i, supp): # takes a set_spec filled with values
        out = dict()
        out['Qemu start'] = i['qemu kvm entry'] - i['qemu start']
        out['Firmware'] = i['qemu kvm port 255'] - i['qemu kvm entry']
        out['Unikraft'] = i['qemu kvm port 254'] - i['qemu kvm port 255']
        return out
    uktrace = parse(uktrace_raw, set_spec, calculate_uktrace)
    uktrace['system'] = 'uk' # we've processed the data to make it uk data

    log("Parsing uk")
    uk_raw = df[df['system'] == 'uk']
    set_spec = {
        'click main()': None,
        'print config': None,
        'print config done': None,
        'initialize elements': None,
        'initialize elements done': None,
        'first packet': None,
        'total startup time': None,
    }
    def calculate_uk(set_nr, i, supp):
        def get_supp(label):
            values = supp[supp['label'] == label]
            j = set_nr % values.shape[0]
            return values['nsec'].array[j]
        qemu_start = get_supp('Qemu start')
        firmware = get_supp('Firmware')
        unikraft = get_supp('Unikraft')
        out = dict()
        print_config = i['print config done'] - i['print config']
        out['Click init'] = i['initialize elements'] - i['click main()'] - print_config
        out['VNF configuration'] = i['initialize elements done'] - i['initialize elements']
        out['First packet'] = i['first packet'] - i['initialize elements done']
        # everything we have no detailed trace for is other
        out['Other'] = i['total startup time'] - out['Click init'] - out['VNF configuration'] - out['First packet'] - qemu_start - firmware - unikraft - print_config
        out['total'] = i['total startup time'] - print_config
        return out
    uk = parse(uk_raw, set_spec, calculate_uk, supplementary_df=uktrace)

    log("Parsing ukebpfjit_supp")
    ukebpfjit_supp_raw = df[df['system'] == 'ukebpfjit']
    set_spec = {
        'total': None,
    }
    def calculate_ukebpfjit_supp(set_nr, i, supp):
        out = dict()
        out['total'] = i['total']
        return out
    ukebpfjit_supp = parse(ukebpfjit_supp_raw, set_spec, calculate_ukebpfjit_supp)

    log("Parsing ukebpfjit")
    ukebpfjit_raw = df[(df['system'] == 'ukebpfjit') & (df['label'] != 'total')]
    set_spec = {
        'init ebpf vm': None,
        'read program': None,
        'lock': None,
        'load elf': None,
        'signature': None,
        'jit': None,
        'print': None,
        'init ebpf done': None,
    }
    def calculate_ukebpfjit(set_nr, i, supp):
        if set_nr == 0:
            # first set is from boot which spends another ~5ms on init_ubpf_vm()
            return dict()
        j = set_nr % supp.shape[0]
        total = supp['nsec'].array[j]
        out = dict()

        # out['Read'] = i['read program']
        # out['Lock'] = i['lock']
        out['Load'] = i['read program'] + i['lock'] + i['load elf']
        out['Validate'] = i['signature']
        out['JIT'] = i['jit']
        out['Print'] = i['print']
        out['VNF configuration'] = i['init ebpf done'] - i['init ebpf vm']
        # out['VNF configuration'] = i['init ebpf done'] - i['init ebpf vm']
        # out['VNF configuration > 1'] = i['jit ebpf'] - i['init ebpf vm']
        # out['VNF configuration > 2'] = i['init ebpf done'] - i['jit ebpf']
        out['Other'] = total - (i['init ebpf done'] - i['init ebpf vm']) - out['Print']
        # out['Init uBPF'] = out['VNF configuration'] - out['Read'] - out['Load'] - out['Validate'] - out['JIT'] # never called
        out['total'] = total - out["Print"]
        return out
    ukebpfjit_supp_ = ukebpfjit_supp[ukebpfjit_supp['label'] == 'total']
    ukebpfjit = parse(ukebpfjit_raw, set_spec, calculate_ukebpfjit, supplementary_df=ukebpfjit_supp_)
    # ukebpfjit = pd.concat([ukebpfjit, ukebpfjit_supp])

    df = pd.concat([linux, uk, uktrace, ukebpfjit])
    return df

def main():
    parser = setup_parser()
    args = parse_args(parser)

    fig = plt.figure(figsize=(args.width, args.height))
    # fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    if not args.slides:
        plt.grid()
    # plt.xlim(0, 0.83)
    log_scale = (False, True) if args.logarithmic else False
    # ax.set_yscale('log' if args.logarithmic else 'linear')

    if args.cached and isfile("/tmp/reconfiguration_stack.pkl"):
        log("Using cached data")
        data = pd.read_pickle("/tmp/reconfiguration_stack.pkl")
    else:
        dfs = []
        for color in COLORS:
            if args.__dict__[color]:
                log(f"Reading files for --name-{color}")
                arg_dfs = [ pd.read_csv(f.name) for f in tqdm(args.__dict__[color]) ]
                arg_df = pd.concat(arg_dfs)
                name = args.__dict__[f'{color}_name']
                arg_df["arglabel"] = name
                dfs += [ arg_df ]
                # throughput = ThroughputDatapoint(
                #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
                #     name=args.__dict__[f'{color}_name'],
                #     color=color,
                # )
                # dfs += color_dfs
        df = pd.concat(dfs, ignore_index=True)
        # hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
        # groups = df.groupby(hue)
        # summary = df.groupby(hue)['rxMppsCalc'].describe()
        # df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'fastclick', 'rate']), axis=1)
        # df_hue = map_hue(df_hue, hue_map)
        # df['is_passthrough'] = df.apply(lambda row: True if "vmux-pt" in row['interface'] or "vfio" in row['interface'] else False, axis=1)

        data = parse_data(df)
        data.to_pickle("/tmp/reconfiguration_stack.pkl")
    log("Preparing plotting data")    # map colors to hues
    # colors = sns.color_palette("pastel", len(df['hue'].unique())-1) + [ mcolors.to_rgb('sandybrown') ]
    # palette = dict(zip(df['hue'].unique(), colors))
    columns = ['system', 'Contributor', 'restart_s']
    systems = [ "click-unikraftvm", "click-linuxvm", "ebpf-linuxvm", "ebpf-unikraftvm" ]
    Contributors = [ "Qemu start", "Firmware", "Unikraft", "Click init", "VNF configuration", "First packet", "Other" ]
    data = data[data['label'].isin(Contributors)]
    data['Contributor'] = data['label']
    data['restart_s'] = data['nsec']
    data = data.assign(system = data['system'].map({'linux': 'click-linuxvm', 'uk':'click-unikraftvm', 'ukebpfjit': 'ebpf-unikraftvm'}))
    data = data[['system', 'Contributor', 'restart_s']]
    df = data.groupby(['system', 'Contributor'])['restart_s'].mean().reset_index()
    df['restart_s'] = df['restart_s']/1000000
    # sudo /bin/sh -c 'time ip l set eno1 xdpgeneric obj ./nix/builds/xdp/lib/reflector.o sec xdp'
    xdp_df = pd.DataFrame([['ebpf-linuxvm', 'VNF configuration', 26]], columns=['system', 'Contributor', 'restart_s'])
    df = pd.concat([df, xdp_df])

    df['system'] = df['system'].apply(lambda row: system_map.get(str(row), row))
    df['system'] = pd.Categorical(df['system'], ['Unikraft/Click', 'Linux/Click', 'XDP', 'MorphOS'])
    # Plot using Seaborn
    sns.histplot(
               data=df,
               x='system',
               weights='restart_s',
               hue="Contributor",
               hue_order = ['Qemu start', 'Firmware', 'Unikraft', 'Click init', 'VNF configuration', 'First packet', 'Other'],
               multiple="stack",
               # palette=palette,
               palette="deep",
               edgecolor="dimgray",
               shrink=0.8,
               )

    # sns.add_legend(
    #         # bbox_to_anchor=(0.5, 0.77),
    #         loc='right',
    #         ncol=1, title=None, frameon=False,
    #                 )

    # # Fix the legend hatches
    # for i, legend_patch in enumerate(grid._legend.get_patches()):
    #     hatch = hatches[i % len(hatches)]
    #     legend_patch.set_hatch(f"{hatch}{hatch}")

    # # add hatches to bars
    # for (i, j, k), data in grid.facet_data():
    #     print(i, j, k)
    #     def barplot_add_hatches(plot_in_grid, nr_hues, offset=0):
    #         hatches_used = -1
    #         bars_hatched = 0
    #         for bar in plot_in_grid.patches:
    #             if nr_hues <= 1:
    #                 hatches_used += 1
    #             else: # with multiple hues, we draw bars with the same hatch in batches
    #                 if bars_hatched % nr_hues == 0:
    #                     hatches_used += 1
    #             # if bars_hatched % 7 == 0:
    #             #     hatches_used += 1
    #             bars_hatched += 1
    #             if bar.get_bbox().x0 == 0 and bar.get_bbox().x1 == 0 and bar.get_bbox().y0 == 0 and bar.get_bbox().y1 == 0:
    #                 # skip bars that are not rendered
    #                 continue
    #             hatch = hatches[(offset + hatches_used) % len(hatches)]
    #             print(bar, hatches_used, hatch)
    #             bar.set_hatch(hatch)
    #
    #     if (i, j, k) == (0, 0, 0):
    #         barplot_add_hatches(grid.facet_axis(i, j), 7)
    #     elif (i, j, k) == (0, 1, 0):
    #         barplot_add_hatches(grid.facet_axis(i, j), 1, offset=(7 if not args.slides else 4))

    # def grid_set_titles(grid, titles):
    #     for ax, title in zip(grid.axes.flat, titles):
    #         ax.set_title(title)
    #
    # grid_set_titles(grid, ["Emulation and Mediation", "Passthrough"])
    #
    # grid.figure.set_size_inches(args.width, args.height)
    # grid.set_titles("foobar")
    # plt.subplots_adjust(left=0.06)
    # bar = sns.barplot(x='num_vms', y='rxMppsCalc', hue="hue", data=pd.concat(dfs),
    #             palette='colorblind',
    #             edgecolor='dimgray',
    #             # kind='bar',
    #             # capsize=.05,  # errorbar='sd'
    #             # log_scale=log_scale,
    #             ax=ax,
    #             )
    if (args.slides):
        sns.move_legend(
            ax, "lower center",
            # ncol=2,
            bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
        )
    else:
        sns.move_legend(
            ax, "upper right",
            ncol=2,
            # bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
        )

    color_hatch_map = dict()
    # Fix the legend hatches
    for i, legend_patch in enumerate(ax.get_legend().get_patches()):
        hatch = hatches[i % len(hatches)]
        legend_patch.set_hatch(f"{hatch}{hatch}")
        color_hatch_map[legend_patch.get_facecolor()] = hatch
        print(f"legend {hatch}")

    for bar in ax.patches:
        hatch = color_hatch_map[bar.get_facecolor()]
        bar.set_hatch(hatch)
    #
    # sns.move_legend(
    #     grid, "lower center",
    #     bbox_to_anchor=(0.45, 1),
    #     ncol=1,
    #     title=None,
    #     # frameon=False,
    # )
    # grid.set_xlabels(XLABEL)
    # grid.set_ylabels(YLABEL)
    #
    if (args.slides):
        ax.annotate(
            "↓ Lower is better", # or ↓ ← ↑ →
            xycoords="axes points",
            # xy=(0, 0),
            xy=(0, 0),
            xytext=(-4, -28),
            # fontsize=FONT_SIZE,
            color="navy",
            weight="bold",
        )

    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)

    # plt.ylim(0, 250)
    if not args.logarithmic:
        plt.ylim(bottom=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # # iterate through each container, hatch, and legend handle
    # for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles[::-1]):
    #     # update the hatching in the legend handle
    #     handle.set_hatch(hatch)
    #     # iterate through each rectangle in the container
    #     for rectangle in container:
    #         # set the rectangle hatch
    #         rectangle.set_hatch(hatch)

    # # Loop over the bars
    # for i,thisbar in enumerate(bar.patches):
    #     # Set a different hatch for each bar
    #     thisbar.set_hatch(hatches[i % len(hatches)])

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    # fig.tight_layout(rect = (0, 0, 0, 0.1))
    # ax.set_position((0.1, 0.1, 0.5, 0.8))
    plt.tight_layout(pad=0.1)
    # plt.subplots_adjust(right=0.78)
    plt.savefig(args.output.name)
    plt.close()





if __name__ == '__main__':
    main()
