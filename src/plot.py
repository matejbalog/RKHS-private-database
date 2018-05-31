import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pylab as plt

from matplotlib.ticker import FuncFormatter

from utils import json_load


def plot_experiment(path_alg1=None, path_alg2=None, path_save=None):
    # Load data
    M_max = 0
    if path_alg1 is not None:
        j1 = json_load(path_alg1)
        algorithm, dataset, public_data, N_private, D, delta = j1['algorithm'], j1['dataset'], j1['public_data'], j1['N_private'], j1['D'], j1['delta']
        assert algorithm == 1
        epsilons, Ms_alg1 = j1['epsilons'], j1['Ns_public']
        M_max = max(M_max, Ms_alg1[-1])
    if path_alg2 is not None:
        j2 = json_load(path_alg2)
        algorithm, dataset, public_data, N_private, D, delta = j2['algorithm'], j2['dataset'], j2['public_data'], j2['N_private'], j2['D'], j2['delta']
        assert algorithm == 2
        epsilons, Ms_alg2 = j2['epsilons'], j2['Ms_report']
        M_max = max(M_max, Ms_alg2[-1])

    # Check consistency if plotting both experiments
    if (path_alg1 is not None) and (path_alg2 is not None):
        for key in ['dataset', 'public_data', 'N_private', 'D', 'epsilons', 'delta']:
            assert j1[key] == j2[key]

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.text(0.5, 0.95, '$D = %d$' % (D), transform = ax.transAxes)

    # x-axis
    ax.set_xlim((0.6, M_max))
    ax.set_xscale('log')
    if public_data == 'leak':
        ax.set_xlabel('$M$ (number of public data points)')
    elif public_data == 'random':
        ax.set_xlabel('$M$ (number of synthetic data points)')

    # y-axis
    ax.set_ylim((5*1e-5, 4.0))
    ax.set_yscale('log')
    ax.set_ylabel('RKHS distance $\|\| \hat{\mu}_X - \cdot \|\|_{\mathcal{H}}$')
    
    # Fix colors for different epsilon curves
    colors = [tableau20(2*ei) for ei, _ in enumerate(epsilons)]

    # --- Plot Algorithm 1 ---
    if path_alg1 is not None:

        # Plot RKHS distances incurred due to projection
        dists = j1['dists_proj']
        ax.plot(Ms_alg1, dists, color='gray', ls='dotted', label='projection distance')

        # Plot curves for different epsilons
        for ei, epsilon in enumerate(epsilons):
            dists = j1['dists_alg1'][ei]
            label = 'Algorithm 1, $\\varepsilon = %s$' % (epsilon)
            ax.plot(Ms_alg1, dists, color=colors[ei], ls='solid', label=label)

        # Experiment leaks: plot uniform baseline distance (private KME <-> public KME)
        if public_data == 'leak':
            dists = j1['dists_base']
            label = 'baseline (uniform weights)'
            ax.plot(Ms_alg1, dists, color='black', ls='dashed', label=label)

    # --- Plot Algorithm 2 ---
    if path_alg2 is not None:

        # Plot as curve with Alg1/random experiment
        if public_data == 'random':
            for ei, epsilon in enumerate(epsilons):
                dists = j2['dists_alg2'][ei]
                label = 'Algorithm 2, $\\varepsilon = %s$' % (epsilon)
                ax.plot(Ms_alg2, dists, color=colors[ei], ls='dashed', label=label)

    # Experiment leaks: Add percentages to x-ticks
    if public_data == 'leak':
        def format_fn(tick_val, tick_pos):
            return '%d\n(%s%%)' % (int(tick_val), 100.0 * tick_val / N_private)
        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

    # Plot legend
    ax.legend(loc='lower left')

    # Save plot
    if path_save is not None:
        save_plot(fig, path_save)


# --- PLOTTING UTILITIES ---
def save_plot(fig, savepath, bbox_extra_artists=None):
    fig.savefig(savepath + '.pdf', bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.savefig(savepath + '.png', bbox_extra_artists=bbox_extra_artists, bbox_inches='tight', format='png')
    print('[OK] Saved plot to %s.{pdf, png}' % (savepath))

def tableau20(k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20[k]


# --- REPLOT FIGURES ---
def main(args_dict):
    plot_alg1 = args_dict['alg1']
    plot_alg2 = args_dict['alg2']
    path_save = args_dict['path_save']
    plot_experiment(plot_alg1, plot_alg2, path_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg1', default=None, type=str, help='file to plot Algorithm 1 results from')
    parser.add_argument('--alg2', default=None, type=str, help='file to plot Algorithm 2 results from')
    parser.add_argument('--path_save', default=None, type=str, help='where to save the resulting plot')
    args_dict = vars(parser.parse_args())
    main(args_dict)
