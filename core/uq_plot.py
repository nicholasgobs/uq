import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from tqdm import tqdm

def set_plot_text():
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}

    plt.rc('font', **font)

    params = {'legend.fontsize': 'large',
              'figure.figsize': (10, 7),
              'axes.labelsize': 'large',
              'axes.linewidth': 2,
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    plt.rcParams.update(params)
    return


def plot_results(output_files, output_description, nbins=20):
    set_plot_text()

    for file, description in zip(output_files, output_description):
        data = np.loadtxt(f"{file}.txt")
        f = plt.figure()
        plt.hist(data, bins=nbins, linewidth=3, color='white', ec='darkblue', histtype='step')
        plt.xlabel(description, fontweight='bold')
        plt.ylabel('count', fontweight='bold')
        plt.title(file, fontweight='bold')
        #f.savefig(file + '.png', bbox_inches='tight')
        
        plt.show()

        plt.close(f)


def plot_normal_distribution():
    set_plot_text()
    scipy.stats.norm.pdf(0,0,0.08)
    x = np.linspace(-4*0.08, 4*0.08, 100)

    f = plt.figure()
    plt.plot(x, scipy.stats.norm.pdf(x, 0, 0.08), color='darkblue', linewidth=3)
    plt.ylim([0, 5])
    plt.grid()
    f.savefig('distribution.png', bbox_inches='tight')
    plt.close(f)
