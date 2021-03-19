import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ALDmodel import ALDGrowth
from core import plot_2d, plot_uq
from functools import partial
from scipy.stats import halfcauchy, triang
from sopt import sbostep


def ALDinitialize(system='Al2O3-200C', noise=0.0):

    # chem = (p, M, beta, tp)
    # chem is the tuple of chemical parameters
    # p: precursor pressure (Pa)
    # M: molecular mass (atomic mass units)
    # beta: sticking probability
    # tp: characteristic time of precursor evacuation

    if system == 'Al2O3-200C':
        chem1 = (26.66, 72, 1e-3, .2, 1.0)
        chem2 = (26.66, 18, 1e-4, .2, 0.0)
        T = 473  # temperature in K
        sitearea = 0.225e-18  # area of a surface site, in m^2

    elif system == 'Al2O3-100C':
        chem1 = (26.66, 72, 1e-4, 3, 1.0)
        chem2 = (26.66, 18, 1e-5, 10, 0.0)
        T = 373  # temperature in K
        sitearea = 0.251e-18  # area of a surface site, in m^2

    elif system == 'TiO2-200C':
        chem1 = (0.6665, 284, 1e-4, .2, 1.0)
        chem2 = (26.66, 18, 1e-4, .2, 0.0)
        T = 473  # temperature in K
        sitearea = 1.17e-18  # area of a surface site, in m^2

    elif system == 'W-200C':
        chem1 = (6.665, 297, 0.2, .2, 1.0)
        chem2 = (26.66, 62, 0.05, .2, 0.0)
        T = 473  # temperature in K
        sitearea = 0.036e-18  # area of a surface site, in m^2

    apf = ALDGrowth(T, sitearea, chem1, chem2, noise)

    return apf


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    sns.set_style('whitegrid') 
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    nt = 100

    systemL = ['Al2O3-200C', 'Al2O3-100C', 'TiO2-200C', 'W-200C']
    chemAL = ['TMA', 'TMA', 'TTIP', '$\mathregular{WF_6}$']
    chemBL = ['$\mathregular{H_2O}$', '$\mathregular{H_2O}$',
              '$\mathregular{H_2O}$', '$\mathregular{Si_2H_6}$']
    bndsL = [
        np.array([0.2, 4, 1, 3]),
        np.array([7, 60, 6, 180]),
        np.array([24, 2, 0.2, 3]),
        np.array([0.04, 4, 0.02, 4]),
        ]

    cL = sns.color_palette()[:4]
    mL = ['o', 's', 'd', '^']

    for system, chemA, chemB, bnds in zip(systemL, chemAL, chemBL, bndsL):

        print('system:', system)

        apf = ALDinitialize(system)

        chemL = ['t1', 't2', 't3', 't4']

        tLL = []
        labelL = []
        for bnd, chem, ii in zip(bnds, chemL, range(4)):
            tL = np.linspace(0, bnd, nt)
            tLL += [tL]

            blab = bnds.astype('str')
            blab[ii] = r'$t_{}$'.format(ii+1)
            label = [blab[0], 's, ', blab[1], 's, ',
                     blab[2], 's, ', blab[3], 's']
            labelL += ["".join(label)]

        zips = zip(range(4), tLL, cL, bnds, labelL)

        plt.figure(num=system + '_uptake-trad', figsize=(5, 4))

        opttimes = np.zeros((4, 5))
        for ii, tL, c, bnd, label in zips:
            gL = np.array([])
            for t in tL:
                bm = bnds.astype('float32')
                bm[ii] = t
                g = apf.cycle(bm[0], bm[1], bm[2], bm[3], 3)[-1]
                gL = np.append(gL, g)

            plt.plot(tL, gL, color=c, marker='', ls='-',
                alpha=0.9, label=label)

        plt.hlines(1, -0.02*np.max(bnds), 0.6*np.max(bnds),
                   colors='k', linestyles='-', label='stable growth per cycle', zorder=10)
        plt.xlabel(r'$t_i$ (s)', fontsize=12)
        plt.ylabel('Normalized growth per cycle', fontsize=12)
        plt.xlim([-0.02*np.max(bnds), 0.6*np.max(bnds)])
        plt.ylim([.95, 1.05])
        plt.legend()
        plt.tight_layout()
        plt.savefig(system + '_uptake-trad.png')

    plt.show()