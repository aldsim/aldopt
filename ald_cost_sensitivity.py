import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ALDmodel import ALDGrowth
from physicsmodel import ALDOpt
from core import plot_2d, plot_uq, WP
from functools import partial
from scipy.stats import halfcauchy, triang
from sopt import sbostep


def ALDinitialize(system='Al2O3-200C', noise=0.01):

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


def ALDsample(x, bnds=None, apf=None, output=True,
              nmult=2, nrep=10, stdgrowth=None,
              noise=0.1, limited=True, wrtfile='log.txt'):

    x = 10**np.array(x)

    bm = 10**bnds[:, 1]
    tmax = np.sum(bm)

    st = apf.cycle(bm[0], bm[1], bm[2], bm[3], 1)  # throw-away cycle

    if stdgrowth is None:
        stdgrowth = st[0]

    xp = x + np.ones((4,))
    g = apf.cycle(x[0], x[1], x[2], x[3], nrep + 1)
    gp = apf.cycle(xp[0], xp[1], xp[2], xp[3], nrep + 1)
    
    gmaxL = [st[0]]
    g0L = [g[0]]
    for ii in range(nrep - 1):
        gmaxL += [apf.cycle(bm[0], bm[1], bm[2], bm[3], 1)[0]]
        g0L += [apf.cycle(x[0], x[1], x[2], x[3], 1)[0]]
    gmax = np.mean(gmaxL)
    g0 = np.mean(g0L)

    gm = np.mean(g[1:])
    gpm = np.mean(gp[1:])

    if output:
        WP('x: ' + str(x), wrtfile)
        WP('g: ' + str(np.round(g, 3)), wrtfile)
        WP('gp: ' + str(np.round(gp, 3)), wrtfile)
        WP('g0L:' + str(np.round(g0L, 3)), wrtfile)

    vc1 = (np.abs(gm - gpm)/noise)  # - nmult
    vc2 = (np.abs(gm - g0)/noise)  # - nmult

    if limited:
        if vc1 < nmult:
            vc1 = nmult
        if vc2 < nmult:
            vc2 = nmult

    gc = (stdgrowth - gm)**2
    tc = np.sum(x)/tmax

    y = gc + vc1 + vc2 + tc

    if output:
        WP('g, gc, vc1, vc2, tc, y: ' + \
           str(np.round(g[-1], 4)) + ' ' + \
           str(np.round(gc, 4)) + ' ' + \
           str(np.round(vc1, 4)) + ' ' + \
           str(np.round(vc2, 4)) + ' ' + \
           str(np.round(tc, 4)) + ' ' + \
           str(np.round(y, 4)), wrtfile)

    return np.log10(y), g[-1], gc, vc1, vc2, tc


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    system = 'Al2O3-200C'
    noise_imposed = 0.001
    output = False
    nmult = 1.0
    nrep = 5
    ntrial = 20
    ntimes = 10
    ndim = 4
    lwr = 0.2
    fileid = system + '_' + str(noise_imposed)
    wrtfile = fileid + '.txt'


    apf = ALDinitialize(system, noise_imposed)
    func = partial(ALDsample, apf=apf)

    """for ALD0"""
    uprL = [.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    for upr in uprL:

        bnds = upr*np.ones((ndim, 2)).astype('float')
        bnds[:, 0] = lwr
        bnds = np.log10(bnds)

        bm = 10**bnds[:, 1]
        st = apf.cycle(bm[0], bm[1], bm[2], bm[3], 100)
        noise = np.std(st[5:])
        stdgrowth = np.mean(st[5:])
        WP('bound: ' + str(upr) + ', noise: ' + str(noise), wrtfile)

        res = func(bnds[:, 1], bnds, output=True,
                   nmult=nmult, stdgrowth=stdgrowth, noise=noise,
                   wrtfile=wrtfile)
        if res[3] == nmult and res[4] == nmult:
            WP('final upper bound: ' + str(upr), wrtfile)
            break

    timeL = (np.log10(upr)-np.log10(lwr))*(np.linspace(0, 1, ntimes)) + \
        np.log10(lwr)
    x_base = np.log10(np.array([2.0, 2.0, 2.0, 2.0]))

    noiseiL = [0.001, 0.003, 0.01, 0.03, 0.1]

    nnoise = len(noiseiL)

    vc1a = np.zeros((nnoise, ntimes, ntrial))
    vc2a = np.zeros((nnoise, ntimes, ntrial))

    for ii in range(nnoise):

        noise = noiseiL[ii]
        apf = ALDinitialize(system, noiseiL[ii])
        func = partial(ALDsample, apf=apf)

        for jj in range(ntimes): 
            for kk in range(ntrial):
                x = np.copy(x_base)
                x[3] = timeL[jj]

                res = ALDsample(x, bnds=bnds, apf=apf, output=output,
                    nmult=nmult, nrep=nrep, stdgrowth=stdgrowth,
                    noise=noise, limited=False, wrtfile=wrtfile)
                vc1a[ii, jj, kk] = res[3]
                vc2a[ii, jj, kk] = res[4]

    vc1f = np.mean(vc1a <= 1.0, 2)
    vc2f = np.mean(vc2a <= 1.0, 2)

    sns.set()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    cL = sns.color_palette('cubehelix', nnoise + 2)[1:-1]

    pcntL = [34, 50, 66]

    for ii, noisei, c in zip(range(nnoise), noiseiL, cL):
        plt.figure(num='vc1')
        low, mid, high = np.percentile(vc1a[ii, ...], pcntL, axis=1)
        plt.fill_between(10**timeL, low, high,
            facecolor=c, alpha=0.3)
        plt.plot(10**timeL, mid, color=c, alpha=0.9, ls='-',
                 marker='', label=r'$\sigma^{meas.} =$' + str(noisei))

        plt.figure(num='vc2')
        low, mid, high = np.percentile(vc2a[ii, ...], pcntL, axis=1)
        plt.fill_between(10**timeL, low, high,
            facecolor=c, alpha=0.3)
        plt.plot(10**timeL, mid, color=c, alpha=0.9, ls='-',
                 marker='', label=r'$\sigma^{meas.} =$ ' + str(noisei))

    plt.figure(num='vc1', figsize=(5, 4))
    plt.hlines(1, lwr, upr, colors='k', label=r'$c^{noise}$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$C_{var1}$', fontsize=14)
    plt.yscale('log')
    plt.xlim(lwr, upr)
    plt.legend()
    plt.tight_layout()

    plt.figure(num='vc2', figsize=(5, 4))
    plt.hlines(1, lwr, upr, colors='k', label=r'$c^{noise}$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$C_{var2}$', fontsize=14)
    plt.yscale('log')
    plt.xlim(lwr, upr)
    plt.legend()
    plt.tight_layout()

    plt.show()

