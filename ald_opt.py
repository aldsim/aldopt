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
              nmult=1, nrep=10, stdgrowth=None,
              noise=0.1, wrtfile='log.txt'):

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

    vc1 = (np.abs(gm - gpm)/noise) - nmult
    vc2 = (np.abs(gm - g0)/noise) - nmult

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


def soptimize(func, apf, bnds, stset, acq='EI', model_type='GP',
              niter=30, plot_freq=1, output=False,
              loocv=False, nmult=2, nrep=10, stdgrowth=None, noise=0.1,
              wrtfile='log.txt'):

    # number of cycles in cost function evaluation
    # cyccf = 1 + 2*(nrep+1) + 2*(nrep-1)
    cyccf = 1 + 2*(5+1) + 2*(5-1)

    # total number of cycles for non-physics based methods
    cyct = cyccf*niter

    if acq == 'Random':
        randA = np.zeros((10000, bnds.shape[0]))
        for ii in range(bnds.shape[0]):
            randA[:, ii] = np.random.uniform(
                bnds[ii, 0], bnds[ii, 1], size=(10000))

    if acq == 'Expert system':
        x_ = 10**bnds[:, 1]

        WP('starting guess: ' + str(x_), wrtfile)

        nitph = np.floor(cyct/nrep).astype('int')

        opt = ALDOpt(Nav=nrep, Nmax=nitph, tmin=bnds[0, 0])
        opt.optimize(apf, x_)
        phyres = np.array(opt.history)

        phycycmax = nrep*phyres.shape[0]

        cycL = np.arange(0, cyct, nrep)

    else:
        cycL = np.arange(0, cyct, cyccf)
        c = np.linspace(1, 0, niter)

    XL, resL, bestL = [], [], []
    best = 1e10
    for ii, cyc in zip(range(len(cycL)), cycL):

        if np.mod(ii+1, plot_freq) == 0 and output and ii >= len(stset):
            plotting = True
        else:
            plotting = False

        if acq == 'Random':
            x = list(randA[ii, :])
        elif acq == 'Expert system':
            if ii < phyres.shape[0]:
                x = np.log10(phyres[ii, :4])
        elif ii < stset.shape[0]:
            x = list(stset[ii, :])
        else:
            x = sbostep(XL, np.array(resL), bnds, acq,
                        model_type, output, plotting, loocv)

        res = np.atleast_1d(func(
            x, bnds, output=output, nmult=nmult, nrep=nrep,
            stdgrowth=stdgrowth, noise=noise, wrtfile=wrtfile))
        XL += [x]
        resL += [res]
        if res[0] < best:
            best = res[0]
        bestL += [best]

        if output:
            WP('iteration ' + str(ii) + ' complete\n', wrtfile)

        if plotting:
            plt.show()

    return cycL, XL, resL, bestL


def lhs_design(npt, ndim, lwr, upr, output=True, wrtfile='log.txt'):
    fname = 'maximin_lhs_l2_' + str(npt) + '_' + str(ndim) + 'd.csv'
    stset = np.loadtxt(fname, delimiter=',')
    stset = (np.log10(upr)-np.log10(lwr))*(stset/np.int16(npt-1)) + np.log10(lwr)
    if output:
        WP(str(10**stset), wrtfile)
    return stset


def main(system='Al2O3-200C', noise_imposed=0.001, acqL=['Random'], ninit=20,
         niter=40, nopt=10, ndim=4, nmult=0.5, nrep=5, lwr=0.2,
         model_type='GP', loocv=False, plot_freq=100, output=False):

    fileid = system + '_' + str(noise_imposed)
    wrtfile = fileid + '.txt'

    apf = ALDinitialize(system, noise_imposed)
    func = partial(ALDsample, apf=apf)
    optval = None

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

    """load the LHS design for ninit points"""
    stset = lhs_design(npt=ninit, ndim=ndim,
                       lwr=lwr, upr=upr, wrtfile=wrtfile)

    sns.set()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    cm = sns.color_palette('mako', 3)

    """plot the 2d objective field"""
    if output and ndim == 2:
        plot_2d(func, bnds)

    figbv = 'best value'
    plt.figure(figbv)

    figg = 'growth'
    plt.figure(figg)

    d = {
        'acquisition function':[], 'trial number':[],'dose 1 (s)':[],
        'purge 1 (s)':[], 'dose 2 (s)':[], 'purge 2 (s)':[],
        'growth':[], 'total cost':[], 'growth cost':[],
        'variability cost 1':[], 'variability cost 2':[],
        'time cost':[]}

    for c, acq in zip(cm, acqL):
        bestA = []
        gA = []
        for jj in range(nopt):
            if acq == 'Expert system':
                nrep = 10
            else:
                nrep = 5

            res = soptimize(
                func, apf, bnds, stset,
                acq=acq, model_type=model_type, niter=niter,
                plot_freq=plot_freq, output=output, loocv=loocv,
                nmult=nmult, nrep=nrep, stdgrowth=stdgrowth,
                noise=noise)
            cycL, XL, resL, cL = res
            resL = np.array(resL)

            bestA.append(cL)
            gA.append(resL[:, 1])

            bindx = np.argmin(resL[:, 0])
            Xlbest = np.round(10**np.array(XL[bindx]), 3)

            d['acquisition function'] += [acq]
            d['trial number'] += [jj]
            d['dose 1 (s)'] += [Xlbest[0]]
            d['purge 1 (s)'] += [Xlbest[1]]
            d['dose 2 (s)'] += [Xlbest[2]]
            d['purge 2 (s)'] += [Xlbest[3]]
            d['growth'] += [resL[bindx, 1]]
            d['total cost'] += [resL[bindx, 0]]
            d['growth cost'] += [resL[bindx, 2]]
            d['variability cost 1'] += [resL[bindx, 3]]
            d['variability cost 2'] += [resL[bindx, 4]]
            d['time cost'] += [resL[bindx, 5]]

            print(acq, jj, 'best value:', np.round(resL[bindx, 0], 4))
            print(acq, jj, 'best indv.:', Xlbest)
            print(acq, jj, 'growth:', np.round(resL[bindx, 1], 4),
                           'growth cost:', np.round(resL[bindx, 2], 4),
                           'variability cost 1:', np.round(resL[bindx, 3], 4),
                           'variability cost 2:', np.round(resL[bindx, 4], 4),
                           'time cost:', np.round(resL[bindx, 5], 4))

        plot_uq(cycL, bestA, c, acq, optval=None, optgap=None,
                fignum=figbv)

        plot_uq(cycL, gA, c, acq, optval=None, optgap=None,
                fignum=figg)

    plt.figure(num=figbv, figsize=(5, 4))
    plt.xlabel('ALD Cycles', fontsize=12)
    plt.ylabel(r'$C_{total}$ best value', fontsize=12)
    plt.xlim(0, 800)
    plt.ylim(0.3, 1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(fileid + '.png')
    plt.close()

    plt.figure(num=figg, figsize=(5, 4))
    plt.xlabel('ALD cycles', fontsize=12)
    plt.ylabel('Normalized growth per cycle', fontsize=12)
    plt.xlim(0, 800)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(fileid + '_growth.png')
    plt.close()

    df = pd.DataFrame(d)
    df.to_csv(fileid + '.csv')


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    acqL = ['Random', 'Bayesian optimization', 'Expert system']
    systemL = ['Al2O3-200C', 'Al2O3-100C', 'TiO2-200C', 'W-200C']
    noiseL = [0.001, 0.00316, 0.01, 0.0316, 0.1]
    theadL = ['dose 1 (s)', 'purge 1 (s)', 'dose 2 (s)', 'purge 2 (s)']
    chemAL = ['TMA', 'TMA', 'TTIP', '$\mathregular{WF_6}$']
    chemBL = ['$\mathregular{H_2O}$', '$\mathregular{H_2O}$',
              '$\mathregular{H_2O}$', '$\mathregular{Si_2H_6}$']

    cm = sns.color_palette('mako', 3)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    for system, chemA, chemB, in zip(systemL, chemAL, chemBL):

        d = {}
        d['ref'] = np.zeros((len(noiseL), 4))

        for acq in acqL:
            shape = (len(noiseL), 4)
            d[acq + '_l'] = np.zeros((len(noiseL), 4))
            d[acq + '_m'] = np.zeros((len(noiseL), 4))
            d[acq + '_u'] = np.zeros((len(noiseL), 4))
        for ii, noise_imposed in enumerate(noiseL):

            main(system=system, noise_imposed=noise_imposed,
                 acqL=acqL, ninit=10, niter=40, nopt=10, ndim=4, nmult=1.0,
                 nrep=10, lwr=0.2, model_type='GP', loocv=False, plot_freq=100,
                 output=False)

            df = pd.read_csv(system + '_' + str(noise_imposed) + '.csv')
            df_uptake = pd.read_csv(system + '_uptake.csv')

            timesref = df_uptake[str(noise_imposed)].values
            timesref[timesref < 0.2] = 0.2
            d['ref'][ii, :] = timesref

            for acq in acqL:
                sel = df['acquisition function'] == acq
                rawdat = df[sel][theadL].values

                l, m, h = np.percentile(rawdat, [2.5, 50, 97.5], axis=0)
                d[acq + '_l'][ii, :] = l
                d[acq + '_m'][ii, :] = m
                d[acq + '_u'][ii, :] = h

        timelabelL = [r'{} dose time (s)'.format(chemA),
                      r'{} purge time (s)'.format(chemA),
                      r'{} dose time (s)'.format(chemB),
                      r'{} purge time (s)'.format(chemB)]

        for jj, thead, timelabel in zip(range(len(theadL)), theadL, timelabelL):
            plt.figure(figsize=(5, 4))
            for acq, c in zip(acqL, cm):
                plt.plot(noiseL, d[acq + '_m'][:, jj],
                         ls='-', alpha=.9, color=c, label=acq)
                plt.plot(noiseL, d[acq + '_l'][:, jj],
                         ls=':', alpha=.9, color=c)
                plt.plot(noiseL, d[acq + '_u'][:, jj],
                         ls=':', alpha=.9, color=c)
                plt.fill_between(
                    noiseL, d[acq + '_l'][:, jj], d[acq + '_u'][:, jj],
                    alpha=0.15, facecolor=c)
            plt.plot(noiseL, d['ref'][:, jj],
                     ls='-', alpha=0.9, color='k', label='optimal')
            plt.xscale('log')
            plt.xlim([0.001, 0.1])
            plt.grid(which='minor', axis='x')
            plt.xlabel(r'$c^{noise}$', fontsize=12)
            plt.ylabel(timelabel, fontsize=12)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(system + '_' + thead + '_pcterr.png')
            plt.close()
