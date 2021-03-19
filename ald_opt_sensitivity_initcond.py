import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ALDmodel import ALDGrowth
from physicsmodel import ALDOpt
from core import plot_2d, plot_uq
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
              nmult=2, nrep=10, stdgrowth=None, noise=0.1):

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
        print('x:', x)
        print('g:', np.round(g, 3))
        print('gp:', np.round(gp, 3))
        print('g0L:', np.round(g0L, 3))

    vc1 = (np.abs(gm - gpm)/noise)
    vc2 = (np.abs(gm - g0)/noise)

    if vc1 < nmult:
        vc1 = nmult
    if vc2 < nmult:
        vc2 = nmult

    gc = (stdgrowth - gm)**2
    tc = np.sum(x)/tmax

    y = gc + vc1 + vc2 + tc

    if output:
        print('g, gc, vc1, vc2, tc, y:',
              np.round(g[-1], 4),
              np.round(gc, 4), np.round(vc1, 4), np.round(vc2, 4),
              np.round(tc, 4), np.round(y, 4))

    return np.log10(y), g[-1], gc, vc1, vc2, tc


def soptimize(func, apf, bnds, stset, acq='EI', model_type='GP',
              niter=30, plot_freq=1, output=False,
              loocv=False, nmult=2, nrep=10, stdgrowth=None,
              noise=0.1, stpt=1.0):

    # number of cycles in cost function evaluation
    cyccf = 1 + 2*(nrep+1) + 2*(nrep-1)

    # total number of cycles for non-physics based methods
    cyct = cyccf*niter

    if acq == 'random':
        randA = np.zeros((10000, bnds.shape[0]))
        for ii in range(bnds.shape[0]):
            randA[:, ii] = np.random.uniform(
                bnds[ii, 0], bnds[ii, 1], size=(10000))

    if acq == 'phys':
        x_ = stpt*(10**bnds[:, 1])
        print(x_)

        nitph = np.floor(cyct/nrep).astype('int')

        opt = ALDOpt(Nav=nrep, Nmax=nitph)
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

        if acq == 'random':
            x = list(randA[ii, :])
        elif acq == 'phys':
            if ii < phyres.shape[0]:
                x = np.log10(phyres[ii, :4])
        elif ii < stset.shape[0]:
            x = list(stset[ii, :])
        else:
            kappa = triang.rvs(c[ii])
            if output:
                print('c, kappa:', c[ii], kappa)
            x = sbostep(XL, np.array(resL), bnds, acq,
                        model_type, output, plotting, loocv, kappa)

        res = np.atleast_1d(func(
            x, bnds, output=output, nmult=nmult, nrep=nrep,
            stdgrowth=stdgrowth, noise=noise))
        XL += [x]
        resL += [res]
        if res[0] < best:
            best = res[0]
        bestL += [best]

        if output:
            print('iteration', ii, 'complete\n')

        if plotting:
            plt.show()

    return cycL, XL, resL, bestL


def lhs_design(npt, ndim, lwr, upr, output=True):
    fname = 'maximin_lhs_l2_' + str(npt) + '_' + str(ndim) + 'd.csv'
    stset = np.loadtxt(fname, delimiter=',')
    stset = (np.log10(upr)-np.log10(lwr))*(stset/np.int16(ninit-1)) + np.log10(lwr)
    if output:
        print(10**stset)
    return stset


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    ninit = 20
    niter = 40
    nopt = 10
    ndim = 4
    nmult = 1.0
    nrep = 5
    output = False
    plot_freq = 100
    acq = 'phys'
    stptL = [0.25, 0.5, 0.75, 1.0]
    model_type = 'GP'
    loocv = False
    lwr = 0.2
    system = 'Al2O3-200C'
    noise_imposed = 0.1

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
        print('bound:', upr, ', noise:', noise)

        res = func(bnds[:, 1], bnds, output=True,
                   nmult=nmult, stdgrowth=stdgrowth, noise=noise)
        if res[3] == nmult and res[4] == nmult:
            print('final upper bound:', upr)
            break

    """load the LHS design for ninit points"""
    stset = lhs_design(ninit, ndim, lwr, upr)

    sns.set()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    cm = sns.color_palette('cubehelix', len(stptL)+2)[1:-1]

    """plot the 2d objective field"""
    if output and ndim == 2:
        plot_2d(func, bnds)

    figbv = 'best value'
    plt.figure(figbv)

    figc = 'cost components'
    plt.figure(figc)

    for c, stpt  in zip(cm, stptL):
        bestA = []
        for jj in range(nopt):
            res = soptimize(
                func, apf, bnds, stset,
                acq=acq, model_type=model_type, niter=niter,
                plot_freq=plot_freq, output=output, loocv=loocv,
                nmult=nmult, nrep=nrep, stdgrowth=stdgrowth,
                noise=noise, stpt=stpt)
            cycL, XL, resL, cL = res
            bestA.append(cL)
            resL = np.array(resL)
            bindx = np.argmin(resL[:, 0])
            Xlbest = np.round(10**np.array(XL[bindx]), 3)
            print(acq, jj, 'best value:', np.round(resL[bindx, 0], 4))
            print(acq, jj, 'best indv.:', Xlbest)
            print(acq, jj, 'growth:', np.round(resL[bindx, 1], 4),
                           'growth cost:', np.round(resL[bindx, 2], 4),
                           'variability cost 1:', np.round(resL[bindx, 3], 4),
                           'variability cost 2:', np.round(resL[bindx, 4], 4),
                           'time cost:', np.round(resL[bindx, 5], 4))


            plt.figure(figc)
            plt.plot(cycL, resL[:, 1], 'g-')
            plt.plot(cycL, resL[:, 5], 'r-')
            plt.plot(cycL, resL[:, 3], 'k-')
            plt.plot(cycL, resL[:, 4], 'b-')

        plot_uq(cycL, bestA, c, r'$t_{init}$ = ' + str(stpt*upr) + 's', optval=None, optgap=None,
                fignum=figbv)

    plt.figure(num=figbv, figsize=(5, 4))
    plt.xlabel('ALD cycles')
    plt.ylabel(r'$C_{total}$ best value')
    plt.xlim(0, 800)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.show()
