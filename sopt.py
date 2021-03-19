import time
import matplotlib.pyplot as plt
import numpy as np
from core import get_model, global_opt, LOOCV, normalize, plot_2d
from scipy.special import erfc


def sbostep(X_, Y, bnds, acq='EI', model_type='GP', output=False,
            plotting=False, loocv=False):

    acqd = {'EI': EI,
            'Bayesian optimization': EI,
            'FMIN': FMIN,
            'SDMAX': SDMAX}

    acqdk = {'EI': {'y_min': Y.min()},
             'Bayesian optimization': {'y_min': Y.min()},
             'FMIN': {},
             'SDMAX': {}}

    acqf = acqd[acq]
    acqk = acqdk[acq]

    ubnds = np.zeros(bnds.shape)
    ubnds[:, 1] = 1.

    X_ = np.array(X_)
    Y = Y[:, 0]

    ndims = bnds.shape[0]

    model = get_model(X_, Y, bnds, model_type)
    if output and model_type == 'GP':
        print('GP kernel:', model.kernel_)

    if loocv:
        if output:
            print('LOOCV:')
        LOOCV(model, X_, Y, output, plotting)

    def surrogate_eval(x, model):
        ymean, ystd = model.predict(
            np.atleast_2d(x), return_std=True)
        return ymean, ystd

    def acqf_opt(x, bnds=None):
        ymean, ystd = model.predict(
            np.atleast_2d(x), return_std=True)
        res = acqf(np.squeeze(ymean), np.squeeze(ystd), **acqk)

        return res

    if output:
        print('find acquistion function minimum')

    x_p_, y_p = global_opt(acqf_opt, ubnds, typ='DE', output=output)

    x_p = np.zeros(x_p_.shape)
    for ii in range(bnds.shape[0]):
        x_p[ii] = x_p_[ii]*(bnds[ii, 1] - bnds[ii, 0]) + bnds[ii, 0]

    if plotting and ndims == 2:

        def obj_fs(x, bnds=None):
            return np.squeeze(model.predict(np.atleast_2d(x),
                              return_std=False))

        def obj_stddev(x, bnds=None):
            ymean, ystd = model.predict(
                np.atleast_2d(x), return_std=True)
            return ystd

        X__ = normalize(X_, bnds)
        x_p_ = normalize(np.atleast_2d(x_p), bnds)

        plot_2d(obj_fs, ubnds, pts=X__, newpts=x_p_,
                resolution=25, fignum='surrogate mean')
        plot_2d(obj_stddev, ubnds, pts=X__, newpts=x_p_,
                resolution=25, fignum='surrogate std. dev.')
        plot_2d(acqf_opt, ubnds, pts=X__, newpts=x_p_,
                resolution=25, fignum='acquisition function')

    return x_p


def EI(y_mean, y_std, y_min):
    alpha = (y_min - y_mean)/(np.sqrt(2)*np.pi)
    ei = y_std*(np.exp(-alpha**2)+np.sqrt(np.pi)*alpha*erfc(-1*alpha)) \
        / np.sqrt(2*np.pi)
    return -ei


def FMIN(y_mean, y_std):
    return y_mean


def soptimize(func, bnds, stset, acq='EI', model_type='GP',
              niter=30, plot_freq=1, output=False,
              loocv=False):

    randA = np.zeros((10000, bnds.shape[0]))
    for ii in range(bnds.shape[0]):
        randA[:, ii] = np.random.uniform(
            bnds[ii, 0], bnds[ii, 1], size=(10000))

    iterL = np.arange(0, niter, 1)
    XL, resL, teL, bestL = [], [], [], []
    c = np.linspace(1, 0, niter)
    best = 1e10
    st = time.time()
    for ii in iterL:
        te = time.time() - st

        if np.mod(ii+1, plot_freq) == 0 and output and ii >= len(stset):
            plotting = True
        else:
            plotting = False

        if acq == 'random':
            x = list(randA[ii, :])
        elif ii < stset.shape[0]:
            x = list(stset[ii, :])
        else:
            x = sbostep(XL, np.array(resL), bnds, acq,
                        model_type, output, plotting, loocv)

        res = np.atleast_1d(func(x, bnds))

        teL += [te]
        XL += [x]
        resL += [res]
        if res[0] < best:
            best = res[0]
        bestL += [best]

        if output:
            print('iteration', ii, 'complete')

        if plotting:
            plt.show()

    return iterL, XL, resL, bestL


def SDMAX(y_mean, y_std):
    return -y_std
