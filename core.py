import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from scipy.optimize import minimize, differential_evolution


def LOOCV(model_, X, y, output=True, plotting=True):

    y_t_p_s = np.zeros((y.size, 3))
    for ii in range(y.size):
        if output:
            print(ii)

        sel = np.arange(y.size) != ii
        X_t = X[sel, :]
        y_t = y[sel]
        X_v = X[ii, :].reshape(1, -1)
        y_v = y[ii]

        model = clone(model_)

        model.fit(X_t, y_t)
        y_v_p = model.predict(X_v, return_std=True)
        y_t_p_s[ii, 0] = y_v
        y_t_p_s[ii, 1] = y_v_p[0]
        y_t_p_s[ii, 2] = y_v_p[1]

    if output:
        print(model.kernel_)

        print('measured, predicted mean, predicted std. dev.')
        print(y_t_p_s)
        lts = np.abs(y_t_p_s[:, 0] - y_t_p_s[:, 1]) < y_t_p_s[:, 2]
        print('% predictions where error < 1 std. dev.', np.mean(lts))
        err = 100*np.abs(y_t_p_s[:, 0] - y_t_p_s[:, 1])/(y.max()-y.min())
        print('% accuracy:', 100 - np.mean(err), '+-', np.std(err))

    if plotting:

        plt.figure(figsize=(4, 3.75))

        ymin = np.min(y_t_p_s[:, :2])
        ymax = np.max(y_t_p_s[:, :2])
        yrng = ymax - ymin
        lwr = ymin - 0.1*yrng
        upr = ymax + 0.1*yrng

        plt.plot([lwr, upr], [lwr, upr], linestyle=':',
                 color='k', linewidth=1)
        plt.errorbar(y_t_p_s[:, 0], y_t_p_s[:, 1], yerr=y_t_p_s[:, 2],
                     marker='o', color='k', alpha=0.5,
                     linestyle='')
        plt.xlim([lwr, upr])
        plt.ylim([lwr, upr])
        plt.axes().set_aspect(1.0)
        plt.xlabel('measured', fontsize=13)
        plt.ylabel('predicted', fontsize=13)
        plt.tight_layout()


def get_model(X_, y, bnds, model_type='GP'):
    X = normalize(X_, bnds)

    n_dims = bnds.shape[0]

    if model_type == 'GP':

        kernel = ConstantKernel(1., (.01, 1000.0)) * \
            Matern(length_scale=np.ones(n_dims),
                   length_scale_bounds=[(0.05, .5)]*n_dims,
                   nu=2.5) +\
            WhiteKernel(noise_level=0.1,
                        noise_level_bounds=(1e-10, 1))

        model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=10).fit(np.atleast_2d(X), y)

    return model


def global_opt(obj, bnds, typ='L-BFGS-B', output=False):

    if typ == 'DE':
        res = differential_evolution(
            obj, bounds=np.atleast_2d(bnds))
        x_min = res.x
        y_min = res.fun
        if output:
            print(
                "DE (nfev, loc, min): ",
                res.nfev, np.round(x_min, 3), np.round(y_min, 3))
    else:
        if output:
            print('L-BFGS-B (x0, nfev, loc, min):')
        y_min = 1e10
        for ii in range(20):
            x0 = np.zeros((bnds.shape[0],))
            for jj in range(bnds.shape[0]):
                x0[jj] = np.random.uniform(bnds[jj, 0], bnds[jj, 1])

            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=np.atleast_2d(bnds), options={'gtol': 1e-5})

            if res.fun < y_min:
                x_min = res.x
                y_min = res.fun

            if output:
                print(
                    np.round(x0, 2), res.nfev,
                    np.round(res.x, 2), np.round(res.fun, 2))

        if output:
            print('y_min:', y_min)
    return x_min, y_min


def normalize(X, bnds=None):
    X = np.array(X)
    Xn = np.zeros(X.shape)
    if bnds is not None:
        for ii in range(bnds.shape[0]):
            Xn[:, ii] = (X[:, ii]-bnds[ii, 0]) / \
                (bnds[ii, 1] - bnds[ii, 0])
    else:
        for ii in range(X.shape[1]):
            Xn[:, ii] = (X[:, ii] - X[:, ii].min()) / \
                (X[:, ii].max() - X[:, ii].min())
    return Xn


def plot_2d(func, bnds, pts=None, newpts=None, resolution=100,
            serial=False, fignum=None):

    rng = bnds[:, 1] - bnds[:, 0]

    x0_ = np.linspace(bnds[0, 0], bnds[0, 1], resolution)
    x1_ = np.linspace(bnds[1, 0], bnds[1, 1], resolution)
    x0_m, x1_m = np.meshgrid(x0_, x1_)
    x0_l = x0_m.reshape((x0_m.size,))
    x1_l = x1_m.reshape((x1_m.size,))
    x = np.vstack([x0_l, x1_l]).T

    if serial:
        y = np.zeros((x.shape[0],))
        for ii in range(x.shape[0]):
            y[ii] = func(x[ii, :], bnds)
    else:
        y = func(x, bnds)

    plt.figure(num=fignum, figsize=(5, 4))
    plt.imshow(y.reshape((resolution, resolution)),
               interpolation='none',
               origin='lower',
               extent=[bnds[0, 0], bnds[0, 1], bnds[1, 0], bnds[1, 1]],
               aspect=rng[0]/rng[1])
    if pts is not None:
        plt.plot(pts[:, 0], pts[:, 1], 'g.')
    if newpts is not None:
        plt.plot(newpts[:, 0], newpts[:, 1], 'y.')

    plt.colorbar()


def plot_uq(iterL, bestA, color, label, optval=None, optgap=None,
            fignum=None):
    plt.figure(num=fignum)
    resA = np.array(bestA)

    low, mid, high = np.percentile(resA.T, [2.5, 50, 97.5], axis=1)

    if optgap is None:
        plt.plot(iterL, mid, alpha=0.9, color=color, label=label)
        plt.plot(iterL, low, alpha=0.9, ls=':', color=color)
        plt.plot(iterL, high, alpha=0.9, ls=':', color=color)
        plt.fill_between(
            iterL, low, high, alpha=0.15, facecolor=color)
    else:
        low = low - optval
        mid = mid - optval
        high = high - optval
        plt.semilogy(iterL, mid, alpha=0.9, color=color, label=label)
        plt.semilogy(iterL, low, alpha=0.9, ls=':', color=color)
        plt.semilogy(iterL, high, alpha=0.9, ls=':', color=color)
        plt.fill_between(
            iterL, low, high, alpha=0.15, facecolor=color)


def scale(x, bnds):
    x_ = np.zeros(x.shape)
    for ii in range(bnds.shape[0]):
        x_[..., ii] = x[..., ii]*(bnds[ii, 1] - bnds[ii, 0]) + bnds[ii, 0]
    return x_


def WP(msg, filename):
    """
    Summary:
        This function takes an input message and a filename, and appends that
        message to the file. This function also prints the message
    Inputs:
        msg (string): the message to write and print.
        filename (string): the full name of the file to append to.
    Outputs:
        both prints the message and writes the message to the specified file
    """
    fil = open(filename, 'a')
    print(msg)
    fil.write(msg)
    fil.write('\n')
    fil.close()
