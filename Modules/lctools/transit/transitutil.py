"""
    Written by Liang Yu, Jan 2017
    Included in this project with permission and licensed under MIT.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
import warnings
from lmfit import minimize, Parameters
import model_transits
import logging
from scipy import interpolate as itp
import urllib
import matplotlib.image as mpimg

module_logger = logging.getLogger(__name__)

def sechmod(t, b, t0, w):
    """Fits a sech model to a transit as a faster, simpler alternative to the Mandel-Agol model.

    INPUTS:
        t - length n array of light curve times (days)
        b - 2*transit depth, defined as negative
        t0 - mid-transit time
        w - width of transit (days)

    RETURNS:
        nd array of model fluxes
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    return 1 + b / (np.exp(-(t - t0) ** 2. / w ** 2.) + np.exp((t - t0) ** 2. / w ** 2.))


def rect_sechmod(t, b, t0, w, a0, a1):
    """Fits a sech model with linear detrending of background.

    INPUTS: see sechmod
        a0, a1 - coefficients of linear detrending function. The background is modelled as a0 + a1*t

    RETURNS: see sechmod
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    return (1 + b / (np.exp(-(t - t0) ** 2. / w ** 2.) + np.exp((t - t0) ** 2. / w ** 2.))) * (a0 + a1 * t)



def residual(params, t, data, period=1, sech=True):
    """Residual function for fitting for midtransit times.

    INPUTS:
        params - lmfit.Parameters() object containing parameters to be fitted.
        t - length n array of light curve times (days).
        data - length n array of normalized light curve fluxes. Median out-of-transit flux should be set to 1.
        period - period of transit (days).
        sech - boolean. If True, will use sech model. Otherwise will fit Mandel-Agol model instead.
        The params argument must match the format of the model chosen.

    RETURNS:
        res - residual of data - model, to be used in lmfit.
    """

    if sech:
        vals = params.valuesdict()
        tc = vals['tc']
        b = vals['b']
        w = vals['w']
        a0 = vals['a0']
        a1 = vals['a1']
        model = rect_sechmod(t, b, tc, w, a0, a1)
    else:
        vals = params.valuesdict()
        tc = vals['tc']
        b = vals['b']
        r_a = vals['Rs_a']
        Rp_Rs = vals['Rp_Rs']
        F = vals['F']
        gamma1 = vals['gamma1']
        gamma2 = vals['gamma2']
        a0 = vals['a0']
        a1 = vals['a1']
        model = model_transits.modeltransit([tc, b, r_a, Rp_Rs, F, gamma1, gamma2], model_transits.occultquad, period,
                                            t)
        model *= (a0 + a1 * t)
    return data - model


def plot_full_lc(t, flux, p, t0, ax):
    """Plots full light curve.

    :param t: length n array of light curve times.
    :param flux: length n array of light curve fluxes (normalized to 1).
    :param p: estimated transit period (days).
    :param t0: estimated transit ephemeris (BJD-2454000).
    :param ax: plot handle for full LC plot.

    :return:
        ax - plot handle of LC plot.
    """
    ax.plot(t, (flux-np.median(flux))/1.e-6, lw=0, marker='.', color='k')
    sigma = mad(flux)
    ax.set_ylim((min(flux)-np.nanmedian(flux))/1.e-6, (7*sigma)/1.e-6)  # inverted y-axis
    ax.set_xlim(min(t), max(t))  # inverted y-axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    end = int(floor((t[-1] - t0) / p) + 1)
    start = int(floor((t[0] - t0) / p))
    midpts = np.arange(start,end)*p + t0
    module_logger.debug("end=%d, start=%d, t[0] = %f, t[-1] = %f, p=%f", end, start, t[0], t[-1], p)

    for midpt in midpts:
        ax.vlines(midpt, (min(flux)-np.nanmedian(flux))/1.e-6, (max(flux)-np.nanmedian(flux))/1.e-6 , color='r', linestyle='--')

    return ax


def calc_epoch(t, p, t0):
    """Calculates the epoch number and midtransit time associated with each data point (no error bars).

    :param t: length n array of light curve times.
    :param p: float, transit period (days).
    :param t0: float, transit ephemeris (BJD-2454000).

    :return:
        dt_all: length n array of each data point's time to the nearest midtransit (days).
        epochs: length n array of epoch numbers associated with all data points.
        midpts: length n array of transit
    """
    end = int(floor((t[-1] - t0) / p) + 1)
    start = int(floor((t[0] - t0) / p))
    cnt = 0
    epochs = []
    dt_all = []
    midpts = []

    for i in range(start, end):
        module_logger.debug('Transit number %d', cnt)
        midt = i * p + t0

        dt = t - midt
        now = np.where((dt >= -p/2) & (dt < p/2))[0]

        dt_all += list(dt[now])
        epochs += [i]*len(now)
        midpts += [midt]*len(now)
        cnt += 1
    return np.array(dt_all), np.array(epochs), np.array(midpts)


def plot_indiv_trans(name, t, f, p, t0, window, p0, plotbad=True, plots=True, sech=True):
    """ Plot individual transits with a choice of sech or Mandel-Agol fit. Also calculates epochs and midpoints
    (with errors) required for O-C plot.

    INPUTS:
        t - nd array of light curve times.
        f - nd array of normalized light curve fluxes.
        window - approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
        p0 - best guess of fit parameters. If sech, p0 = [w0, depth]. w0 is fractional width of transit from read_lc.
        If Mandel-Agol, p0 = [b,Rs_a,Rp_Rs,gamma1,gamma2].
        plotbad - set to True if you want to plot incomplete or misshapen transits along with good ones.
        sech - set to True if you want a sech fit. Otherwise use Mandel-Agol model.

    RETURNS:
        dt_all - nd array showing the time (days) to the nearest transit for each point.
        epochs - nd array of epoch number of each point.
        midpts - nd array of midtransit times associated with all points.
        err - array of errors on each midtransit time.
    """

    end = int(floor((t[-1] - t0) / p) + 1)
    start = int(floor((t[0] - t0) / p))
    cnt = 0
    epochs = []
    dt_all = []
    midpts = []
    err = []
    valid_trans = []

    params = Parameters()
    if sech:
        w0 = p0[0]
        depth = p0[1]
        params.add('tc', value=0, vary=True, min=-0.1, max=0.1)
        params.add('b', value=depth * 2, vary=False)
        params.add('w', value=w0 * p, vary=False)
        params.add('a0', value=1)
        params.add('a1', value=0)
    else:
        depth = -p0[2] ** 2
        params.add('tc', value=0, vary=True, min=-0.1, max=0.1)
        params.add('b', value=p0[0], vary=False)
        params.add('Rs_a', value=p0[1], vary=False)
        params.add('Rp_Rs', value=p0[2], vary=False)
        params.add('F', value=1, vary=False)
        params.add('gamma1', value=p0[3], vary=False)  # should I let these float?
        params.add('gamma2', value=p0[4], vary=False)
        params.add('a0', value=1, vary=True)
        params.add('a1', value=0, vary=True)

    for i in range(start, end):
        module_logger.debug('Transit number %d', cnt)
        midt = i * p + t0

        dt = t - midt
        oot = np.where((abs(dt) > window) & (abs(dt) < window + 0.2 * p))[0]

        try:
            fn = f / np.median(f[oot])
        except:
            pass

        select = np.where(abs(dt) < (window + 0.1 * p))[0]  # select single transit
        good = np.where(abs(dt) <= p / 2)[0]  # all points belonging to current transit

        if plots:
            if cnt % 8 == 0:
                plt.close('all')
                fig, ax = plt.subplots(8, figsize=(6, 12), sharex=True)

            if plotbad or (len(select) > 5):
                ax[cnt % 8].plot(dt[select], fn[select], lw=0, marker='.')
                ax[cnt % 8].axvline(x=0, color='k', ls='--')
                ax[cnt % 8].set_xlabel('Time from midtransit (days)')
                ax[cnt % 8].set_ylabel('Relative flux')
                ax[cnt % 8].set_ylim(1 + depth - 0.0003, 1 + 0.0003)
                ax[cnt % 8].set_xlim(-0.3, 0.3)
                ax[cnt % 8].locator_params(axis='y', nbins=5)
                ax[cnt % 8].get_yaxis().get_major_formatter().set_useOffset(False)
                ax[cnt % 8].annotate(str(cnt), xy=(0.85, 0.1), xycoords='axes fraction', size=15)

        dt_all += list(dt[good])

        if len(select) > 5:
            # fit sech to each transit

            try:
                fit = minimize(residual, params, args=(dt[select], fn[select], p, sech))
                fiterr = np.sqrt(fit.covar[0][0])
                err.append(fiterr)

                midpts += len(good) * [fit.params['tc'].value + i * p + t0]
                epochs += len(good) * [i]

                if plots:
                    tc = fit.params['tc'].value
                    a0 = fit.params['a0'].value
                    a1 = fit.params['a1'].value
                    tarr = np.linspace(dt[select][0], dt[select][-1], 200)
                    if sech:
                        fmod = rect_sechmod(tarr, depth * 2, tc, w0 * p, a0, a1)
                    else:
                        fmod = model_transits.modeltransit([fit.params['tc'].value, fit.params['b'].value,
                                                            fit.params['Rs_a'].value, fit.params['Rp_Rs'].value, 1,
                                                            fit.params['gamma1'].value,
                                                            fit.params['gamma2'].value], model_transits.occultquad, p,
                                                           tarr)
                        fmod *= (fit.params['a0'].value + fit.params['a1'].value * tarr)
                    ax[cnt % 8].plot(tarr, fmod, color='r')

                valid_trans.append(i)
            except TypeError:
                midpts += len(good) * [np.nan]
                epochs += len(good) * [np.nan]
                err.append(np.nan)
                module_logger.error('Fit failed')
                pass
        else:
            midpts += len(good) * [np.nan]
            err.append(np.nan)
            epochs += len(good) * [np.nan]
            module_logger.error('Too few data points')

        if plots and ((cnt % 8 == 7) or (i == end - 1)):
            plt.savefig('outputs/' + name + 'alltrans' + str(ceil(cnt / 8. + 0.01)) + '.pdf', dpi=150,
                        bbox_inches='tight')
        if plotbad or (len(select) > 5):
            cnt += 1

    module_logger.debug('total transits: %d', cnt)
    epochs = np.array(epochs)
    print 'good transits:', np.unique(epochs[np.where(~np.isnan(epochs))[0]])

    return np.array(dt_all), epochs, np.array(midpts), np.array(err)


def make_folded_lc(dt, f, epochs, midpts, window, fig=None):
    """Returns dt, flux, epochs and midpoints belonging to data points within transit windows.
    Makes plot of folded light curve if fig parameter is not None.

    :param dt: length n array showing the time (days) to the nearest transit for each point. Output dt_all of plot_indiv_trans.
    :param f: length n array of normalized fluxes.
    :param epochs: length n array of epoch numbers associated with all points.
    :param midpts: length array of midtransit times associated with all points.
    :param window: approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
    :param fig: plot handle indicating desired plot dimensions. No plots will be made if set to None.

    :return:
        dt_tra - array of selected dt that fall within transit windows.
        f_tra - array of selected fluxes that fall within transit windows.
        epochs_tra - array of selected epochs that fall within transit windows.
        midpts_tra - array of selected midtransit times that fall within transit windows.
        fig - plot handle of folded transit.
    """
    transwindow = np.where(abs(dt) < window * 2.2)  # transit window size is hard to determine
    dt_tra = dt[transwindow]
    f_tra = f[transwindow]
    epochs_tra = epochs[transwindow]
    midpts_tra = midpts[transwindow]
    # bin data
    bins = np.linspace(min(dt_tra), max(dt_tra)+0.01, 50)
    binsize = bins[1] - bins[0]
    binned = np.digitize(dt_tra, bins)
    bin_mean = [f_tra[binned == i].mean() for i in range(1, len(bins))]


    if fig is not None:
        # fig = plt.figure(figsize=(10, 4))
        sigma = mad(f_tra)
        fig.plot(dt_tra*24, (f_tra-np.median(f_tra))/1.e-6, lw=0, marker='.', color='0.7')
        fig.plot(24*(bins[:-1]+binsize/2), (np.array(bin_mean)-np.nanmedian(f_tra))/1e-6, marker='.', lw=0, color='#1c79e8', markersize=20)
        fig.axvline(x=0, ls='--', color='k')
        fig.set_ylim((min(f_tra)-np.median(f_tra))/1.e-6, sigma*6/1.e-6)
        fig.get_yaxis().get_major_formatter().set_useOffset(False)

    order = sorted(range(len(dt_tra)), key=lambda k: dt_tra[k])
    dt_tra = dt_tra[order]
    f_tra = f_tra[order]
    epochs_tra = epochs_tra[order]
    midpts_tra = midpts_tra[order]

    if fig is not None:
        return dt_tra, f_tra, epochs_tra, midpts_tra, fig
    else:
        return dt_tra, f_tra, epochs_tra, midpts_tra


def get_fold_fit(dt_tra, f_tra, depth, period, window, fig=None):
    """Uses lmfit to get a good estimate of the Mandel-Agol parameters from the folded light curve. The curve fitting
    routine will then be rerun using these better parameters.

    :param dt_tra: array of selected dt that fall within transit windows.
    :param f_tra: array of selected fluxes that fall within transit windows.
    :param depth: estimate of transit depth obtained from sech fit. Defined as positive.
    :param period: estimate of transit period (days).
    :param window: approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
    :param fig: plot handle indicating desired plot dimensions. e.g. fig = plt.figure(figsize=(10,4)).
    No plots will be made if set to None.

    :return:
        fit - best-fit Mandel-Agol parameters from lmfit.minimise(). Contains the following params:
            tc - midtransit time. Centred at 0.
            b - impact parameter.
            Rs_a - radius of star/semimajor axis.
            F - out-of-transit flux, fixed at 1.
            gamma1, gamma2 - quadratic limb darkening parameters from Mandel & Agol (2002)
        fig - plot handle of folded transit with best-fit model.
    """
    # plotting can always be skipped now unless you want to debug
    params = Parameters()
    params.add('tc', value=0, vary=False, min=-0.1, max=0.1)
    params.add('b', value=0.8, vary=True, max=1.2)
    params.add('Rs_a', value=0.1, vary=True, min=0., max=0.5)
    params.add('Rp_Rs', value=depth ** 0.5, vary=True, min=0)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=0.3, vary=True, min=0, max=0.5)  # should I let these float?
    params.add('gamma2', value=0.3, vary=True, min=0, max=0.5)
    params.add('a0', value=1, vary=False)
    params.add('a1', value=0, vary=False)

    fit = minimize(residual, params, args=(dt_tra, f_tra, period, False))
    tarr = np.linspace(min(dt_tra), max(dt_tra), 100)
    fmod = model_transits.modeltransit([fit.params['tc'].value, fit.params['b'].value, fit.params['Rs_a'].value,
                                        fit.params['Rp_Rs'].value, 1, fit.params['gamma1'].value,
                                        fit.params['gamma2'].value], model_transits.occultquad, period, tarr)

   
    if fig is not None:
        fig.plot(dt_tra, f_tra, lw=0, marker='.', color='0.7')
        # fig.plot(tarr, fmod, color='r')
        fig.plot(tarr * 24., (fmod-np.nanmedian(f_tra))/1.e-6, color='r', lw=2)
        fig.axvline(x=-window, color='k', ls='--')
        fig.axvline(x=window, color='k', ls='--')
        fig.set_xlabel('Time from midtransit (days)')
        fig.set_ylabel('Relative flux')
    if fig is not None:
        return fit, fig
    else:
        return fit


def get_oc(all_epochs, all_midpts, err, fig=None):
    """Calculates accurate values for ephemeris and period. Plots O-C diagram if desired.

    :param all_epochs: nd array of epoch numbers associated with all points. From plot_indiv_trans.
    :param all_midpts: nd array of midtransit times associated with all points. From plot_indiv_trans.
    :param err: array of errors on midtransit times. One value for each unique time. From plot_indiv_trans.
    :param fig: plot handle indicating desired plot dimensions. e.g. fig = plt.figure(figsize=(10,4)).
    No plots will be made if set to None.

    :return:
        p_fit - best-fit transit period (days).
        t0_fit - best-fit transit ephemeris.
        fig - plot handle for O-C plot.
    """
    try:
        epochs = np.unique(all_epochs[np.where(~np.isnan(all_epochs))[0]])
        midpts = np.unique(all_midpts[np.where(~np.isnan(all_midpts))[0]])
        err = np.unique(err[np.where(~np.isnan(err))[0]])
    except:
        print('Error: invalid epochs and/or ephemerides')
        raise

    if len(epochs) > 2:
        coeffs, cov = np.polyfit(epochs, midpts, 1, cov=True)
        p_fit = coeffs[0]
        p_err = np.sqrt(cov[0, 0])
        t0_fit = coeffs[1]
        t0_err = np.sqrt(cov[1, 1])
    else:
        p_fit = (midpts[1] - midpts[0]) / (epochs[1] - epochs[0])
        p_err = 0
        t0_fit = (midpts[1] * epochs[0] - midpts[0] * epochs[1]) / (epochs[0] - epochs[1])
        t0_err = 0

    print 'p=', p_fit, '+-', p_err
    print 't0=', t0_fit, '+-', t0_err

    if len(epochs) > 2:
        fit = np.polyval(coeffs, epochs)
        oc = (midpts - fit) * 24.
    else:
        oc = midpts * 0

    err = np.array(err) * 24.
    if fig is not None:
        plt.close('all')
        # fig = plt.figure(figsize=(9, 4))
        plt.errorbar(epochs, oc, yerr=err, fmt='o')
        plt.axhline(color='k', ls='--')
        plt.ylabel('O-C (hours)')
        plt.xlabel('Epochs')
        plt.xlim(-0.1, max(epochs) + 1)
        # plt.savefig('outputs/' + name + '_oc.pdf', dpi=150, bbox_inches='tight')

    if fig is not None:
        return p_fit, t0_fit, fig
    else:
        return p_fit, t0_fit


def odd_even(dt_tra, f_tra, epochs_tra, window, period, p0, ax1, ax2):
    """Plots odd vs. even transits and calculates difference in depth.

    :param dt_tra: see get_fold_fit.
    :param f_tra: see get_fold_fit.
    :param epochs_tra: see get_fold_fit.
    :param window: see get_fold_fit.
    :param period: see get_fold_fit.
    :param p0: lmfit.MinimizerResult object. Good estimate of Mandel-Agol parameters from get_fold_fit.
    params = [b, Rs_a, Rp_Rs, gamma1, gamma2]
    :param ax1: plot handle for odd transit.
    :param ax2: plot handle for even transit.

    :return:
        ax1, ax2 - plot handles for odd-even comparison plot.
    """
    #
    odd = np.where(epochs_tra % 2 != 0)[0]
    even = np.where(epochs_tra % 2 == 0)[0]

    params = Parameters()
    params.add('tc', value=0, vary=False)
    params.add('b', value=p0.params['b'].value, vary=True)
    params.add('Rs_a', value=p0.params['Rs_a'].value, vary=True, min=0., max=0.5)
    params.add('Rp_Rs', value=p0.params['Rp_Rs'], vary=True)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=p0.params['gamma1'], vary=False)
    params.add('gamma2', value=p0.params['gamma2'], vary=False)
    params.add('a0', value=1, vary=False)
    params.add('a1', value=0, vary=False)

    fit_odd = minimize(residual, params, args=(dt_tra[odd], f_tra[odd], period, False))
    fit_even = minimize(residual, params, args=(dt_tra[even], f_tra[even], period, False))

    oot = np.where(abs(dt_tra) > window)[0]
    sigma = mad(f_tra[oot])

    tarr = np.linspace(min(dt_tra), max(dt_tra), 200)
    oddmod = model_transits.modeltransit([fit_odd.params['tc'].value, fit_odd.params['b'].value,
                                          fit_odd.params['Rs_a'].value, fit_odd.params['Rp_Rs'].value, 1,
                                          fit_odd.params['gamma1'].value,
                                          fit_odd.params['gamma2'].value], model_transits.occultquad, period, tarr)
    evenmod = model_transits.modeltransit([fit_even.params['tc'].value, fit_even.params['b'].value,
                                           fit_even.params['Rs_a'].value, fit_even.params['Rp_Rs'].value, 1,
                                           fit_even.params['gamma1'].value,
                                           fit_even.params['gamma2'].value], model_transits.occultquad, period, tarr)
    odd_depth = min(oddmod)
    even_depth = min(evenmod)
    diff = abs(odd_depth - even_depth) / sigma

    # bin data
    bins = np.linspace(min(dt_tra), max(dt_tra)+0.01, 50)
    binsize = bins[1] - bins[0]
    binned_odd = np.digitize(dt_tra[odd], bins)
    binned_even = np.digitize(dt_tra[even], bins)
    bin_mean_odd = [f_tra[odd][binned_odd == i].mean() for i in range(1, len(bins))]
    bin_mean_even = [f_tra[even][binned_even == i].mean() for i in range(1, len(bins))]

    # plt.subplots_adjust(wspace=0, hspace=0)
    ax1.plot(dt_tra[odd] * 24., (f_tra[odd]-np.nanmedian(f_tra))/1.e-6, lw=0, marker='.', color='0.7')
    ax1.plot(24*(bins[:-1]+binsize/2), (np.array(bin_mean_odd)-np.nanmedian(f_tra))/1e-6, marker='.', color='#1c79e8', markersize=20, lw=0)
    ax1.plot(tarr * 24., (oddmod-np.nanmedian(f_tra))/1.e-6, color='#e88b1c', lw=4)
    ax1.axhline(y=(odd_depth-np.nanmedian(f_tra))/1.e-6, color='b', ls='--', lw=2)
    ax1.set_xlim(min(dt_tra) * 24, max(dt_tra) * 24)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.set_ylim(min(f_tra-np.nanmedian(f_tra))/1.e-6, 6*sigma/1.e-6)

    ax2.plot(dt_tra[even] * 24., (f_tra[even]-np.nanmedian(f_tra))/1.e-6, lw=0, marker='.', color='0.7')
    ax2.plot(24*(bins[:-1]+binsize/2), (np.array(bin_mean_even)-np.nanmedian(f_tra))/1e-6, marker='.', color='#1c79e8', markersize=20, lw=0)
    ax2.plot(tarr * 24., (evenmod-np.nanmedian(f_tra))/1.e-6, color='#e88b1c', lw=4)
    ax2.axhline(y=(even_depth-np.nanmedian(f_tra))/1.e-6, color='b', ls='--', lw=2)
    ax2.set_xlim(min(dt_tra) * 24, max(dt_tra) * 24)
    ax2.annotate(r'Diff: %.3f $\sigma$' % diff, xy=(0.52, 0.1), xycoords='figure fraction', size=15)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.set_ylim(min(f_tra-np.nanmedian(f_tra))/1.e-6, 6*sigma/1.e-6)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    # plt.savefig('outputs/' + name + '_oddeven.pdf', dpi=150, bbox_inches='tight')
    return ax1, ax2


def occultation(dt, f, p, fig):
    """Plots folded light curve between two transits to check for secondary eclipses.

    :param dt: nd array showing the time (days) to the nearest transit for each point. Output dt_all of plot_indiv_trans.
    :param f: nd array of light curve flux.
    :param p: best-fit period (days).
    :param fig: plot handle of secondary eclipse plot.

    :return:
        fig - plot handle of secondary eclipse plot.
    """
    phase = dt / p
    phase[np.where(phase < 0)] += 1
    occ = np.where((phase > 0.2) & (phase < 0.8))
    ph_occ = phase[occ]
    f_occ = f[occ]

    tbins = np.linspace(0.2, 0.8, 21)
    fbin = []
    stddev = []
    for i in range(0, 20):
        inds = np.where((ph_occ >= tbins[i]) & (ph_occ < tbins[i + 1]))[0]
        fbin.append(np.mean(f_occ[inds]))
        stddev.append(np.std(f_occ[inds]))

    tbins = tbins[0:-1] + 0.6 / 20.
    sigma = np.std(f_occ)

    fig.plot(ph_occ*p*24., (f_occ-np.nanmedian(f_occ))/1.e-6, lw=0, marker='.', color='0.75')
    fig.plot(tbins*p*24., (fbin-np.nanmedian(f_occ))/1.e-6, lw=2, marker='.', color='#1c79e8', markersize=20)
    #ax2.plot(24*(bins[:-1]+binsize/2), (np.array(bin_mean_even)-np.nanmedian(f_tra))/1e-6, marker='.', color='#1c79e8', markersize=20, lw=0)
    #if (min(f_occ)-np.nanmedian(f_occ))< -6*sigma:
    #fig.set_ylim((min(f_occ)-np.nanmedian(f_occ))/1.e-6, 6*sigma/1.e-6)
    #else:
    fig.set_ylim(-3*sigma/1.e-6, 2*sigma/1.e-6)

    fig.get_yaxis().get_major_formatter().set_useOffset(False)

    return fig


def plot_centroid(dt, epochs, x, y, window, axx, axy):
    """Plot x and y centroid positions.

    :param dt: length n array of times to the next midtransit.
    :param epochs: length n array of epoch number corresponding to each point.
    :param x: length n array of x centroid positions.
    :param y: length n array of y centroid positions.
    :param window: size of transit window (days).
    :param axx: plot handle for x centroid plot.
    :param axy: plot handle for y centroid plot.

    :return:
        axx, axy - plot handles for x and y centroids as functions of time from midtransit.
    """
    x_clean = np.array([])
    y_clean = np.array([])
    dt_clean = np.array([])

    # gaps = np.diff(t)
    # std = np.std(gaps)
    # gaploc = np.where(gaps > 3*std)[0]
    # gaploc += 1
    # gaploc = np.insert(gaploc, 0, 0)
    # gaploc = np.append(gaploc, len(t)-1)
    #
    # # make long segment shorter
    # gapsize = np.diff(gaploc)
    # offset = 0
    # # for i in range(len(gapsize)):
    # #     if gapsize[i] > 800:
    # #         gaploc = np.insert(gaploc, i+1+offset, gaploc[i]+gapsize[i]/2)
    # #         offset += 1

    # adjust segment length if needed. Currently segments are separated by gaps.

    # fig = plt.figure()

    epoch_list = np.unique(epochs[np.where(~np.isnan(epochs))])
    for epoch in epoch_list:
        now = np.where(epochs == epoch)[0]

        #print x
        x_now = x[now]
        y_now = y[now]
        dt_now = dt[now]

        if len(now) < 5:
            fit = np.polyfit(dt_now, x_now, 1)
            xmod = np.polyval(fit, dt_now)

            fit = np. polyfit(dt_now, y_now, 1)
            ymod = np.polyval(fit, dt_now)

        else:
            tck = itp.splrep(dt_now, x_now, s=len(dt_now) - np.sqrt(2 * len(dt_now)))
            xmod = itp.splev(dt_now, tck)
            sigx = np.std(x_now - xmod)

            tck = itp.splrep(dt_now, y_now, s=len(dt_now) - np.sqrt(2 * len(dt_now)))
            ymod = itp.splev(dt_now, tck)
            sigy = np.std(y_now - ymod)

            try:
                # sigma clip
                good_x = np.where(abs(x_now-xmod) < 2*sigx)[0]
                good_y = np.where(abs(y_now-ymod) < 2*sigy)[0]

                tck = itp.splrep(dt_now[good_x], x_now[good_x], s=len(good_x) - np.sqrt(2 * len(good_x)))
                xmod = itp.splev(dt_now, tck)

                tck = itp.splrep(dt_now[good_y], y_now[good_y], s=len(good_y) - np.sqrt(2 * len(good_y)))
                ymod = itp.splev(dt_now, tck)

            except TypeError:
                pass
        # plt.plot(t_now, x_now, 'k.')
        # plt.plot(t_now, xmod, color='r', lw=2)
        # plt.show()

        x_now -= xmod
        y_now -= ymod

        x_clean = np.append(x_clean, x_now)
        y_clean = np.append(y_clean, y_now)
        dt_clean = np.append(dt_clean, dt_now)

    trans = np.where(abs(dt_clean) < window*2.2)
    sig = np.std(x_clean)
    y_clean += max(x_clean) - min(y_clean) + 0.8*sig
    axx.plot(dt_clean[trans]*24., x_clean[trans], 'k.')
    axy.plot(dt_clean[trans]*24., y_clean[trans], 'r.')

    axx.set_ylim(min(x_clean), max(y_clean))  # there's probably an easier way to set the same ylim for the two axes
    axy.set_ylim(min(x_clean), max(y_clean))
    return axx, axy


def finder_chart(epic, ax):
    """Retrieves finder chart from ExoFOP for targets with known EPIC number. Currently only works for K2.

    :param epic: str, EPIC ID of target.
    :param ax: plot handle for finder chart.
    :return:
    ax: plot handle for finder chart.
    """
    urllib.urlretrieve('https://cfop.ipac.caltech.edu/k2/files/' + epic + '/Finding_Chart/' + epic + 'F-mc20150630.png',
                       epic + '_chart.png')
    img = mpimg.imread(epic + '_chart.png')

    ax.imshow(img)
    ax.axis('off')
    return ax

def transit_fit(p0, dt_tra, f_tra, period):

    params = Parameters()
    params.add('tc', value=0, vary=False)
    params.add('b', value=p0.params['b'].value, vary=True)
    params.add('Rs_a', value=p0.params['Rs_a'].value, vary=True, min=0., max=0.5)
    params.add('Rp_Rs', value=p0.params['Rp_Rs'], vary=True)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=p0.params['gamma1'], vary=False)
    params.add('gamma2', value=p0.params['gamma2'], vary=False)
    params.add('a0', value=1, vary=False)
    params.add('a1', value=0, vary=False)

    fit = minimize(residual, params, args=(dt_tra, f_tra, period, False))

    return fit
