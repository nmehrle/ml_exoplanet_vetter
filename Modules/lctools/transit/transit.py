#!/usr/bin/env python
import numpy as np
import scipy as sp
import logging
from lctools.util.constants import *
from lctools.util.util import mad, rect_sechmod
from lmfit import minimize, Parameters
from model_transits import modeltransit, occultquad
# create logger
logger = logging.getLogger(__name__)



class Transit():
    def __init__(self, name='',P= 1.0, q=0.01, qg=0.25, depth=1.e-3, epoch=0., SN=8., SNR=10.):
        self.required_keys=['name','P','q','qg','dip','epoch', 'SN', 'SNR']     
        self.name = name
        self.P = P
        self.q = q
        self.qgress = qg
        self.depth = depth
        self.epoch = epoch
        self.SN = SN
        self.SNR = SNR
        return
    

	def __str__(self):
		return '%13.6f %13.6f %6.4e %4.2e \n' % (self.P, self.epoch,self.depth,self.q)

    def calq(self,rstar=1,logg=4.5):
		if(self.dip<=0):
			raise ValueError('A real planet to calculate qexp should have positive depth')
			return
		else:
			rpstar=math.sqrt(self.dip)
			sqrtgm=math.sqrt(10**(logg)*(rstar*rsun)**2.)
			periods=self.P*day
			sqrtsemia=(periods*sqrtgm/(2*math.pi))**(1./3.)
			vcir=2*math.pi*sqrtsemia**2./periods
			dur=2*(rpstar+1)*rstar*rsun/vcir
			qexp=dur/periods
			return qexp

    def calc_epoch(self, jd):
        """Calculates the epoch number and midtransit time associated with each data point (no error bars).
    
        :param jd: length n array of light curve times.
    
        :return:
            dt_all: length n array of each data point's time to the nearest midtransit (days).
            epochs: length n array of epoch numbers associated with all data points.
            midpts: length n array of transit
        """
        end = int(np.ceil((jd[-1] - self.epoch) / self.P) + 1)
        start = int(np.floor((jd[0] - self.epoch) / self.P))
        cnt = 0
        epochs = []
        dt_all = []
        midpts = []
    
        for i in range(start, end):
            logger.debug('Transit number %d', cnt)
            midt = i * self.P + self.epoch
    
            dt = jd - midt
            now = np.where((dt >= -self.P/2) & (dt < self.P/2))[0]
    
            dt_all += list(dt[now])
            epochs += [i]*len(now)
            midpts += [midt]*len(now)
            cnt += 1
        self.dt = np.array(dt_all)
        self.epochs = np.array(epochs)
        self.midpts = np.array(midpts)
        logger.debug("end=%d, start=%d", end, start) 

    def get_intran(self, jd):
        indexlist = []
        for epoch in np.unique(self.epochs):
            index = (self.epochs == epoch)*(self.transwindow)
            if len(self.dt[index]) > 3:
                indexlist.append(index)

        return indexlist 
    def get_primary(self, jd, flux):
        self.transwindow = np.abs(self.dt) < self.q*self.P * 2.2 # transit window size is hard to determine
        dt_tra = self.dt[self.transwindow]
        f_tra = flux[self.transwindow]
        epochs_tra = self.epochs[self.transwindow]
        midpts_tra = self.midpts[self.transwindow]
        # bin data
        bins = np.linspace(min(dt_tra), max(dt_tra)+0.01, 50)
        binsize = bins[1] - bins[0]
        binned = np.digitize(dt_tra, bins)
        bin_mean = [f_tra[binned == i].mean() for i in range(1, len(bins))]

        # self.tranfit = get_fold_fit(dt_tra, f_tra, self.depth, self.P, self.window)
        return [dt_tra, f_tra-np.nanmedian(f_tra), (bins[:-1]+binsize/2), (np.array(bin_mean)-np.nanmedian(f_tra))]

    def get_occultation(self, jd, flux):
        phase = self.dt / self.P
        phase[np.where(phase < 0)] += 1
        occ = np.where((phase > 0.2) & (phase < 0.8))
        ph_occ = phase[occ]
        f_occ = flux[occ]

        tbins = np.linspace(0.2, 0.8, 21)
        f_bin = []
        stddev = []
        for i in range(0, 20):
            inds = np.where((ph_occ >= tbins[i]) & (ph_occ < tbins[i + 1]))[0]
            f_bin.append(np.mean(f_occ[inds]))
            stddev.append(np.std(f_occ[inds]))

        tbins = (tbins[0:-1] + 0.6 / 20.)*self.P


        return [ph_occ*self.P, f_occ-np.nanmedian(f_occ), tbins, f_bin-np.nanmedian(f_occ)] 

    def get_odd_even(self, jd, flux):

        dt_tra = self.dt[self.transwindow]
        f_tra = flux[self.transwindow]
        epochs_tra = self.epochs[self.transwindow]
        order = sorted(range(len(dt_tra)), key=lambda k: dt_tra[k])
        dt_tra = dt_tra[order]
        f_tra = f_tra[order]
        epochs_tra = epochs_tra[order]



        odd = np.where(epochs_tra % 2 != 0)[0]
        even = np.where(epochs_tra % 2 == 0)[0]
        
        self.transit_fit_fixP(dt_tra, f_tra)

        fit_odd = self.transit_refit_depth(dt_tra[odd], f_tra[odd])
        fit_even = self.transit_refit_depth(dt_tra[even], f_tra[even])

        oot = np.where(abs(dt_tra) > self.P*self.q)[0]
       
        sigma = mad(f_tra[oot])

        tarr = np.linspace(min(dt_tra), max(dt_tra), 200)
        oddmod = modeltransit([fit_odd.params['tc'].value, fit_odd.params['b'].value,
                                              fit_odd.params['Rs_a'].value, fit_odd.params['Rp_Rs'].value, 1,
                                              fit_odd.params['gamma1'].value,
                                              fit_odd.params['gamma2'].value], occultquad, self.P, tarr)
        evenmod = modeltransit([fit_even.params['tc'].value, fit_even.params['b'].value,
                                               fit_even.params['Rs_a'].value, fit_even.params['Rp_Rs'].value, 1,
                                               fit_even.params['gamma1'].value,
                                               fit_even.params['gamma2'].value], occultquad, self.P, tarr)
        # bin data
        bins = np.linspace(min(dt_tra), max(dt_tra)+0.01, 50)
        binsize = bins[1] - bins[0]
        binned_odd = np.digitize(dt_tra[odd], bins)
        binned_even = np.digitize(dt_tra[even], bins)
        f_odd_bin = [f_tra[odd][binned_odd == i].mean() for i in range(1, len(bins))]
        f_even_bin = [f_tra[even][binned_even == i].mean() for i in range(1, len(bins))]
        
        dt_bin = bins[:-1]+binsize/2.
        return [dt_tra[odd], f_tra[odd]-np.nanmedian(f_tra[odd]), dt_bin, np.array(f_odd_bin)-np.nanmedian(f_tra[odd]), tarr, oddmod-np.nanmedian(f_tra[odd]), dt_tra[even], f_tra[even]-np.nanmedian(f_tra[even]), dt_bin, np.array(f_even_bin)-np.nanmedian(f_tra[even]), tarr, evenmod-np.nanmedian(f_tra[even]), sigma]

    def transit_fit_fixP(self, dt_tra, f_tra):
        params = Parameters()
        params.add('tc', value=0, vary=False, min=-0.1, max=0.1)
        params.add('b', value=0.6, vary=True, min=-1.2, max=1.2)
        params.add('Rs_a', value=0.1, vary=True, min=0., max=0.5)
        params.add('Rp_Rs', value=self.depth ** 0.5, vary=True, min=0)
        params.add('F', value=1, vary=False)
        params.add('gamma1', value=0.3, vary=True, min=0, max=0.5)  # should I let these float?
        params.add('gamma2', value=0.3, vary=True, min=0, max=0.5)
        params.add('a0', value=1, vary=False)
        params.add('a1', value=0, vary=False)

        self.fit = minimize(self.residual, params, args=(dt_tra, f_tra, self.P, False))
        return  

    def transit_refit_depth(self, dt_tra, f_tra):
        p0 = self.fit
        params = Parameters()
        params.add('tc', value=0, vary=False)
        params.add('b', value=p0.params['b'].value, vary=True)
        params.add('Rs_a', value=p0.params['Rs_a'].value, vary=True, min=0., max=0.5)
        params.add('Rp_Rs', value=p0.params['Rp_Rs'].value, vary=True)
        params.add('F', value=1, vary=False)
        params.add('gamma1', value=p0.params['gamma1'].value, vary=False)
        params.add('gamma2', value=p0.params['gamma2'].value, vary=False)
        params.add('a0', value=1, vary=False)
        params.add('a1', value=0, vary=False)

        fit = minimize(self.residual, params, args=(dt_tra, f_tra, self.P, False))

        return fit

    def residual(self, params, t, data, period=1, sech=True):
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
            model = modeltransit([tc, b, r_a, Rp_Rs, F, gamma1, gamma2], occultquad, period,
                                                t)
            model *= (a0 + a1 * t)
        return data - model


    def gentran(time):
        ftime=sp.zeros(len(time))
        ftime=(time-self.epoch-0.5*self.period)/self.period-((time-self.epoch-0.5*self.period)/self.period).astype(int)
        ind=ftime<0
        ftime[ind]+=1
        #print min(ftime),max(ftime)
        intran=(ftime > (0.5-q/2.0))*(ftime < (0.5+q/2.0))
        #print ftime[intran])
        return intran
	
