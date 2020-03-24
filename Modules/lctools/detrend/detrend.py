# 
# Copyright (C) 2017 - Massachusetts Institute of Technology (MIT) 
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 
# 
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details. 
# 
# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 


import math
import numpy as np
import scipy as sp
import copy
from scipy import linalg
from scipy.optimize import leastsq
from scipy import interpolate
from scipy import signal
from scipy import stats
import pandas as pds
import ConfigParser
import logging
from scipy.optimize import minimize

import matplotlib
from matplotlib import pyplot as plt
import lctools 
from lctools.util.configurable import ConfigurableObject
from lctools.thirdparty import fit_kepler_spline
import patools
from patools.util.query import QlpQuery
# import warnings
# warnings.filterwarnings("error")

# create logger
logger = logging.getLogger(__name__)

class DetrendFunc(ConfigurableObject):
    config_keys = ['LC']
    def __init__(self):
        super(DetrendFunc, self).__init__() 
        return
    
    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d.keys():
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d.keys():
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

   
    @staticmethod
    def create(cfg, method='cos'): 
        for c in [PolyDetrend, COSDetrend, COSDetrendSegment, EPDDetrend, EPDQuatDetrend,EPDDetrendSegment, FocusDetrend, GPDetrend, KSPDetrend]:
            if c.istype(method):
                return c(cfg)

        raise NotImplementedError, 'detrending method %s is not recognized' % method 
    def check_data(self, lightcurve):
        for key in self.required_cols:
            lightcurve.check_data(key)
        return 

    def detrend(self, lightcurve, detrend_col='rlc'):
        self.check_data(lightcurve)
        ltf = self._detrend(lightcurve.data['jd'], lightcurve.data[detrend_col], lightcurve.data)
        lightcurve.data['dlc'] = ltf
        return  
    
    def _detrend(data):
        raise NotImplementedError
        

class PolyDetrend(DetrendFunc):
    # Detrend with polynomial fit
    METHOD = 'poly'
    def __init__(self, wn=13, norder=5):
        super(PolyDetrend, self).__init__()
        self.wn = wn
        self.norder = norder
        
        self.required_keys = ['wn', 'norder']

    @staticmethod
    def istype(method):
        return method == PolyDetrend.METHOD


    def _detrend(self, jd, mag, data, noplot=True, intran=[]):
        otime = jd 
        if not intran:
            intran = otime < 0
        oflux = mag 
        order = self.norder
        length = len(oflux)
        ctime = otime[-intran]
        cflux = sp.signal.medfilt(oflux[-intran] - np.mean(oflux[-intran]), self.wn)
        fitflux, c = lspolyordern(ctime, cflux, order)
        rflux = np.zeros(length)
        for i in range(order + 1):
            rflux += c[i]*otime**(order-i)
        
        if not noplot:

            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(otime, oflux, '.')
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(otime, rflux, '.', ctime, cflux, 'x')
            plt.show()
        dflux = oflux - rflux
        dflux -= np.mean(dflux) - np.mean(oflux)
        
        return dflux



class KSPDetrend(DetrendFunc):
    # Detrend with high pass filter
    METHOD = 'ksp'
    def __init__(self, penalty_coeff=0.75, bkspace_min=0.75, bkspace_max=1.5, quadfile=''):
        super(KSPDetrend, self).__init__()
        self.required_keys = ['penalty_coeff', 'bkspace_min', 'bkspace_max']
        self.required_cols=['jd', 'rlc']
        self.penalty_coeff = float(penalty_coeff)
        self.bkspace_min = float(bkspace_min)
        self.bkspace_max = float(bkspace_max)
        self.quats = pds.read_csv(quadfile)
        logger.info("USE a high pass filter to detrend, it is configured with %s", self.__str__())
    
    def __str__(self):
        parastr=''
        for key in self.required_keys:
            parastr+="%s=%s, " % (key, getattr(self,key))
        return parastr

    @staticmethod
    def istype(method):
        return method == KSPDetrend.METHOD

    def _detrend(self, jd, mag, data, noplot=True, intran=[]):
        
        index = (self.quats.flag==0) * (~np.isnan(mag)) 
        #import matplotlib 
        #from matplotlib import pyplot as plt
        #plt.plot(jd[index], mag[index], '.')
        #plt.show()

        ksp, kspobj = fit_kepler_spline([jd[index]], [mag[index]], penalty_coeff=self.penalty_coeff,  bkspace_min=self.bkspace_min, bkspace_max = self.bkspace_max)

        dmag = mag*1.0
        dmag[index]-=(ksp[0]-np.nanmedian(mag))
        #import matplotlib 
        #from matplotlib import pyplot as plt
        #plt.plot(jd[index], mag[index], '.')
        #plt.plot(jd[index], dmag[index], '.')
        #plt.show()
        return dmag



class COSDetrend(DetrendFunc):
    # Detrend with high pass filter
    METHOD = 'cos'
    def __init__(self, tmin=1.0, wn=13):
        super(COSDetrend, self).__init__()
        self.required_keys = ['tmin', 'wn']
        self.required_cols=['jd', 'rlc']
        self.tmin = float(tmin)
        self.wn = int(wn)
        logger.info("USE a high pass filter to detrend, it is configured with %s", self.__str__())
    
    def __str__(self):
        parastr=''
        for key in self.required_keys:
            parastr+="%s=%s, " % (key, getattr(self,key))
        return parastr

    @staticmethod
    def istype(method):
        return method == COSDetrend.METHOD

    def _detrend(self, jd, mag, data, noplot=True, intran=[]):
        otime = jd 
        oflux = mag
        length = len(oflux)
        index = (~np.isnan(oflux))
        ctime = otime[index]
        cflux = oflux[index] - np.nanmean(oflux[index])
        cflux = sp.signal.medfilt(cflux, self.wn)

        if not noplot:
            import matplotlib
            from matplotlib import pyplot as plt
            plt.plot(otime, oflux-np.nanmean(oflux[index]), '+')
            plt.plot(ctime, cflux, 'r.')
            plt.xlim(7740,7750)
            plt.show()
        k = (cflux[-1] - cflux[0]) / (ctime[-1] - ctime[0])
        b = cflux[0] - k * ctime[0]
        cflux -= (k * ctime + b)
        e0 = min(ctime)
        timespan = max(ctime) - min(ctime)
        logger.debug("E0=%f, len(cflux)=%d, len(ctime)=%d", e0, len(cflux),len(ctime))
        ncomp = int(round(timespan/self.tmin))
        amatrix = matrixgen(ctime-e0, ncomp, timespan)
        # print A.shape,n
        c, resid, rank, sigma = linalg.lstsq(amatrix, cflux)
        # print resid
        e = np.rollaxis(np.sin(np.array(np.r_['c', 0: ncomp]
                                        * (otime[np.arange(length)]-e0))
                               * math.pi/timespan), 1, 0)
        rflux = np.dot(e, c)
        eflux = k * otime + b
        # print rflux.shape,eflux.shape
        rflux += eflux
        if not noplot:

            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(otime, rflux, '.', otime, eflux, '+', ctime, cflux, 'x')
            plt.show()
        dflux = oflux - rflux
        dflux -= np.nanmean(dflux)
        dflux += np.nanmean(oflux)
        return dflux


class COSDetrendSegment(DetrendFunc):
    # Detrend with high pass filter
    METHOD = 'cosseg'
    def __init__(self, tmin=1.0, wn=13, segstart=['0']):
        super(COSDetrendSegment, self).__init__()
        self.required_keys = ['tmin', 'wn', 'segstart']
        self.tmin = float(tmin)
        self.wn = int(wn)
        self.segstart = np.array(segstart).astype(float)
        logger.info("USE a high pass filter to detrend, it is configured with %s", self.__str__())
    
    def __str__(self):
        parastr=''
        for key in self.required_keys:
            parastr+="%s=%s, " % (key, getattr(self,key))
        return parastr

    @staticmethod
    def istype(method):
        return method == COSDetrendSegment.METHOD


    def _detrend(self, jd, mag, data, noplot=True, seg_length=100*29.4/60./24.):
        # seglength is in units of days, is the length we are use to merge two segment of light curve
        #print len(jd), len(mag)
        dmag = np.zeros(len(jd))+np.nan
        for i in range(len(self.segstart)):
            if i < len(self.segstart) - 1:
                seg = np.where((jd >= self.segstart[i]) & (jd < self.segstart[i+1]))
            else:
                seg = np.where(jd >= self.segstart[i])  # last segment
            if len(mag[seg])<2*self.wn:
                mag_seg = mag[seg]
                jd_seg = jd[seg]
            else:
                mag_seg = self.seg_detrend(jd[seg], mag[seg], noplot=noplot)
                jd_seg = jd[seg]
            if i ==0:
                #if len(mag_seg)>50:
                if np.max(jd_seg) - np.min(jd_seg)>seg_length/2.:
                    head = jd_seg<np.min(jd_seg)+seg_length/10.
                    mag_seg[head] = np.nan
                    tail = jd_seg>np.max(jd_seg)-seg_length/2.
                    mag_seg[tail] = np.nan
            elif i== len(self.segstart)-1:
                #if len(mag_seg)>50:
                mag_seg[:] = np.nan
                dmag[seg] = mag_seg 
                break
            else:    
                #if len(mag_seg)>100:
                if np.max(jd_seg)-np.min(jd_seg)>seg_length:
                    head = jd_seg<np.min(jd_seg)+seg_length/2.
                    mag_seg[head] = np.nan
                    tail = jd_seg>np.max(jd_seg)+seg_length/2.
                    mag_seg[tail] = np.nan

            if i>0:
                last_seg = jd<self.segstart[i]
                if not np.isnan(dmag[last_seg]).all():
                    lastjd = np.max(jd[last_seg][~np.isnan(dmag[last_seg])])
                    lasttail = jd>(lastjd-seg_length) * (jd<lastjd)
                    head = jd_seg<np.min(jd_seg)+seg_length
                    mag_seg += (np.nanmedian(dmag[lasttail]) - np.nanmedian(mag_seg[head]))  # join segments smoothly
            dmag[seg] = mag_seg
            #assert(len(mag_seg) == len(jd[seg]))
        #assert(len(dmag)==len(jd))
        #print len(dmag), np.nanmedian(dmag)
        return dmag



    def seg_detrend(self, jd, mag, noplot=False):
        otime = jd 
        oflux = mag
        length = len(oflux)
        index = (~np.isnan(oflux))
        ctime = otime[index]
        cflux = oflux[index] - np.nanmean(oflux[index])
        cflux = sp.signal.medfilt(cflux, self.wn)
        if len(cflux) ==0:
            return oflux
        noplot = True
        if not noplot:
            import matplotlib
            from matplotlib import pyplot as plt
            plt.plot(otime, oflux-np.nanmean(oflux[index]), '+')
            plt.plot(ctime, cflux, 'r.')
            #plt.xlim(7740,7750)
            plt.show()
        k = (cflux[-1] - cflux[0]) / (ctime[-1] - ctime[0])
        b = cflux[0] - k * ctime[0]
        cflux -= (k * ctime + b)
        e0 = min(ctime)
        timespan = max(ctime) - min(ctime)
        ncomp = int(round(timespan/self.tmin))
        logger.debug("E0=%f, len(cflux)=%d, len(ctime)=%d, ncomp=%d", e0, len(cflux),len(ctime), ncomp)
        #print len(cflux), ncomp
        if len(cflux) < ncomp or ncomp<2:
            return oflux
        amatrix = matrixgen(ctime-e0, ncomp, timespan)
        # print A.shape,n
        c, resid, rank, sigma = linalg.lstsq(amatrix, cflux)
        # print resid
        e = np.rollaxis(np.sin(np.array(np.r_['c', 0: ncomp]
                                        * (otime[np.arange(length)]-e0))
                               * math.pi/timespan), 1, 0)
        rflux = np.dot(e, c)
        eflux = k * otime + b
        # print rflux.shape,eflux.shape
        rflux += eflux
        if not noplot:

            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(otime, rflux, '.', otime, eflux, '+', ctime, cflux, 'x')
            plt.show()
        dflux = oflux - rflux
        dflux -= np.nanmean(dflux)
        dflux += np.nanmean(oflux)
        return dflux




class EPDDetrend(DetrendFunc):
    # Detrend with external parameters
    METHOD = 'epd'
    def __init__(self, level=1, niter=2, lim=5, centroidfile='centroids.txt', pcafile='pca.txt', minimization='metrix'):
        super(EPDDetrend, self).__init__()
        self.required_cols = ['jd', 'rlc']
        self.level = int(level)
        self.niter = int(niter)
        self.lim = 5
        self.centroidfile = centroidfile
        self.pcafile = pcafile
        self.minimization = 'metrix'
        logger.info("USE a EPD method to detrend, it is configured with %s", self.__str__())
        # FIXME: I am assuming centroid file and pca file have exactly the same length as the light curve file.
        centroids = np.loadtxt(self.centroidfile)
        self.x = centroids[:, 1]-np.nanmedian(centroids[:,1])
        self.y = centroids[:, 2]-np.nanmedian(centroids[:,2])
        self.segflag = centroids[:, 3]
        if self.level == 3: 
            self.pca = np.loadtxt(self.pcafile)
        return
    
    @staticmethod
    def istype(method):
        return method == EPDDetrend.METHOD

    def __str__(self):
        return "level=%d, niter=%d, lim=%d, centroidfile=%s, pcafile=%s, minimization=%s" % (self.level, self.niter, self.lim, self.centroidfile, self.pcafile, self.minimization)

    def gen_matrix(self):
        if self.level == 1:
            return np.c_[np.ones(len(self.x[self.mask])), np.sin(2*np.pi*self.x[self.mask]), np.cos(2*np.pi*self.x[self.mask]), np.sin(2*np.pi*self.y[self.mask]), np.cos(2*np.pi*self.y[self.mask])]
        if self.level == 2:
            return np.c_[np.ones(len(self.x[self.mask])), np.sin(2*np.pi*self.x[self.mask]), np.cos(2*np.pi*self.x[self.mask]), np.sin(2*np.pi*self.y[self.mask]), np.cos(2*np.pi*self.y[self.mask]), np.sin(4*np.pi*self.x[self.mask]), np.cos(4*np.pi*self.x[self.mask]), np.sin(4*np.pi*self.y[self.mask]), np.cos(4*np.pi*self.y[self.mask])]
        if self.level == 3: 
            return np.c_[np.ones(len(self.x[self.mask])), np.sin(2*np.pi*self.x[self.mask]), np.cos(2*np.pi*self.x[self.mask]), np.sin(2*np.pi*self.y[self.mask]), np.cos(2*np.pi*self.y[self.mask]), self.pca[self.mask,:]]
        raise NotImplementedError

    def errorfunc(self, c, mag):
        A = self.gen_matrix()
        # print A.shape
        return np.inner(c, A ) - mag
        
    def _detrend(self, jd, mag, data):
        # FIXME: force rlc to be flux 
        data['x'] = self.segflag
        data['y'] = self.y
        mask1 = self.segflag>0
        nrlist=[]
        nrmedianlist=[]
        for nr in list(np.unique(self.segflag[mask1])):
            if nr<0:
                continue
            indexnr = self.segflag==nr
            
            nrlist.append(np.nanmedian(jd[indexnr]))
            nrmedianlist.append(np.nanmedian(mag[indexnr]))
        #print np.min(nrlist)
        print np.max(nrlist)
        nrlist = np.array(nrlist)
        nrmedianlist = np.array(nrmedianlist)
        from scipy.interpolate import UnivariateSpline
        w = None
        s = None
        noplot = False
        for i in xrange(self.niter):
            splseg = UnivariateSpline(nrlist,nrmedianlist,k=5, w=w, s=s)
            stdmag = np.sqrt(2)* np.nanmedian(np.abs(nrmedianlist - splseg(nrlist)))
            logger.debug("first spline fit, iter=%d, stdmag=%f", i, stdmag)
            tempmask = np.abs(nrmedianlist - splseg(nrlist))>self.lim*stdmag
            nrlist = nrlist[~tempmask]
            nrmedianlist = nrmedianlist[~tempmask]
            if len(nrlist) < 10:
                logger.warning("epd failed")
                return mag
            w = np.ones(len(nrlist))*(1./stdmag)
            s = len(w)
            if not noplot:

               import matplotlib
               import matplotlib.pyplot as plt
               fig = plt.figure()
               ax = fig.add_subplot(1, 1, 1)
               ax.plot(jd[mask1], mag[mask1], '.')
               ax.plot(nrlist, nrmedianlist, 'go')
               x = np.linspace(np.min(jd), np.max(jd), 100)
               ax.plot(x, splseg(x), 'r.')
               plt.show()

         
        noplot = False
        if not noplot:

           import matplotlib
           import matplotlib.pyplot as plt
           fig = plt.figure()
           ax = fig.add_subplot(1, 1, 1)
           ax.plot(jd[mask1], mag[mask1], '.')
           ax.plot(nrlist, nrmedianlist, 'go')
           x = np.linspace(np.min(jd), np.max(jd), 100)
           ax.plot(x, splseg(x), 'r.')
           plt.show()

        smoothmag = 1.0*mag 
        smoothmag[mask1] = mag[mask1]- splseg(jd[mask1])
        if not noplot:
           fig = plt.figure()
           ax = fig.add_subplot(1, 1, 1)
           ax.plot(jd[mask1], smoothmag[mask1], '.')
           plt.show()

        mask2 = ~np.isnan(smoothmag)    
        
        w = None
        s = None
        for i in xrange(self.niter):
            splx = UnivariateSpline(self.x[mask1*mask2],smoothmag[mask1*mask2],k=4, w=w, s=s)
            if not noplot:

                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(self.x[mask1*mask2], smoothmag[mask1*mask2], '.')
                x = np.linspace(-0.5, 0.5, 20)
                ax.plot(x, splx(x), 'r.')
                plt.show()
 

            stdmag = np.sqrt(2)* np.nanmedian(np.abs(smoothmag - splx(self.x)))
            logger.debug("second spline fit, iter=%d, stdmag=%f", i, stdmag)
            tempmask = np.abs(smoothmag - splx(self.x))>self.lim*stdmag
            mask2 = mask2*(~tempmask) 
            if len(smoothmag[mask1*mask2]) < 10:
                logger.warning("epd failed")
                return mag
            w = np.ones(len(smoothmag[mask1*mask2]))*(1./stdmag)
            s = len(w)
        intranmask = (jd >17)*(jd<18.2) 
        self.mask = mask1*mask2*~intranmask 
        
        if self.level == 1: 
            c0 = np.zeros(5) + 1.
        elif self.level == 2:
            c0 = np.zeros(9) + 1.
        elif self.level == 3:
            c0 = np.zeros(5+self.pca.shape[1]) + 1.
        else: 
            raise NotImplementedError
        if self.minimization == 'metrix':
            A= self.gen_matrix()
            #print A.shape
            c1, residual, rank, sigma = linalg.lstsq(A, smoothmag[self.mask])
        elif self.minimization == 'leastsq':
            c1, success = leastsq(self.errorfunc, c0, args=smoothmag[self.mask])
        else: 
            raise NotImplementedError
        if not noplot:
           fig = plt.figure()
           ax = fig.add_subplot(1, 1, 1)
           ax.plot(self.x[self.mask], smoothmag[self.mask], '.')
           ax.plot(self.x[self.mask], self.errorfunc(c1, smoothmag[self.mask])+smoothmag[self.mask], '.')
           plt.show()

        self.mask = np.ones(len(mag)).astype(bool) 
        logger.debug("coefficient is %s", str(c1))
        if (c1 == 1).all():
            dmag = mag * 1.0
            logger.warning("epd failed") 
        else:
            dmag = -self.errorfunc(c1, mag[self.mask])
        dmag = dmag - np.nanmedian(dmag) + np.nanmedian(mag)
        return dmag

#--------------------------------New detrend func-----------------------------------

class EPDDetrendSegment(DetrendFunc):
    # Detrend with external parameters
    METHOD = 'epdseg'

    def __init__(self, level=1, niter=2, lim=5, centroidfile='centroids.txt', pcafile='pca.txt',
                 segstart=['0'], minimization='metrix'):
        super(EPDDetrendSegment, self).__init__()
        self.required_cols = ['jd', 'rlc']
        self.required_keys = ['segstart']
        self.level = int(level)
        self.niter = int(niter)
        self.segstart = np.array(segstart).astype(float)
        self.lim = 5
        self.centroidfile = centroidfile
        self.pcafile = pcafile
        self.minimization = 'metrix'
        logger.info("USE a EPD method to detrend in segments, it is configured with %s", self.__str__())
        # FIXME: I am assuming centroid file and pca file have exactly the same length as the light curve file.
        # Centroid file needs to be divided into segments, but there are no dates in them. Should we use jd or just
        # indices as segstart?
        centroids = np.loadtxt(self.centroidfile)
        self.cads = centroids[:,0]
        self.x = centroids[:, 1] - np.nanmedian(centroids[:, 1])
        self.y = centroids[:, 2] - np.nanmedian(centroids[:, 2])
        self.segflag = centroids[:, 3]  # maybe we should add jd to the centroid file
        if self.level == 3:
            self.pca = np.loadtxt(self.pcafile)
        return

    @staticmethod
    def istype(method):
        return method == EPDDetrendSegment.METHOD

    def __str__(self):
        return "level=%d, niter=%d, lim=%d, centroidfile=%s, pcafile=%s, segstart=%s, minimization=%s" % (
        self.level, self.niter, self.lim, self.centroidfile, self.pcafile, str(self.segstart), self.minimization)

    def gen_matrix(self, xcen, ycen, mask, pca):
        if self.level == 1:
            return np.c_[np.ones(len(xcen[mask])), np.sin(2 * np.pi * xcen[mask]), np.cos(
                2 * np.pi * xcen[mask]), np.sin(2 * np.pi * ycen[mask]), np.cos(
                2 * np.pi * ycen[mask])]
        if self.level == 2:
            return np.c_[np.ones(len(xcen[mask])), np.sin(2 * np.pi * xcen[mask]), np.cos(
                2 * np.pi * xcen[mask]), np.sin(2 * np.pi * ycen[mask]), np.cos(
                2 * np.pi * ycen[mask]), np.sin(4 * np.pi * xcen[mask]), np.cos(
                4 * np.pi * xcen[mask]), np.sin(4 * np.pi * ycen[mask]), np.cos(
                4 * np.pi * ycen[mask])]
        if self.level == 3:
            return np.c_[np.ones(len(xcen[mask])), np.sin(2 * np.pi * xcen[mask]), np.cos(
                2 * np.pi * xcen[mask]), np.sin(2 * np.pi * ycen[mask]), np.cos(
                2 * np.pi * ycen[mask]), pca[mask, :]]
        raise NotImplementedError

    def errorfunc(self, c, A):
        # c = coefficients of each row in gen_matrix
        #A = self.gen_matrix(xcen, ycen, mask, pca)
        # print A.shape
        return np.inner(c, A) - mag

    def _detrend(self, jd, mag, data, noplot=True):
        data['x'] = self.segflag
        data['y'] = self.y

        dmag = np.array([])
        for i in range(len(self.segstart)):
            if i < len(self.segstart) - 1:
                seg = np.where((jd >= self.segstart[i]) & (jd < self.segstart[i+1]))
            else:
                seg = np.where(jd >= self.segstart[i])  # last segment

            mag_seg = self.seg_detrend(jd[seg], mag[seg], self.segflag[seg], self.x[seg], self.y[seg], self.pca[seg], noplot)
            if (len(dmag) > 10) and (len(mag_seg) > 10):
                mag_seg += (np.median(dmag[-10:]) - np.median(mag_seg[:10]))  # join segments smoothly
            dmag = np.concatenate((dmag, mag_seg))
            assert(len(mag_seg) == len(jd[seg]))
        assert(len(dmag)==len(jd))
        return dmag

    def seg_detrend(self, jd, mag, segflag, xcen, ycen, pca, noplot=True):
        # FIXME: force rlc to be flux

        nrlist = []
        nrmedianlist = []
        mask1 = segflag > 0  # FIXME: do we throw out the first segment?
        for nr in list(np.unique(segflag[mask1])):
            # if nr < 0:
            #     continue
            indexnr = segflag == nr  # boolean array of indices of current segment

            nrlist.append(np.nanmedian(jd[indexnr]))  # median jd of current seg
            nrmedianlist.append(np.nanmedian(mag[indexnr]))  # median flux of current seg
        logger.debug('JD start: %s. JD end: %s.', np.min(nrlist), np.max(nrlist))

        nrlist = np.array(nrlist)
        nrmedianlist = np.array(nrmedianlist)
        from scipy.interpolate import UnivariateSpline
        w = None
        s = None

        for i in xrange(self.niter):
            splseg = UnivariateSpline(nrlist, nrmedianlist, k=5, w=w, s=s)
            stdmag = np.sqrt(2) * np.nanmedian(np.abs(nrmedianlist - splseg(nrlist)))
            logger.debug("first spline fit, iter=%d, stdmag=%f", i, stdmag)
            tempmask = np.abs(nrmedianlist - splseg(nrlist)) > self.lim * stdmag
            nrlist = nrlist[~tempmask]
            nrmedianlist = nrmedianlist[~tempmask]
            if len(nrlist) < 10:
                logger.warning("epd failed")
                return mag
            w = np.ones(len(nrlist)) #* (1. / stdmag)
            s = len(w)  # FIXME: how do I stop the spline from shooting off to infinity at the ends?
            if not noplot:
                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(jd[mask1], mag[mask1], '.')
                ax.plot(nrlist, nrmedianlist, 'go')
                x = np.linspace(np.min(jd), np.max(jd), 100)
                ax.plot(x, splseg(x), 'r.')
                ax.set_title("first spline fit, iter=%d, stdmag=%f" % (i, stdmag))
                plt.show()

        
        smoothmag = 1.0 * mag
        smoothmag[mask1] = mag[mask1] - splseg(jd[mask1])  # mask1 is pretty much everything?
        # smoothmag = mags after subtracting best-fit function of time
        if not noplot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(jd[mask1], smoothmag[mask1], '.')
            plt.show()

        mask2 = ~np.isnan(smoothmag)

        w = None
        s = None
        for i in xrange(self.niter):
            splx = UnivariateSpline(xcen[mask1 * mask2], smoothmag[mask1 * mask2], k=3, w=w, s=s)
            if not noplot:
                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(xcen[mask1 * mask2], smoothmag[mask1 * mask2], '.')
                ax.set_title('Centroid spline fit, iter=%d' % (i))
                x = np.linspace(min(xcen), max(xcen), 30)
                ax.plot(x, splx(x), 'r.')
                plt.show()

            stdmag = np.sqrt(2) * np.nanmedian(np.abs(smoothmag - splx(xcen)))
            logger.debug("second spline fit, iter=%d, stdmag=%f", i, stdmag)
            tempmask = np.abs(smoothmag - splx(xcen)) > self.lim * stdmag  # sigma clipping
            mask2 = mask2 * (~tempmask)
  
            if len(smoothmag[mask1 * mask2]) < 10:
                logger.warning("epd failed")
                return mag
            w = np.ones(len(smoothmag[mask1 * mask2])) #* (1. / stdmag)
            s = len(w)
        intranmask = (jd > 17) * (jd < 18.2)
        mask = mask1 * mask2 * ~intranmask

        if self.level == 1:
            c0 = np.zeros(5) + 1.
        elif self.level == 2:
            c0 = np.zeros(9) + 1.
        elif self.level == 3:
            c0 = np.zeros(5 + self.pca.shape[1]) + 1.
        else:
            raise NotImplementedError
        if self.minimization == 'metrix':
            A = self.gen_matrix(xcen, ycen, mask, pca)
            # print A.shape
            c1, residual, rank, sigma = linalg.lstsq(A, smoothmag[mask])
        elif self.minimization == 'leastsq':
            c1, success = leastsq(self.errorfunc, c0, args=(smoothmag[mask], xcen, ycen, mask, pca))
        else:
            raise NotImplementedError
        if not noplot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xcen[mask], smoothmag[mask], '.')
            ax.plot(xcen[mask], self.errorfunc(c1, smoothmag[mask], xcen, ycen, mask, pca)+ smoothmag[mask], '.')
            ax.set_title('Least squares minimization')
            plt.show()

        mask = np.ones(len(mag)).astype(bool)
        logger.debug("coefficient is %s", str(c1))
        if (c1 == 1).all():
            dmag = mag * 1.0
            logger.warning("epd failed")
        else:
            dmag = -self.errorfunc(c1, mag[mask], xcen, ycen, mask, pca)
        dmag = dmag - np.nanmedian(dmag) + np.nanmedian(mag)
        return dmag

class EPDQuatDetrend(DetrendFunc):
    # Detrend with external parameters
    METHOD = 'epdq'
    def __init__(self, level=1, niter=2, lim=5,  minimization='metrix', quadstats=None, sigma=1.07, aprad=2.25, quadfile='', quadlim=0.15, ccd=0, cam=0, orbit_id=0, cadence_type=30):
        super(EPDQuatDetrend, self).__init__()
        self.ccd = ccd
        self.cam = cam
        self.orbit_id = str(orbit_id)
        self.cadence_type =str(cadence_type)
        self.required_cols = ['jd', 'rlc', 'x', 'y', 'bg', 'bgerr']
        self.niter = int(niter)
        self.level = int(level)
        self.lim = float(lim)
        self.quadlim = float(quadlim)
        self.quadfile = quadfile
        self.minimization = minimization 
        self.sigma = sigma
        self.aprad = aprad
        logger.info("USE a EPD method and the quarternions to detrend, it is configured with %s", self.__str__())
        # FIXME: I am assuming centroid file and pca file have exactly the same length as the light curve file.
        #self.quats = np.loadtxt(quadfile)
        self.quats = pds.read_csv(quadfile)

        return
    
    @staticmethod
    def istype(method):
        return method == EPDQuatDetrend.METHOD

    def __str__(self):
        return "niter=%d, lim=%d, quadfile=%s, minimization=%s" % (self.niter, self.lim,  self.quadfile, self.minimization)
    
    def compute_overlap(self, d):
        #print self.aprad, self.sigma
        areas  = 2*np.pi*np.exp(-(d**2.+self.aprad**2.)/2./self.sigma)*sp.special.iv(0, -d*self.aprad/self.sigma**2.)
        return areas/2./np.exp(-1./2./self.sigma**2.)



    def gen_matrix(self):
        if self.level==1:
            A = np.c_[np.ones(len(self.x[self.mask])), self.area_ratio[self.mask], self.area_ratio[self.mask]**2., self.area_ratio[self.mask]**3.]
        elif self.level==2:
            A = np.c_[np.ones(len(self.x[self.mask])), (self.x-self.x.astype(int))[self.mask], (self.x-self.x.astype(int))[self.mask]**2., (self.y-self.y.astype(int))[self.mask], (self.y-self.y.astype(int))[self.mask]**2,  self.bg[self.mask]/np.nanmedian(self.bg[self.mask]), np.log10(self.bgerr[self.mask]), self.jd[self.mask], self.jd[self.mask]**2.]
        return A 

    def cal_distance(self):
        
        if self.ccd==0:
            raise ValueError, "need to specify the ccd numbers 1-4 to use quaternions"
        
        from lctools.util.deltq2deltpix import Dq2Dpix
        x0 = np.nanmedian(self.x)
        y0 = np.nanmedian(self.y)
        #print x0, y0, self.ccd
        geometry = Dq2Dpix(self.quadfile)
        gpsquad, dx, dy = geometry(self.ccd, x0, y0)
        distance = (dx**2.+dy**2.)**0.5
        return distance

    def errorfunc(self, c, mag):
        A = self.gen_matrix()
        #print A.shape, len(mag)
        return np.inner(c, A) - mag
        
    def _detrend(self, jd, mag, data):
        self.jd = data["jd"]
        self.x = data["x"]
        self.y = data["y"]
        self.bg = data["bg"]
        self.bgerr = data["bgerr"]
        if self.level==1:
            distance = self.cal_distance() 
            #plt.plot(distance, '.')
            #plt.show()
            gaussian_observed=self.compute_overlap(distance)
            #plt.plot(gaussian_observed, '.')
            #plt.show()
            from lctools.util.binstatistics import BinnedStatistics
            gpsquad = self.quats[:, 0]
            # FIXME
            qlpquery = QlpQuery(dbinfo_file="/pdo/users/xuhuang/.config/qlp/lc-dbinfo")
            realtime = qlpquery.query_frames_by_orbit(self.orbit_id, self.cadence_type, self.cam)
            gpstime = realtime["gps_time"]
            gpsbin = np.array(list(gpstime)+[gpstime[-1]+ float(self.cadence_type)*60.]) 
            #plt.plot(jdquad, gaussian_observed)
            #plt.show()
            gaussian_stats = BinnedStatistics(gpsquad, gaussian_observed, gpsbin)
            #print gaussian_observed
            self.area_ratio = gaussian_stats.average()
            #plt.plot(jd, mag-np.nanmedian(mag), '.')
            #plt.plot(jd, self.area_ratio, '.')
            #plt.show()
        mad = np.nanmedian(np.abs(mag-np.nanmedian(mag)))
        logger.debug("mad=%f" % mad)
        outlier = np.abs(mag-np.nanmedian(mag))>5*mad
        self.mask = ~(np.isnan(mag)+outlier)
        plt.plot(jd, mag, '.')
        plt.plot(jd[self.mask], mag[self.mask],'.')
        plt.show()
        if len(mag[self.mask])<30:
            return np.zeros(len(mag)) + np.nan
        if self.level==1:
            c0 = np.array([np.nanmedian(mag)] + list(np.zeros(3)))
        else:
            c0 = np.array([np.nanmedian(mag)]+list(np.zeros(8)))
        if self.minimization == 'metrix':
            A = self.gen_matrix()
            # print A.shape
            c1, residual, rank, sigma = linalg.lstsq(A, mag[self.mask])
        elif self.minimization == 'leastsq':
            c1, success = leastsq(self.errorfunc, c0, args=(mag[self.mask]))
        else:
            raise NotImplementedError
        self.mask = np.ones(len(mag)).astype(bool)

        dmag = -self.errorfunc(c1, mag[self.mask])
        plt.plot(mag-dmag, '.')
        plt.plot(mag,'.')
        plt.show()
        plt.plot(dmag, '.')
        plt.show()
        return dmag-np.nanmedian(dmag)+np.nanmedian(mag[self.mask])




class GPDetrend(DetrendFunc):
    # note: GP sometimes overfits. Need to run a filter and only do GP on light curves with rms >> expected poisson
    # noise.
    METHOD = 'gp'
    def __init__(self):
        super(GPDetrend, self).__init__()
        self.logger = logging.getLogger(GPDetrend.__name__)
        self.logger.addHandler(logging.NullHandler())
        self.required_cols = ['jd', 'rlc']

    @staticmethod
    def istype(method):
        return method == GPDetrend.METHOD

    @staticmethod
    def get_period(time, flux):
        model = LombScargleFast(fit_offset=True).fit(time, flux)
        _, _ = model.periodogram_auto(oversampling=10)
        model.optimizer.period_range = (0.2, min([30, time[-1]-time[0]]))
        print 'Best-fit period: %s' % (model.best_period)
        logger.info('Best-fit period: %s', model.best_period)
        return model.best_period

    @staticmethod
    def neg_log_like(params, y, gp):
        gp.kernel[:] = params
        ll = gp.lnlikelihood(y, quiet=True)
        # The scipy optimizer doesn't play well with infinities.
        return -ll if np.isfinite(ll) else 1e25

    @staticmethod
    def grad_nll(p, y, gp):
        # Update the kernel parameters and compute the likelihood.
        gp.kernel[:] = p
        return -gp.grad_lnlikelihood(y, quiet=True)

    def _detrend(self, jd, mag, data):
        from gatspy.periodic import LombScargleFast
        import george
        from george.kernels import ExpSquaredKernel, ExpSine2Kernel
        time = jd 
        # FIXME: does this have to be flux or can mag also work?
        flux = mag
        period = self.get_period(time, flux)

        sigma = np.std(flux[:50])
        f_sorted = np.sort(flux)
        amp = f_sorted[-3] - f_sorted[3]
        a, l, G, P = amp / 2, 5, 0.1, period
        kernel = a * ExpSquaredKernel(l) * ExpSine2Kernel(G, P)
        gp = george.GP(kernel, mean=np.mean(flux))
        gp.compute(time, sigma)
        logger.info('Initial ln likelihood: %s', gp.lnlikelihood(flux))

        bounds = ((None, None), (max([1, period]), None), (None, None), (period/2, None))

        initial_params = gp.kernel.vector
        r = minimize(self.neg_log_like, initial_params, method="L-BFGS-B", jac=self.grad_nll, bounds=bounds, args=(flux, gp))
        print r
        gp.kernel[:] = r.x
        pred_mean, var = gp.predict(flux, time)

        sigma = np.std(flux - pred_mean)
        good = np.where(abs(flux - pred_mean) < 2 * sigma)
        gp.compute(time[good], sigma)
        initial_params = gp.kernel.vector
        r = minimize(self.neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, jac=self.grad_nll, args=(flux[good], gp))
        gp.kernel[:] = r.x
        pred_mean, var = gp.predict(flux[good], time)
        logger.info('Final ln likelihood: %s', gp.lnlikelihood(flux[good]))
        print 'Final params:', gp.kernel.vector

        noplot = False
        if not noplot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(time, flux, '.')
            plt.plot(time, pred_mean, lw=2)
            plt.show()
        return flux-pred_mean


class FocusDetrend(DetrendFunc):
    # Detrend with Focus series only
    METHOD = 'focus'
    def __init__(self, wn=13, norder=3):
        super(FocusDetrend, self).__init__()
        self.required_cols = ['cadence', 'rlc']
        self.required_keys = ['wn', 'norder']
        self.norder = norder
        self.wn = wn
        if not cfgfile == '':
            self.config(cfgfile)
        return
    
    @staticmethod
    def istype(method):
        return method == FocusDetrend.METHOD



    def _detrend(self, data):
        xf = ((data['cadence'] / 48. / 13.7 + 0.5) % 1) - 0.5
        mag = data['rlc']
        focus = 0.0 + 10.0 * np.exp(-0.5 * (xf / 0.1)**2.)
        dmag = np.zeros(len(mag)) + mag
        noflux = sp.signal.medfilt(mag, self.wn)
        n = self.norder
        seg1 = data['cadence'] < 624
        x = data['cadence'][seg1]
        xi = sp.zeros(len(x)) + 1
        xii = sp.zeros(len(x)) + focus[seg1]
        amatrix = (x**n)[:, np.newaxis]
        for i in xrange(1, n):
            amatrix = np.hstack((amatrix, (x**(n-i))[:, np.newaxis]))
        amatrix = np.c_[amatrix, xi[:, np.newaxis]]
        amatrix = np.c_[amatrix, xii[:, np.newaxis]]
        c, resid, rank, sigma = sp.linalg.lstsq(amatrix, noflux[seg1])
        z = sp.zeros(len(x))
        for i in xrange(n+1):
            z += c[i] * x**(n-i)
        z += c[-1] * xii
        dmagseg1 = mag[seg1] - z + np.median(mag)
        
        seg2 = data['cadence'] > 624
        x = data['cadence'][seg2]
        xi = sp.zeros(len(x)) + 1
        xii = sp.zeros(len(x)) + focus[seg2]
        amatrix = (x**n)[:, np.newaxis]
        for i in xrange(1, n):
            amatrix = np.hstack((amatrix, (x**(n-i))[:, np.newaxis]))
        amatrix = np.c_[amatrix, xi[:, np.newaxis]]
        amatrix = np.c_[amatrix, xii[:, np.newaxis]]
        c, resid, rank, sigma = sp.linalg.lstsq(amatrix, noflux[seg2])
        z = sp.zeros(len(x))
        for i in xrange(n+1):
            z += c[i] * x**(n-i)
        z += c[-1] * xii
        dmagseg2 = mag[seg2] - z + np.median(mag)
        dmag[seg1] = dmagseg1
        dmag[seg2] = dmagseg2
        return dmag


def lspolyordern(x, y, n):
    x = sp.array(x)
    y = sp.array(y)
    xi = sp.zeros(len(x)) + 1
    amatrix = (x**n)[:, np.newaxis]
    for i in range(1, n):
        amatrix = np.hstack((amatrix, (x**(n-i))[:, np.newaxis]))

    amatrix = np.c_[amatrix, xi[:, np.newaxis]]
    c, resid, rank, sigma = sp.linalg.lstsq(amatrix, y)
    z = sp.zeros(len(x))
    for i in range(n+1):
        z += c[i]*x**(n-i)
    return [z, c]


def matrixgen(time, n, timespan):
    # generate least square fitting matrix with n cos filter,
    # t is the total time span, formulism refer to Huang and Bacos (2012) eq [1]'''
    tn = len(time)
    a = np.rollaxis(np.sin(np.array(np.r_['c', 0:n] * time[np.arange(tn)])
                           * math.pi / timespan), 1, 0)
    return a


def flatfunc(c, ctime, timespan):
    n = len(c) - 1
    b = np.rollaxis(np.sin(np.array(np.r_['c', 0:n] * ctime) * math.pi / timespan),
                    1, 0)
    rflux = np.dot(b, c[0:n])
    rflux -= np.mean(rflux)
    return rflux


def tranfunc(c, ctime, intran, timespan):
    n = len(c)-1
    b = np.rollaxis(np.sin(np.array(np.r_['c', 0: n] * ctime) * math.pi / timespan),
                    1, 0)
    rflux = np.dot(b, c[0: n])
    try:
        rflux[intran] += c[n]
    except TypeError:
        print c
        raise
    rflux -= np.mean(rflux)
    return rflux


def lsfitrecon(otime, oflux, intran, n=30, noplot=True, dipguess=0.0001):
    length = len(oflux)
    ctime = np.array(otime)
    epoch0 = min(ctime)
    timespan = max(ctime)-min(ctime)
    cflux = np.array(oflux)-np.mean(oflux)

    def e(c, time, index, flux, timespan):
        return tranfunc(c, time, index, timespan) - flux

    c0 = list(np.zeros(n))
    c0.append(dipguess)
    c, success = leastsq(e, c0, args=(ctime - epoch0, intran, cflux, timespan), maxfev=10000)
    b = np.rollaxis(np.sin(np.array(np.r_['c', 0:n]
                                    * (ctime[np.arange(length)]-E0))
                           * math.pi/timespan), 1, 0)
    rflux = np.dot(b, c[0:n])
    tran = sp.zeros(len(ctime))
    tran[intran] += c[n]
    tran -= np.mean(tran)
    # print c[n],len(tran[intran])
    if not noplot:

        import matplotlib
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ctime, cflux, 'x', ctime, rflux, '.', ctime, tran, '+')
        plt.show()
        plt.close(fig)
    # rl = len(rflux)
    # nn = len(otime)
    dflux = oflux - rflux + np.mean(oflux)
    return dflux
