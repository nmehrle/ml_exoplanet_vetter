#!/usr/bin/env python
import numpy as np
import scipy as sp
from scipy import linalg 
class TFA(object):

    def __init__(self, trendlclist, trendlcx, trendlcy, minsep=5, sigmaclipping=5):
        self.trendlcx = trendlcx
        self.trendlcy = trendlcy
        self.minsep = int(minsep)
        self.sigmaclipping = int(sigmaclipping)
        self.templc = np.zeros([len(trendlclist[0].jd), len(trendlclist) ])
        for i in xrange(len(trendlclist)):
            self.templc[:, i] = self.zero_lc_average(trendlclist[i].mag)
            
        self.oldtemplate, self.oldtemplate_inv = self.compose_template(np.ones(len(trendlclist)).astype(bool)) 
        return
   
    def compose_template(self, index):
        # construct a template matrix using the template lightcurves

        return [self.templc[:, index], sp.linalg.pinv2(self.templc[:, index])] 

    def zero_lc_average(self, mag):
        copymag = mag*1.0
        avmag = np.zeros(len(copymag))
        magstd = np.nanstd(copymag)      
        outlier = np.abs(copymag-np.nanmean(copymag))>self.sigmaclipping*magstd
        copymag[outlier] = np.nan
        print magstd, len(avmag[~outlier])
        avmag[~outlier] = copymag[~outlier]-np.nanmean(copymag) 
        return avmag

    def select_template(self, x, y):
        
        dist = (self.trendlcx-x)**2. + (self.trendlcy-y)**2. 
        #print "dist=", dist
        if (dist>self.minsep**2.).all():
            return False 
        else:
            self.newtemplate, self.newtemplate_inv=self.compose_template(dist>self.minsep**2.)
            return True

    def call(self, lc, x, y):
        base = self.zero_lc_average(lc.mag)
        changeflag = self.select_template(x, y)
        if changeflag:
            coeffs = np.dot(self.newtemplate_inv, base)
            print coeffs/coeffs[0]
            lc.mag-=np.dot(self.newtemplate, coeffs)
        else:
            coeffs = np.dot(self.oldtemplate_inv, base)
            #print np.sum(lc.mag-base), np.mean(lc.mag)
            #coeffs, res, rank, s = sp.linalg.lstsq(self.oldtemplate, base)
            
            print coeffs/coeffs[0]
            lc.mag-=np.dot(self.oldtemplate, coeffs) 
        return 
