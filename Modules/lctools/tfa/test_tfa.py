#!/usr/bin/env python
import numpy as np
import scipy as sp
from numpy import random
import matplotlib
from matplotlib import pyplot as plt
from tfa import TFA
import lctools
from lctools.util.dataio import readcolumn
import os

def rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1], )

    return np.std(np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides), axis=1)

def get_stats(mag):
    mean = np.nanmean(mag)
    median = np.nanmedian(mag)
    meanstddev = np.nanstd(mag)
    mmd = np.nanmedian(np.abs(mag-median))
    rollingrms = np.nanmedian(rolling_window(mag, 13)/np.sqrt(13))
    #rollingrms = np.nanmedian(rolling_window(mag, 195)/np.sqrt(195))
    
    return [meanstddev, mmd, rollingrms]



class FakeLightcurve(object):
    def __init__(self, jd, mag):
        self.jd = jd
        self.mag = mag


def genfake(length, baseint):
    base = [np.sin(np.arange(length)/1.), np.sin(np.arange(length)/2.), np.sin(np.arange(length)/4.), np.sin(np.arange(length)/8.), np.sin(np.arange(length)/10.),np.cos(np.arange(length)/1.), np.cos(np.arange(length)/2.), np.cos(np.arange(length)/4.), np.cos(np.arange(length)/8.), np.cos(np.arange(length)/10.)]
    coeffs = np.random.rand(len(base))
    fakemag = np.zeros(length)
    fakemag=base[baseint]*coeffs[baseint]
    return fakemag
def testtfa():
    ntrend = 10 
    trendjdlist = []
    for i in xrange(ntrend):
        lc=FakeLightcurve(np.arange(100), 0.005*np.random.randn(100)+np.random.random()*6.+9.)
        #lc=FakeLightcurve(np.arange(100), 0.005*genfake(100, i))
        #plt.plot(lc.jd, lc.mag)
        #plt.show()
        trendjdlist.append(lc)
    targetlc = FakeLightcurve(np.arange(100), np.random.randn(100)*0.001)
    coeffs = np.random.randn(10)
    for i in xrange(ntrend):
        targetlc.mag+=coeffs[i]*trendjdlist[i].mag
    print coeffs/coeffs[0]
    targetlc.mag-=np.mean(targetlc.mag)
    print np.nanstd(targetlc.mag)
    #model = np.sin(np.arange(100)*1.0/5.)*0.002
    model = np.zeros(100)
    model[30:40]+=0.002
    targetlc.mag+=np.random.random()*6.+9.+model
    plt.plot(targetlc.jd, targetlc.mag, '.')
    plt.plot(targetlc.jd, model+np.nanmean(targetlc.mag), '.')
    plt.title("Target")
    plt.show()
    trendlcx = np.ones(ntrend)
    trendlcy = np.ones(ntrend)
    tfaengin =  TFA(trendjdlist, trendlcx, trendlcy)
    tfaengin.call(targetlc, 5., 5.)
    plt.plot(targetlc.jd, targetlc.mag, '.')
    plt.plot(targetlc.jd, model+np.nanmean(targetlc.mag), '.')
    print np.nanstd(targetlc.mag-model)
    plt.title("Target TFA")
    plt.show()
    plt.imshow(tfaengin.oldtemplate)
    plt.show()
    return

def testtfa_tess():
    #indir = "/scratch/xuhuang/ramp/orbit-6149-30min/cam4/ccd1_d70500_ccd1_FC05-32-2b-211-C6/asciiLC/"
    #outdir = "/scratch/xuhuang/ramp/orbit-6149-30min/cam4/ccd1_d70500_ccd1_FC05-32-2b-211-C6/TFALC/"
    #indir = "/scratch/xuhuang/ramp/orbit-6149-30min/cam4/ccd1/asciiLC/"
    #outdir = "/scratch/xuhuang/ramp/orbit-6149-30min/cam4/ccd1/TFALC/"
    indir = "/scratch/xuhuang/ramp/orbit-6150-30min/cam4/ccd1/asciiLC/"
    outdir = "/scratch/xuhuang/ramp/orbit-6150-30min/cam4/ccd1/TFALC/"
    #indir = "/scratch/xuhuang/ramp/orbit-6149-2m/cam4/ccd1/asciiLC/"
    #outdir = "/scratch/xuhuang/ramp/orbit-6149-2m/cam4/ccd1/TFALC/"
    #inlist = "2m/tfainputlist"
    inlist = "30m/tfainputlist"
   
    tid =[]; readcolumn(tid, 1, inlist); tid = np.array(tid) 
    tmags =[]; readcolumn(tmags, 4, inlist); tmags = np.array(tmags)
    tlcx =[]; readcolumn(tlcx, 5, inlist); tlcx = np.array(tlcx) 
    tlcy =[]; readcolumn(tlcy, 6, inlist); tlcy = np.array(tlcy)
    #infile = "394657354.rlc"
    trendjdlist = []
    #trendfile = "2m/trendlist"
    trendfile = "30m/trendlist"

    trendid =[]; readcolumn(trendid, 1, trendfile); trendid = np.array(trendid) 
    trendlcx =[]; readcolumn(trendlcx, 5, trendfile); trendlcx = np.array(trendlcx) 
    trendlcy =[]; readcolumn(trendlcy, 6, trendfile); trendlcy = np.array(trendlcy)
    lcx = []
    lcy = []
    #badjds = np.array([144,152,153,156,165,171,207])
    for j in xrange(len(trendid)):
        trendlc = os.path.join(indir, "%d.rlc" % (int(trendid[j])))
        tjd = []; readcolumn(tjd, 2, os.path.join(indir, trendlc)); tjd = np.array(tjd)
        tmag = []; readcolumn(tmag, 75, os.path.join(indir, trendlc)); tmag = np.array(tmag)
        #tmag = []; readcolumn(tmag, 26, os.path.join(indir, trendlc)); tmag = np.array(tmag)
        #index = tjd>100
        #index = tjd>1405
        index = tjd<103
        tjd = tjd[index]
        tmag = tmag[index]
        #for jd in badjds:
        #    badcadence = tjd == jd
        #    tmag[badcadence] = np.nan
        index = np.isnan(tmag)
        
        tjd = tjd[~index]
        tmag = tmag[~index]
        print len(tmag)
        #if len(tmag)>55:
        if len(tmag)>60:
            trendjdlist.append(FakeLightcurve(tjd, tmag))
            lcx.append(trendlcx[j])
            lcy.append(trendlcy[j])
    #trendlcy = np.ones(ntrend)
    tfaengin =  TFA(trendjdlist, lcx, lcy)
    frms = open("rms_tfa.txt", mode='w')
    for i in xrange(len(tid)):
        infile = os.path.join(indir, "%d.rlc" % (int(tid[i])))
        if os.path.exists(infile):
            jd = []; readcolumn(jd, 2, os.path.join(indir, infile)); jd = np.array(jd)
            mag = []; readcolumn(mag, 75, os.path.join(indir, infile)); mag = np.array(mag)
            #mag = []; readcolumn(mag, 26, os.path.join(indir, infile)); mag = np.array(mag)
            #index = jd>100
            index = jd<103
            jd = jd[index]
            mag = mag[index]
            #for j in badjds:
            #    badcadence = jd == j
            #    mag[badcadence] = np.nan
            
            index = np.isnan(mag)
            jd = jd[~index]
            mag = mag[~index]
            oldmag = mag*1.0
            #x = 60.8; y = 52.8
            targetlc = FakeLightcurve(jd, mag)
            try:
                tfaengin.call(targetlc, tlcx[i],  tlcy[i])
            except ValueError:
                continue
            outfile = os.path.join(outdir, "%d.tfalc" % (int(tid[i])))
            fout = open(outfile, mode='w')
            for j in xrange(len(targetlc.jd)):
                fout.write("%f %f\n" %( targetlc.jd[j], targetlc.mag[j]))
            fout.close()
            if len(targetlc.mag)<30:
                continue
            rms, mad, cdpp = get_stats(targetlc.mag)
            frms.write("%d %f %f %f %f\n" % (int(tid[i]), tmags[i], rms, mad, cdpp))
            #break
        
    frms.close()
    #plt.plot(targetlc.jd,  oldmag, '.')
    #plt.plot(targetlc.jd, targetlc.mag, '.')
    #print np.nanstd(targetlc.mag-model)
    #plt.title("Target TFA")
    #plt.show()
    return



if __name__ == '__main__':
    #testtfa()
    testtfa_tess()
