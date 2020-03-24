#!/usr/bin/env python
# 
# Copyright (C) 2017 - Massachusetts Institute of Technology (MIT) 
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 
# 
# This program is distributed in the hope that it will be useful, 
# but ITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details. 
# 
# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 


import os
import time
import commands
import logging
import sys
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
from lctools.util.util import getext, mad 
from lctools.util.dataio import readtableline 
from lctools.util.configurable import ConfigurableObject

# create logger
logger = logging.getLogger(__name__)

class VtBls(ConfigurableObject):
    LENGTH = 20
    config_keys = ["IOSettings", "BLS"]
    """
    A python class that configure and run the vartools BLS algorithm.
    """
    def __init__(self, f0=0.0625, f1=2.0, fn=100000, qmin=0.008, qmax=0.08, nbin=200, peaknum=3, bls_dir='', analfile='.blsanal'):
        super(VtBls, self).__init__()
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.fn = int(fn)
        self.qmin = float(qmin)
        self.qmax = float(qmax)
        self.nbin = int(nbin)
        self.peaknum = int(peaknum)
        self.outdir = bls_dir
        self.analfile = analfile
        logger.debug("readfrom configure file, f0=%f, f1=%f, fn=%d, qmin=%f, qmax=%f, nbin=%d, peaknum=%d, outdir=%s, analfile=%s" % (self.f0, self.f1, self.fn, self.qmin, self.qmax, self.nbin, self.peaknum, self.outdir, self.analfile))
        return
    def get_blsanalfile(self, infile):
        blsanalf = self.outdir+self.analfile
        blsanalfile = getext(infile, blsanalf, '')
        return blsanalfile
    
    def bin_SR(self, x, y, binbounds):
            bin_index = np.searchsorted(x, binbounds, side='right')
            split_quat_coeffs = np.split(y, bin_index)[1:-1]

            split_time_bins = np.split(x, bin_index)[1:-1]
            median_bin_y = map(np.nanmedian, split_quat_coeffs)
            center_bin_x = np.zeros(len(binbounds)-1)
            for i in xrange(len(center_bin_x)):
                center_bin_x[i] = (binbounds[i]*binbounds[i+1])**0.5
            return [median_bin_y, center_bin_x]
 

    def __call__(self, lcfile, replace=False):
        if self.outdir == '':
            outdir = os.path.dirname(lcfile.name) + '/'
        else:
            outdir = self.outdir
        blsanalf = os.path.join(outdir, self.analfile)
        blsanalfile = getext(lcfile.name, blsanalf, '')
        blspath = os.path.dirname(blsanalfile)
        logger.debug("outdir=%s, blsanalf=%s, blsanalfile=%s, blspath=%s" % (outdir, blsanalf, blsanalfile, blspath))
        if blspath == "":
            blspath = "./"
        modelpath = blspath
        phasepath = blspath
        if os.path.exists(blsanalfile) and (not replace):
            logger.warning("File %s exists, do not allow to replace, set -r to force overwrite" % blsanalfile)
            return
        elif (not os.path.exists(lcfile.name)):
            logger.warning("Input file %s do not exists, do nothing" % lcfile.name)
            return

        else:
            cmdline = "vartools -i %s -header " \
                      "-readformat 1 %d %d 3 " \
                      "-BLS q %f %f %f %f %d %d %d %d 1 %s 0 0 " \
                      "fittrap nobinnedrms stepP" \
                      % (lcfile.name, lcfile.cols['jd'], lcfile.cols['ltflc'],
                         self.qmin, self.qmax, 1./self.f1, 1./self.f0, self.fn,
                         self.nbin, 0, self.peaknum, blspath)
            logger.info("excuting command line %s", cmdline)
            # print cmdline
            status, output = commands.getstatusoutput(cmdline)
            blsfile = os.path.basename(lcfile.name) + '.bls' 
            blsfile = os.path.join(blspath, blsfile)
            if not os.path.exists(blsfile):
                return
            blsspec = np.loadtxt(blsfile)
            period = blsspec[:, 0]
            power = blsspec[:, 1]
                
            
            # print output
            header, result = output.split('\n')
            newheader = " ".join(header.split()[1:VtBls.LENGTH+1])
            newlines = ""
            for i in xrange(self.peaknum):
                newlines += "%d " % (i + 1)
                newlines += " ".join(result.split()[1 + i * VtBls.LENGTH:(1 + VtBls.LENGTH * (i + 1))]) + "\n"
            fout = open(blsanalfile, mode="w")
            fout.write("# BLS_No ")
            fout.write(newheader)
            fout.write("\n")
            fout.write(newlines.rstrip())
            fout.close()
            candparam = readtableline(blsanalfile) 
            #import matplotlib
            #from matplotlib import pyplot as plt
            #plt.plot(period, power, '.')
            # FIXME:
            #print min(period), max(period)
            power = power[np.argsort(period)]
            period = np.sort(period)
            index = period<np.max(period)
            power = power[index]
            period = period[index]
            perbin = np.logspace(np.log10(min(period)), np.log10(max(period)), 30)
            #print perbin
            powbin, perbincenter = self.bin_SR(period, power, perbin) 
            #plt.plot(perbincenter, powbin, '.')
            #plt.show()
            fpow = interp1d(perbincenter, powbin, bounds_error=False, fill_value='extrapolate')
            lowfqpower = fpow(period)
            # Remove baseline
            highfqpower = power- lowfqpower
            #plt.plot(period, highfqpower, '.')
            #plt.show()
            peakperiod = candparam["Period"]
            #print peakperiod
            if peakperiod < period[0]:
                peakSR = power[0]
            elif peakperiod > period[-1]:
                peakSR = power[-1]
            else:
                peakSR = interp1d(period, power)(peakperiod)
            #print peakSR, mad(highfqpower), np.nanmedian(highfqpower)
            #print highfqpower-np.nanmedian(highfqpower)
            signaltonoise = (peakSR-fpow(peakperiod))/mad(highfqpower)*0.67
            candparam["SN"] = signaltonoise
            candparam.pop("No", None)
            import collections
            od = collections.OrderedDict(sorted(candparam.items()))
            fout = open(blsanalfile, mode="w")
            line1 = "# BLS_NO "
            line2 = "1 " 
            for key, value in od.iteritems():
                
                line1+="BLS_%s_1_0 " % (key)
                line2+="%f " % value
            line1+="\n"
            line2+="\n"
            fout.write(line1)
            fout.write(line2)
            fout.close()

            return
                

