#!/usr/bin/env python
# under construction
from dataio import *
import numpy as np
import scipy as sp
import os
import shutil
import random
import tarfile

def mad(data):
    # median absolute deviation of 1D array
    return np.nanmedian(abs(data - np.nanmedian(data)))

def rect_sechmod(t, b, t0, w, a0, a1):
    """Fits a sech model with linear detrending of background.
    INPUTS:
        t - times in days
        b - 2*transit depth
        t0 - mid transit time
        w - width of transit in days
    """
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    return 1 + b / (np.exp(-(t-t0) ** 2./w ** 2.) + np.exp((t-t0)**2./w**2.))

def getext_base(example,ext):
        if not(example==''):
                return os.path.basename(example)

        else:
                return ext

def getext(example,ext,result):
        if(result==''):
                assert(not example=='')
                return os.path.dirname(ext)+'/'+os.path.splitext(os.path.basename(example))[0]+os.path.splitext(os.path.basename(ext))[0]
        else:
                return result

def gps_to_tjd(gpstime):
    gpstime0 = 1212946200.25 
    tjd0 = 1283.229762
    tjd = (gpstime - gpstime0)/24./3600. + tjd0
    return tjd
   
def get_default_keys(cam=0, ccd=0, planetno=1, indir='', orbit_id=''):
    if cam==0 or ccd ==0:
        default_dict = dict({'inlist':'', 'coljd':1, 'colmag':2, 'indir':''})
    else:
        phot_dir = os.path.join(indir, "cam%d/ccd%d/phot" % (cam, ccd))
        # FIXME: we might want this to go to a cache directory at some point 
        ascii_dir = os.path.join(indir, "cam%d/ccd%d/asciiLC/" % (cam, ccd))
        #catfile = os.path.join(indir, "cam%d/ccd%d/astrom/catalog_full.txt" % (cam, ccd))
        catfile = os.path.join(indir, "catalog_%s_%d_%d_full.txt" % (orbit_id, cam, ccd))
        rmt1_dir = os.path.join(indir, "cam%d/ccd%d/ITER%d" % (cam, ccd, planetno))
        if planetno==1:
            lc_dir = os.path.join(indir, "cam%d/ccd%d/LC" % (cam, ccd))
            bls_dir = os.path.join(indir, "cam%d/ccd%d/BLS" % (cam, ccd))
            dft_dir = os.path.join(indir, "cam%d/ccd%d/DFT" % (cam, ccd))
            blsanal_sum = os.path.join(indir, "cam%d/ccd%d/BLS/blsanal_sum.txt" % (cam, ccd))
            bestapfile = os.path.join(indir, "cam%d/ccd%d/bestap" % (cam, ccd))
            bgphot_dir = os.path.join(indir, "cam%d/ccd%d/subphot" % (cam, ccd))
            bg_dir = os.path.join(indir, "cam%d/ccd%d/subLC" % (cam, ccd))
            h5_dir = lc_dir
        elif planetno>1:
            #indir = os.path.join(indir, "ITER%d" % (planetno-1))
            lc_dir = os.path.join(indir, "cam%d/ccd%d/ITER%d/LC" % (cam, ccd, planetno-1))
            bls_dir = os.path.join(indir, "cam%d/ccd%d/ITER%d/BLS" % (cam, ccd, planetno-1))
            dft_dir = os.path.join(indir, "cam%d/ccd%d/ITER%d/DFT" % (cam, ccd, planetno-1))
            blsanal_sum = os.path.join(indir, "cam%d/ccd%d/ITER%d/BLS/blsanal_sum.txt" % (cam, ccd, planetno-1))
            bestapfile = os.path.join(indir, "cam%d/ccd%d/ITER%d/bestap" % (cam, ccd, planetno-1))
            bgphot_dir = os.path.join(indir, "cam%d/ccd%d/subphot" % (cam, ccd))
            bg_dir = os.path.join(indir, "cam%d/ccd%d/subLC" % (cam, ccd))
            h5_dir = lc_dir

        default_dict = dict({'inlist':'', 'coljd':1, 'colmag':2, 'indir':'',  'lc_dir':lc_dir, 'bls_dir':bls_dir, 'dft_dir':dft_dir, 'h5_dir':h5_dir, 'phot_dir':phot_dir, 'ascii_dir':ascii_dir, 'blsanal_sum':blsanal_sum, 'Camera': cam, 'CCD': ccd, 'bestapfile': bestapfile, 'catfile':catfile, 'rmt1_dir': rmt1_dir, 'bg_dir':bg_dir, 'bgphot_dir':bgphot_dir})
    return default_dict




