from __future__ import print_function

import sys, os
import numpy as np
# import matplotlib.pyplot as plt
from statsmodels.robust import scale
import collections
import h5py

# From astronet
import median_filter

#-- File Management
def loadFiles(lcfile, blsfile, outfile, overwrite=False):
  outfolder = os.path.dirname(outfile)
  lcfolder  = os.path.dirname(lcfile)
  assert os.path.normpath(outfolder) != os.path.normpath(lcfolder), "Won't overwrite data files"
  
  #check lcname has .h5 extension
  if not lcfile.split('.')[-1] == 'h5':
    raise OSError("LC file doesn't have h5 extension")
  
  # check blsfile exists
  if not os.path.exists(blsfile):
    raise OSError("BLS file doesn't exist")

  if os.path.exists(outfile):
    if overwrite:
      os.remove(outfile)
    else:
      raise OSError('Output file '+outfile+' already exists.')

  h5inputfile = h5py.File(lcfile, 'r')
  blsanal = np.genfromtxt(blsfile, dtype='float', delimiter=' ', names=True)
  h5outfile = h5py.File(outfile, 'w')

  return h5inputfile, blsanal, h5outfile

def unpackBLS(blsanal):
  period  = blsanal['BLS_Period_1_0']
  duration = blsanal["BLS_Qtran_1_0"] * period
  t0 = blsanal["BLS_Tc_1_0"]

  return period, duration, t0

def saveBLS(blsanal, h5outputfile):
  blsgrp = h5outputfile.create_group('BLS Analysis')
  for name in blsanal.dtype.names:
    blsgrp.create_dataset(name, data=blsanal[name])
  return h5outputfile

def loadLC(h5file, apKey, og_time=None, medianCutoff=5):
  if og_time is None:
    og_time = np.array(h5file['LightCurve']['BJD'])

  all_mag  = np.array(h5file["LightCurve"]["AperturePhotometry"][apKey]["KSPMagnitude"])

  real_indices = ~np.isnan(all_mag)
  all_mag  = all_mag[real_indices]
  all_time = og_time[real_indices]
  
  mad           = scale.mad(all_mag)
  valid_indices = np.where(all_mag > np.median(all_mag)-medianCutoff*mad)
  assert len(valid_indices) <= 1, "Need more data points"

  all_mag       = all_mag[valid_indices]
  all_time      = all_time[valid_indices]

  # Convert mag to flux
  all_flux = 10.**(-(all_mag - np.median(all_mag))/2.5)

  return all_flux, all_time
###

#-- Processing LC
def phaseFold(flux, time, period, t0):
  half_period = period/2
  folded_time = np.mod(time + (half_period-t0), period) - half_period
  sorted_idx = np.argsort(folded_time)
  folded_time = folded_time[sorted_idx]
  folded_flux = flux[sorted_idx]

  return folded_flux, folded_time

def genGlobalView(folded_flux, folded_time, period, nbins_global=201):
  bin_width_global = period * 1.2 / nbins_global
  (tmin_global,tmax_global) = (-period / 2, period / 2)
  view, error = median_filter.median_filter(folded_time, folded_flux, nbins_global,
                                      bin_width_global, tmin_global,tmax_global)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_global/2))
  # view.rotate(rotate - minindex)
  return np.array(view), np.array(error)

def genLocalView(folded_flux, folded_time, period, duration, nbins_local=61,
  num_durations=2
):
  bin_width_local = duration * 0.16
  tmin_local = max(-period / 2, -num_durations * duration)
  tmax_local = min(period / 2, num_durations* duration)

  view, error = median_filter.median_filter(folded_time, folded_flux, nbins_local,
                                      bin_width_local, tmin_local,tmax_local)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_local/2))
  # view.rotate(rotate - minindex) # hardcoded assuming nbins_local = 61
  return np.array(view), np.array(error)

def genSecondaryView(folded_flux, folded_time, period, duration, nbins=201):
  bin_width_local = duration * 0.16
  tmin_local = -period/4
  tmax_local = period/4

  view, error = median_filter.median_filter(folded_time, folded_flux, nbins,
                                      bin_width_local, tmin_local,tmax_local)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_local/2))
  # view.rotate(rotate - minindex) # hardcoded assuming nbins_local = 61
  return np.array(view), np.array(error)

def processApertures(h5inputfile, h5outfile, blsanal,
  nbins_global=201,
  nbins_local=61
):
  aps_list = list(h5inputfile["LightCurve"]["AperturePhotometry"].keys())
  globalviews = h5outfile.create_group('GlobalView')
  localviews  = h5outfile.create_group('LocalView')

  globalerrors = globalviews.create_group('Errors')
  localerrors = localviews.create_group('Errors')

  og_time = np.array(h5inputfile["LightCurve"]["BJD"])
  period, duration, t0 = unpackBLS(blsanal)

  for apnum in range(len(aps_list)):
    apKey = "Aperture_%.3d" % apnum

    # Load Data
    all_flux, all_time = loadLC(h5inputfile, apKey, og_time)

    # Phase Fold
    folded_flux, folded_time = phaseFold(all_flux, all_time, period, t0)

    ##############
    # Global & Local view
    ##############
    globalview, gverror = genGlobalView(folded_flux, folded_time, period, nbins_global)
    localview, lverror = genLocalView(folded_flux, folded_time, period, duration, nbins_local)
    
    globalviews.create_dataset(apKey,(nbins_global,), data=globalview)
    globalerrors.create_dataset(apKey,(nbins_global,), data=gverror)

    localviews.create_dataset(apKey,(nbins_local,), data=localview)
    localerrors.create_dataset(apKey,(nbins_local,), data=lverror)

def processEvenOdd(h5inputfile, h5outfile, blsanal, nbins=61):
  evenOdd = h5outfile.create_group('EvenOdd')
  errorGroup = evenOdd.create_group('Errors')

  period, duration, t0 = unpackBLS(blsanal)

  bestAp = "Aperture_%.3d" % h5outfile['bestap'][0]

  flux, time = loadLC(h5inputfile, bestAp)
  folded_flux, folded_time = phaseFold(flux, time, period*2, t0+period/2)
  cut_index = min(range(len(folded_time)), key=lambda i: abs(folded_time[i]))

  evenFlux = folded_flux[:cut_index]
  evenTime = folded_time[:cut_index] + period/2

  oddFlux = folded_flux[cut_index:]
  oddTime = folded_time[cut_index:] - period/2

  even_view, even_error = genLocalView(evenFlux, evenTime, period, duration, nbins)
  odd_view, odd_error  = genLocalView(oddFlux, oddTime, period, duration, nbins)

  evenOdd.create_dataset('Even', (nbins, ), data=even_view)
  errorGroup.create_dataset('Even', (nbins, ), data=even_error)

  evenOdd.create_dataset('Odd', (nbins, ), data=odd_view)
  errorGroup.create_dataset('Odd', (nbins, ), data=odd_error)

def processSecondary(h5inputfile, h5outfile, blsanal, nbins=201):
  # secondary = h5outfile.create_group('Secondary')
  period, duration, t0 = unpackBLS(blsanal)
  secondarygroup = h5outfile.create_group('Secondary')

  bestAp = "Aperture_%.3d" % h5outfile['bestap'][0]

  flux, time = loadLC(h5inputfile, bestAp)
  folded_flux, folded_time = phaseFold(flux, time, period, t0+period/2)
  secondaryView, secondaryError = genSecondaryView(folded_flux, folded_time, period, duration, nbins)

  secondarygroup.create_dataset('Data', (nbins, ), data=secondaryView)
  secondarygroup.create_dataset('Error', (nbins, ), data=secondaryError)
###

def processLC(lcfile, blsfile, outfile,
  score=None,
  overwrite=False,
  verbose=False,
  nbins_global=201,
  nbins_local=61,
  nbins_evenOdd=61,
  nbins_secondary=201
):
  try:
    h5inputfile, blsanal, h5outputfile = loadFiles(lcfile, blsfile, outfile, overwrite=overwrite)
  except Exception as e:
    if verbose:
      print('Error reading in {}'.format(lcfile))
      print(e)
    return -1, e

  best_ap = "Aperture_%.3d" % h5inputfile["LightCurve"]["AperturePhotometry"].attrs['bestap']
  h5outputfile.create_dataset("bestap",(1,), data =  int(best_ap[-3:]))
  
  if score is not None:
    h5outputfile.create_dataset('AstroNetScore', (1,), data=score)

  h5outputfile = saveBLS(blsanal, h5outputfile)

  return_val = 1

  try:
    processApertures(h5inputfile, h5outputfile, blsanal,
      nbins_global=nbins_global,
      nbins_local=nbins_local)
  except Exception as e:
    if verbose:
      print('Error on Process Global/Local View in {}'.format(lcfile))
      print(e)
    return_val = 0

  try:
    processEvenOdd(h5inputfile, h5outputfile, blsanal,
      nbins=nbins_evenOdd)
  except Exception as e:
    if verbose:
      print('Error on Process Even Odd in {}'.format(lcfile))
      print(e)
    return_val = 0

  try:
    processSecondary(h5inputfile, h5outputfile, blsanal,
      nbins=nbins_secondary)
  except Exception as e:
    if verbose:
      print('Error on Process Secondary in {}'.format(lcfile))
      print(e)
    return_val = 0

  h5inputfile.close()
  h5outputfile.close()

  return return_val, 0