import sys, os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust import scale
import collections
import h5py


# From Chelsea
sys.path.append('../Modules')

# From astronet
sys.path.append('../Modules') 
from astronet import median_filter

def loadFiles(lcname, lcfolder, blsfolder, outfolder, overwrite=False):
  assert os.path.normpath(outfolder) != os.path.normpath(lcfolder), "Won't overwrite data files"

  blsname = lcname.replace('h5','blsanal')

  lcfile = os.path.join(lcfolder, lcname)
  blsfile = os.path.join(blsfolder, blsname)

  #check lcname has .h5 extension
  if not lcname.split('.')[-1] == 'h5':
    raise("LC file doesn't have h5 extension")
  
  # check blsfile exists
  if not os.path.exists(blsfile):
    raise("BLS file doesn't exist")

  outfile = os.path.join(outfolder, lcname)
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

def getLC(h5file, apKey, og_time=None, medianCutoff=5):
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
  view  = median_filter.median_filter(folded_time, folded_flux, nbins_global, \
                                      bin_width_global, tmin_global,tmax_global)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_global/2))
  # view.rotate(rotate - minindex)
  return np.array(view)

def genLocalView(folded_flux, folded_time, period, duration, nbins_local=61,
  num_durations=2):
  bin_width_local = duration * 0.16
  tmin_local = max(-period / 2, -num_durations * duration)
  tmax_local = min(period / 2, num_durations* duration)

  view  = median_filter.median_filter(folded_time, folded_flux, nbins_local, \
                                      bin_width_local, tmin_local,tmax_local)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_local/2))
  # view.rotate(rotate - minindex) # hardcoded assuming nbins_local = 61
  return np.array(view)

def genSecondaryView(folded_flux, folded_time, period, duration, nbins=201):
  bin_width_local = duration * 0.16
  tmin_local = -period/4
  tmax_local = period/4

  view  = median_filter.median_filter(folded_time, folded_flux, nbins, \
                                      bin_width_local, tmin_local,tmax_local)

  # Center about zero flux
  view -= np.median(view)

  # Shift bins so bin with minimum flux is centered
  # view = collections.deque(view)
  # minindex = np.argmin(view)
  # rotate = int(np.floor(nbins_local/2))
  # view.rotate(rotate - minindex) # hardcoded assuming nbins_local = 61
  return np.array(view)

def processApertures(h5inputfile, h5outfile, blsanal,
  nbins_global=201,
  nbins_local=61
):
  aps_list = list(h5inputfile["LightCurve"]["AperturePhotometry"].keys())
  globalviews = h5outfile.create_group('GlobalView')
  localviews  = h5outfile.create_group('LocalView')

  og_time = np.array(h5inputfile["LightCurve"]["BJD"])
  period, duration, t0 = unpackBLS(blsanal)

  for apnum in range(len(aps_list)):
    apKey = "Aperture_%.3d" % apnum

    # Load Data
    all_flux, all_time = getLC(h5inputfile, apKey, og_time)

    # Phase Fold
    folded_flux, folded_time = phaseFold(all_flux, all_time, period, t0)

    ##############
    # Global & Local view
    ##############
    globalview = genGlobalView(folded_flux, folded_time, period, nbins_global)
    localview  = genLocalView(folded_flux, folded_time, period, duration, nbins_local)
    
    globalviews.create_dataset(apKey,(nbins_global,), data=globalview)
    localviews.create_dataset(apKey,(nbins_local,), data=localview)

def processEvenOdd(h5inputfile, h5outfile, blsanal, nbins=61):
  evenOdd = h5outfile.create_group('EvenOdd')
  period, duration, t0 = unpackBLS(blsanal)

  bestAp = "Aperture_%.3d" % h5outfile['bestap'][0]

  flux, time = getLC(h5inputfile, bestAp)
  folded_flux, folded_time = phaseFold(flux, time, period*2, t0+period/2)
  cut_index = min(range(len(folded_time)), key=lambda i: abs(folded_time[i]))

  evenFlux = folded_flux[:cut_index]
  evenTime = folded_time[:cut_index] + period/2

  oddFlux = folded_flux[cut_index:]
  oddTime = folded_time[cut_index:] - period/2

  even_view = genLocalView(evenFlux, evenTime, period, duration, nbins)
  odd_view  = genLocalView(oddFlux, oddTime, period, duration, nbins)

  evenOdd.create_dataset('Even', (nbins, ), data=even_view)
  evenOdd.create_dataset('Odd', (nbins, ), data=odd_view)

def processSecondary(h5inputfile, h5outfile, blsanal, nbins=201):
  # secondary = h5outfile.create_group('Secondary')
  period, duration, t0 = unpackBLS(blsanal)

  bestAp = "Aperture_%.3d" % h5outfile['bestap'][0]

  flux, time = getLC(h5inputfile, bestAp)
  folded_flux, folded_time = phaseFold(flux, time, period, t0+period/2)
  secondaryView = genSecondaryView(folded_flux, folded_time, period, duration, nbins)

  h5outfile.create_dataset('Secondary', (nbins, ), data=secondaryView)

def processLC(lcname, lcfolder, blsfolder, outputfolder,
  overwrite=True,
  nbins_global=201,
  nbins_local=61,
  nbins_evenOdd=61,
  nbins_secondary=201
):
  try:
    h5inputfile, blsanal, h5outputfile = loadFiles(lcname, lcfolder, blsfolder, outputfolder, overwrite=True)
  except Exception as e:
    print('Error reading in {}'.format(os.path.join(lcfolder,lcname)))
    print(e)
    return -1

  best_ap = "Aperture_%.3d" % h5inputfile["LightCurve"]["AperturePhotometry"].attrs['bestap']
  h5outputfile.create_dataset("bestap",(1,), data =  int(best_ap[-3:]))
  return_val = 1

  try:
    processApertures(h5inputfile, h5outputfile, blsanal,
      nbins_global=nbins_global,
      nbins_local=nbins_local)
  except Exception as e:
    print('Error on Process Global/Local View in {}'.format(os.path.join(lcfolder,lcname)))
    print(e)
    return_val = 0

  try:
    processEvenOdd(h5inputfile, h5outputfile, blsanal,
      nbins=nbins_evenOdd)
  except Exception as e:
    print('Error on Process Even Odd in {}'.format(os.path.join(lcfolder,lcname)))
    print(e)
    return_val = 0

  try:
    processSecondary(h5inputfile, h5outputfile, blsanal,
      nbins=nbins_secondary)
  except Exception as e:
    print('Error on Process Secondary in {}'.format(os.path.join(lcfolder,lcname)))
    print(e)
    return_val = 0

  h5inputfile.close()
  h5outputfile.close()

  return return_val

def main():
  # Bins for single period phase folded
  nbins_global    = 201
  nbins_local     = 61
  nbins_evenOdd   = 61
  nbins_secondary = 201

  # Select data folder containing light curves and .blsanal
  lcfolder  = "../Data/2020_03_26_TestData/LC/"
  blsfolder = "../Data/2020_03_26_TestData/BLS/"

  # Select folder where binned lightcurves are saved. Lightcurve names are the same as input
  outputfolder = "./test/"

  # Loop through all files in LC folder
  allfiles = os.listdir(lcfolder)
  ngood    = 0

  errors = []
  fatal  = []

  for i, lcname in enumerate(allfiles):
    print("{} / {}\r".format(i,len(allfiles)-1),end="")
    
    val = processLC(lcname, lcfolder, blsfolder, outputfolder)
    if val == 1:
      ngood+=1
    elif val == 0:
      errors.append(lcname)
    elif val == -1:
      fatal.append(lcname)
      os.remove(os.path.join(outputfolder, lcname))

  print('Processed {} lightcurves'.format(len(allfiles)))
  print('  --  {} Done successfully'.format(ngood))
  print('  --  {} Done partially'.format(len(errors)))
  print('  --  {} failed'.format(len(fatal)))