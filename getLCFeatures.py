from __future__ import print_function
import sys, os
import numpy as np
from scipy.optimize import curve_fit
import h5py
import pickle

from functools import partial
import multiprocessing as mp

def type_of_script():
  try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
      return 'jupyter'
    if 'terminal' in ipy_str:
      return 'ipython'
  except:
    return 'terminal'

if type_of_script() == 'jupyter':
  from tqdm import tqdm_notebook as tqdm
else:
  from tqdm import tqdm

def getApKey(apnum):
  return "Aperture_%.3d" % apnum

def closest_idx(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def trapezoid(x, depth, t0, total_duration, full_duration, baseline):
  total_duration = np.maximum(1, total_duration)
  full_duration  = np.maximum(0, full_duration)
  y = np.ones(len(x)) * baseline
  t1 = t0 - total_duration/2.0
  t2 = t0 - full_duration/2.0
  t3 = t0 + full_duration/2.0
  t4 = t0 + total_duration/2.0
  t_idx = [closest_idx(x,t) for t in [t1,t2,t3,t4]]
  
  slope = -depth * 2 / (total_duration - full_duration)
  ingress = (x-t1) * -slope
  egress  = (x-t3) * slope + depth
  y[t_idx[0]:t_idx[1]] = np.minimum(0,ingress[t_idx[0]:t_idx[1]]) + baseline
  y[t_idx[1]:t_idx[2]] = depth + baseline
  y[t_idx[2]:t_idx[3]] = np.maximum(depth,egress[t_idx[2]:t_idx[3]]) + baseline
  
  return y

def fitTrap(data, errors=None, view='local'):
  x = np.arange(len(data))

  if view == 'global':
    guess = [np.min(data), len(data)/2, len(data)/10, 2, 0]
  else:
    guess = [np.min(data), len(data)/2, len(data)/4, 2, 0]

  if errors is None:
    popt, pcov = curve_fit(trapezoid, x, data, guess)
  else:
    popt, pcov = curve_fit(trapezoid, x, data, guess,
      sigma=errors, absolute_sigma=True)

  return x, popt, pcov

def getDepth(data, errors=None, view='local'):
  x, p, cov = fitTrap(data, errors=errors, view=view)
  depth = p[0]
  error = np.sqrt(np.diag(cov))[0]

  if error == np.inf:
    transitRange = np.round([p[1] - p[3]/2, p[1] + p[3]/2])
    lower = int(transitRange[0])
    upper = int(transitRange[1])

    if lower == upper:
      lower = lower - 1
      upper = upper + 1

    inTransit = data[lower: upper+1]
    error = np.std(inTransit)/np.sqrt(len(inTransit))

  return depth, error

def sortOutApertures(group, bestap, view):
  n = np.shape(group[getApKey(bestap)])[0]
  organizedData = np.zeros((3, n))
  depths = np.zeros(3)
  depthErrors = np.zeros(3)

  if bestap == 0:
    apnums = [0,1,2]
  elif bestap == 4:
    apnums = [2,3,4]
  else:
    apnums = [bestap-1, bestap, bestap+1]

  for i in range(3):
    data  = group[getApKey(apnums[i])]
    error = group['Errors'][getApKey(apnums[i])]

    depth, depthError = getDepth(data, error, view=view)

    depths[i] = depth
    depthErrors[i] = depthError
    organizedData[i] = data

  return organizedData, depths, depthErrors

def processAuxillary(data):
  evenOdd = np.zeros((2,61))
  eoDepths = np.zeros(2)
  eoDepthErrors = np.zeros(2)
  keys = ['Even','Odd']
  for i in range(2):
    lc = data['EvenOdd'][keys[i]]
    error = data['EvenOdd']['Errors'][keys[i]]
    depth, depthError = getDepth(lc, error, 'local')

    evenOdd[i] = lc
    eoDepths[i] = depth
    eoDepthErrors[i] = depthError

  secondary = np.array(data['Secondary']['Data'])
  secondaryError = np.array(data['Secondary']['Error'])
  secondaryDepth, secondaryDepthError = getDepth(secondary, secondaryError, 'global')

  return evenOdd, eoDepths, eoDepthErrors, secondary, secondaryDepth, secondaryDepthError

def processLCFile(lcfilename):
  data = h5py.File(lcfilename, 'r')
  ret = {}

  bestap = data['bestap'][0]

  localData, localDepths, localDepthErrors = sortOutApertures(data['LocalView'], bestap, 'local')
  globalData, globalDepths, globalDepthErrors = sortOutApertures(data['GlobalView'], bestap, 'global')

  ret['LocalView'] = localData
  ret['LocalDepths'] = localDepths
  ret['LocalDepthErrors'] = localDepthErrors

  ret['GlobalView'] = globalData
  ret['GlobalDepths'] = globalDepths
  ret['GlobalDepthErrors'] = globalDepthErrors

  
  evenOdd, eoDepths, eoDepthErrors, secondary, secondaryDepth, secondaryDepthError = processAuxillary(data)

  ret['EvenOdd'] = evenOdd
  ret['EvenOddDepths'] = eoDepths
  ret['EvenOddDepthErrors'] = eoDepthErrors

  ret['Secondary'] = secondary
  ret['SecondaryDepth'] = secondaryDepth
  ret['SecondaryDepthError'] = secondaryDepthError

  ret['AstroNetScore'] = float(data['AstroNetScore'][0])
  ret['StellarParams'] = {}
  for param in ['logg','mass','rad','teff','tmag']:
    ret['StellarParams'][param] = float(data['Stellar Params'][param][0])

  data.close()
  return ret

def loadPlanetLabels(dataPath, labelsFile):
  labels_tsv = np.genfromtxt(os.path.join(dataPath,labelsFile), delimiter="\t",skip_header=3,usecols=(0),dtype="i8,S5",names=["id"])
  planetLabels = []
  for each in labels_tsv['id']:
    planetLabels.append(str(each))

  return planetLabels

def getSectorData(dataDir, sector, planetLabels, processFN=processLCFile,
  subpath='preprocessed/'
):
  nsuccess = 0
  nfail = 0
  failed_tics = []
  sector_data = []

  fulldatapath = os.path.join(dataDir, sector, subpath)
  if not os.path.exists(fulldatapath):
    print("{} data does not exist.".format(sector))
    return

  for i, lcfilename in enumerate(tqdm(os.listdir(fulldatapath), desc=sector)):
    try:
      ticID, extension = lcfilename.split('.')
      if extension != 'h5':
        continue
    except ValueError:
      continue

    try:
      lc_data = processFN(os.path.join(fulldatapath,lcfilename))
    except Exception as e:
      failed_tics.append(ticID)
      nfail+=1
      continue

    if ticID in planetLabels:
      label = 1
    else:
      label = 0

    try:
      lc_data['label'] = label
    except (IndexError, TypeError):
      lc_data = np.append(lc_data, label)

    sector_data.append(lc_data)
    nsuccess+=1
  return sector_data, failed_tics, nsuccess, nfail

def processSector(i, sectors, dataDir, planetLabels,
  processFN=processLCFile, subpath='preprocessed', outputFolder='lcFeatures/'
):
  # sector has '/' at end
  sector = sectors[i]
  sector_data, failed_tics, nsuccess, nfail = getSectorData(dataDir, sector, planetLabels, processFN, subpath)

  with open(os.path.join(dataDir, outputFolder, sector[:-1]+'.pickle'), 'wb') as f:
      pickle.dump(sector_data, f)
  with open(os.path.join(dataDir, sector, 'TICSFailedFeatures.txt'),'w') as f:
      for line in failed_tics:
        print(line, file=f)

def processAllSectors(dataDir, sectors, cores, labelsFile='labels.tsv',
  processFN=processLCFile, subpath='preprocessed', outputFolder='lcFeatures/'
):
  planetLabels = loadPlanetLabels(dataDir, labelsFile)

  func = partial(processSector,
          sectors=sectors,
          dataDir=dataDir,
          planetLabels=planetLabels,
          processFN=processFN,
          subpath=subpath,
          outputFolder=outputFolder)

  pbar = tqdm(total=len(sectors), desc='Sector')
  pool = mp.Pool(processes=cores)
  seq = pool.imap_unordered(func, range(len(sectors)))
  for i in seq:
    pbar.update()
  pbar.close()
