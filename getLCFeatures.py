from __future__ import print_function
import sys, os
import numpy as np
from scipy.optimize import curve_fit
import h5py
import pickle

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
  full_duration  = np.maximum(1, full_duration)
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
    guess = [np.min(data), len(data)/2, len(data)/10, max(2,len(data)/10 - 10), 0]
  else:
    guess = [np.min(data), len(data)/2, len(data)/3, max(2,len(data)/3 - 10), 0]

  popt,_ = curve_fit(trapezoid, x, data, guess)

  return x, popt

def getDepth(data, view='local'):
  x, p = fitTrap(data, view=view)
  depth = p[0]
  error = np.std(data - trapezoid(x,*p))

  return depth,error

def processLCFile(lcfilename):
  # Features:
  #   [-1] label
  #   [0] astronet score
  #   [1] depth best ap - 1
  #   [2] depth best ap (global)
  #   [3] depth best ap + 1 or best ap
  #   [4] error best ap - 1
  #   [5] error best ap (global)
  #   [6] error best ap + 1
  # 
  #   [7] depth best ap - 1
  #   [8] depth best ap (local)
  #   [9] depth best ap + 1 or best ap
  #   [10] error best ap - 1
  #   [11] error best ap (local)
  #   [12] error best ap + 1
  # 
  #   [13] depth even
  #   [14] depth odd
  #   [15] error even
  #   [16] error odd

  #   [17] depth secondary
  #   [18] error seconday

  #   [19] TIC ID
  #   [20] Tess Magnitude
  #   [21] Stellar Log g
  #   [22] Stellar Mass
  #   [23] Stellar Rad
  #   [24] Stellar Teff
  lcfile = h5py.File(lcfilename,'r')
  data = np.zeros(25)

  data[0] = lcfile['AstroNetScore'][0]
  bestApNum = lcfile['bestap'][0]

  for i in [-1,0,1]:
    apnum = bestApNum + i

    d_loc,e_loc = getDepth(lcfile['LocalView'][getApKey(apnum)])
    d_glo,e_glo = getDepth(lcfile['GlobalView'][getApKey(apnum)])

    # record depths
    data[2+i] = d_glo
    data[8+i] = d_loc

    #record error
    data[5+i] = e_glo
    data[11+i] = e_loc

  d_even,e_even = getDepth(lcfile['EvenOdd']['Even'])
  d_odd,e_odd = getDepth(lcfile['EvenOdd']['Odd'])
  d_sec, e_sec = getDepth(lcfile['Secondary'])

  data[13] = d_even
  data[14] = d_odd
  data[15] = e_even
  data[16] = e_odd
  data[17] = d_sec
  data[18] = e_sec

  stellarParams = lcfile['Stellar Params']
  data[19] = stellarParams['id'][0]
  data[20] = stellarParams['tmag'][0]
  data[21] = stellarParams['logg'][0]
  data[22] = stellarParams['mass'][0]
  data[23] = stellarParams['rad'][0]
  data[24] = stellarParams['teff'][0]

  lcfile.close()
  return np.array(data)

def sortOutApertures(group, bestap):
  n = np.shape(group[getApKey(bestap)])[0]
  organizedData = np.zeros((3, n))

  if bestap == 0:
    organizedData[0] = group[getApKey(0)]
    organizedData[1] = group[getApKey(1)]
    organizedData[2] = group[getApKey(2)]
  elif bestap == 4:
    organizedData[0] = group[getApKey(2)]
    organizedData[1] = group[getApKey(3)]
    organizedData[2] = group[getApKey(4)]
  else:
    organizedData[0] = group[getApKey(bestap - 1)]
    organizedData[1] = group[getApKey(bestap    )]
    organizedData[2] = group[getApKey(bestap + 1)]

  return organizedData

def quickLCFILE(lcfilename):
  data = h5py.File(lcfilename, 'r')
  ret = {}

  bestap = data['bestap'][0]

  ret['LocalView'] = sortOutApertures(data['LocalView'], bestap)
  ret['GlobalView'] = sortOutApertures(data['GlobalView'], bestap)

  ret['LocalDepths'] = np.array([getDepth(ret['LocalView'][i], 'local')[0] for i in range(3)])
  ret['GlobalDepths'] = np.array([getDepth(ret['GlobalView'][i], 'global')[0] for i in range(3)])

  return ret

def loadPlanetLabels(dataPath, labelsFile):
  labels_tsv = np.genfromtxt(os.path.join(dataPath,labelsFile), delimiter="\t",skip_header=3,usecols=(0),dtype="i8,S5",names=["id"])
  planetLabels = []
  for each in labels_tsv['id']:
    planetLabels.append(str(each))

  return planetLabels

def processSector(dataDir, sector, planetLabels, processFN=processLCFile,
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
      label = -1

    try:
      lc_data['label'] = label
    except (IndexError, TypeError):
      lc_data = np.append(lc_data, label)

    sector_data.append(lc_data)
    nsuccess+=1
  return sector_data, failed_tics, nsuccess, nfail

def processAllSectors(sectors, labelsFile, dataDir,
  processFN=processLCFile,
  subpath='preprocessed', output='lcFeatures'
):
  planetLabels = loadPlanetLabels(dataDir, labelsFile)

  for sector in sectors:
    try:
      sector_data, failed_tics, nsuccess, nfail = processSector(dataDir, sector,planetLabels, processFN, subpath)
    except TypeError:
      continue

    print('Loaded {:4d} files from {}'.format(nsuccess, sector))
    print('Omitting {} files.'.format(nfail))
    print('')
    print('')

    with open(os.path.join(dataDir, sector, output+'.pickle'), 'wb') as f:
      pickle.dump(sector_data, f)
    with open(os.path.join(dataDir, sector, 'TICSFailedFeatures.pickle'),'wb') as f:
      pickle.dump(failed_tics,f)

def removeDuplicates(dataPath, sectors,
  subpath='preprocessed/', duplicatePath='duplicates/'
):
  # Sort sectors in descending order
  sector_nums = [int(sector.split('-')[1]) for sector in sectors]
  sorted_idx = np.argsort(sector_nums)
  sorted_sectors = list(np.array(sectors)[sorted_idx])[::-1]

  foundTics    = []
  foundSectors = []

  fileCount = 0
  duplicateCount = 0

  for sector in tqdm(sorted_sectors):
    # Path to lightcurves
    lcPath = os.path.join(dataPath, sector, subpath)

    # Path to duplicate Lightcurves
    duplicatePath = os.path.join(lcPath, 'duplicates/')
    if not os.path.exists(duplicatePath):
      os.mkdir(duplicatePath)

    for lcfile in os.listdir(lcPath):
      # get TIC ID from lightcurve files
      if '.h5' not in lcfile:
        continue
      tic = lcfile.split('.')[0]

      if tic in foundTics:
        # This TIC ID has been found in a later sector
        # Move the lightcurve file to the duplicate folder
        idx = np.where(np.array(foundTics) == tic)[0][0]
        srcName = os.path.join(lcPath,lcfile)
        destName = os.path.join(duplicatePath, f'{tic}_{foundSectors[idx]}.h5')
        os.rename(srcName, destName)
        duplicateCount+=1
      else:
        # New TIC ID, log that it was found
        foundTics.append(tic)
        foundSectors.append(sector)
      fileCount+=1

  print(f'{fileCount} files found, {duplicateCount} duplicates, {fileCount-duplicateCount} originals')

# # for blender
# dataPath = 'Data/'
# labelsFile = 'labels.tsv'
# sectors = []
# for i in range(1,23):
#   sectors.append('sector-'+str(i))

# processAllSectors(sectors, labelsFile, dataPath)
