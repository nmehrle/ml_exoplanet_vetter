from __future__ import print_function
import sys, os
import numpy as np
from scipy.optimize import curve_fit
import h5py

def getApKey(apnum):
  return "Aperture_%.3d" % apnum

def closest_idx(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def trapezoid(x, depth, t0, total_duration, full_duration):
  y = np.zeros(len(x))
  t1 = t0 - total_duration/2.0
  t2 = t0 - full_duration/2.0
  t3 = t0 + full_duration/2.0
  t4 = t0 + total_duration/2.0
  t_idx = [closest_idx(x,t) for t in [t1,t2,t3,t4]]
  
  slope = -depth * 2 / (total_duration - full_duration)
  ingress = (x-t1) * -slope
  egress  = (x-t3) * slope + depth
  y[t_idx[0]:t_idx[1]] = np.minimum(0,ingress[t_idx[0]:t_idx[1]])
  y[t_idx[1]:t_idx[2]] = depth
  y[t_idx[2]:t_idx[3]] = np.maximum(depth,egress[t_idx[2]:t_idx[3]])
  
  return y

def fitTrap(data, view='local'):
  x = np.arange(len(data))

  if view == 'local':
    guess = [np.min(data), len(data)/2, len(data)/3, max(2,len(data)/3 - 10)]
  elif view == 'global':
    guess = [np.min(data), len(data)/2, len(data)/10, max(2,len(data)/10 - 10)]
  else:
    guess = [np.min(data), len(data)/2, len(data)/3, max(2,len(data)/3 - 10)]

  popt,_ = curve_fit(trapezoid, x, data, guess)

  return x, popt

def getDepth(data, view='local'):
  x, p = fitTrap(data, view)
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
  data = np.zeros(19)

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
  data[19] = stellarParams['id']
  data[20] = stellarParams['tmag']
  data[21] = stellarParams['logg']
  data[22] = stellarParams['mass']
  data[23] = stellarParams['rad']
  data[24] = stellarParams['teff']

  lcfile.close()
  return np.array(data)

def loadPlanetLabels(dataPath, labelsFile):
  labels_tsv = np.genfromtxt(os.path.join(dataPath,labelsFile), delimiter="\t",skip_header=3,usecols=(0),dtype="i8,S5",names=["id"])
  planetLabels = []
  for each in labels_tsv['id']:
    planetLabels.append(str(each))

  return planetLabels

def processSector(sectorPath, processedTICIDs, planetLabels):
  nsuccess = 0
  nomit   = 0
  sector_data = []

  for i,lcfilename in enumerate(os.listdir(sectorPath)):
    print("{} / {}\r".format(i, len(os.listdir(sectorPath))-1), end="")
    # skip if already processed
    if lcfilename in processedTICIDs:
      continue
    # skip if not lc file
    if lcfilename.split('.')[-1] != 'h5':
      continue

    try:
      lc_data = processLCFile(os.path.join(sectorPath,lcfilename))
    except:
      nomit+=1
      continue

    ticID = lcfilename.split('.')[0]
    if ticID in planetLabels:
      label = 1
    else:
      label = -1

    lc_data = np.append(lc_data, label)

    sector_data.append(lc_data)
    processedTICIDs.append(lcfilename)
    nsuccess+=1

  return np.array(sector_data), processedTICIDs, nsuccess, nomit

def processAllSectors(sectors, labelsFile, dataPath, subpath='preprocessed', output='lcFeatures'):
  processedTICIDs = []
  planetLabels = loadPlanetLabels(dataPath, labelsFile)

  for i in range(len(sectors)-1,-1,-1):
    sector = sectors[i]

    print(sector, end='')
    print('  ----')

    sectorPath = os.path.join(dataPath,sector, subpath)
    assert os.path.exists(sectorPath), "{} data does not exist.".format(sector)

    sector_data, processedTICIDs, nsuccess, nomit = processSector(sectorPath, processedTICIDs, planetLabels)

    print('Loaded {:4d} files from {}'.format(nsuccess, sectorPath))
    print('Omitting {} files.'.format(nomit))
    print('')
    print('')

    np.save(os.path.join(dataPath, sector, output), sector_data)

# for blender
dataPath = 'Data/'
labelsFile = 'labels.tsv'
sectors = []
for i in range(1,23):
  sectors.append('sector-'+str(i))

processAllSectors(sectors, labelsFile, dataPath)
