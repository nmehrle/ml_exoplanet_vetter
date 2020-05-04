from __future__ import print_function
import numpy as np
import os,sys
import socket
from tqdm import tqdm

pdo = socket.gethostname()
print(pdo)

def loadPlanetInfo(pathtolabels):
  raw = np.loadtxt(pathtolabels, dtype=str, skiprows=2,delimiter='\t')
  data = raw[1:]

  TICID = data[:,0]
  source = data[:,16]
  sectors_strs = data[:,-5]
  
  sectors = []
  for sectors_str in sectors_strs:
    float_sectors = []
    if len(sectors_str.strip()) == 0:
      sectors.append(None)
      continue
    for sector in sectors_str.strip().split(','):
      try:
        if '"' in sector:
          try:
            float_sec = int(sector.strip()[1:])
          except ValueError:
            float_sec = int(sector.strip()[:-1])
        else:
          float_sec = int(sector.strip())
      except Exception as e:
        print(e)
      float_sectors.append(float_sec)
    float_sectors.sort()
    sectors.append(float_sectors[-1])
  return TICID, source, sectors

def getNumUniqueSystems(pathtolabels):
  tics, _, _ = loadPlanetInfo(pathtolabels)
  return len(np.unique(tics))

def findInProcessed(tic):
  baseDir='/pdo/users/nmehrle/'
  sectorFiles = os.listdir(baseDir)
  sectorFiles = [sector for sector in sectorFiles if 'sector' in sector]

  for sector in sectorFiles:
    sectorPath = os.path.join(baseDir, sector, 'preprocessed/')
    if os.path.exists(os.path.join(sectorPath,tic+'.h5')):
      return sector
  return False

def findInRaw(tic):
  baseDir = '/pdo/qlp-data/'
  sectors = os.listdir(baseDir)
  sectors = [sector for sector in sectors if 'sector' in sector]

  cams = ['cam1','cam2','cam3','cam4']
  ccds = ['ccd1', 'ccd2', 'ccd3', 'ccd4']
  
  for sector in sectors:
    for cam in cams:
      for ccd in ccds:
        datapath = os.path.join(baseDir, sector, 'ffi/',cam,ccd,'LC/')
        if os.path.exists(datapath+tic+'.h5'):
          return sector,cam,ccd
  return False

def whatsMissing(prevFound, labelFile):
  tics, _, _ = loadPlanetInfo(labelFile)
  found = [[],[]]
  skipped = [[],[]]
  missing = []

  seenTics = []

  for tic in tqdm(tics):
    if tic in prevFound[0]:
      continue

    if tic in seenTics:
      continue

    seenTics.append(tic)

    fp = findInProcessed(tic)
    if fp is not False:
      found[0].append(tic)
      found[1].append(str(fp))
      # print(tic, 'found')
    else:
      fr = findInRaw(tic)
      if fr is not False:
        skipped[0].append(tic)
        skipped[1].append(str(fr))
        # print(tic, 'skipped')
      else:
        # Not in Processsed or Raw
        missing.append(tic)
        # print(tisc, 'missing')
  return found, skipped, missing

def getPrev(dataFile):
  try:
    with open(dataFile,'r') as f:
      lines = f.readlines()
  except IOError:
    return [[],[]],[[],[]],[]

  found = [[],[]]
  skipped = [[],[]]
  missing = []
  for line in lines:
    line = line.strip()
    status = line.split(' ')[0]
    tic = line.split(' ')[1]
    info = ' '.join(line.split(' ')[2:])
    
    if status == 'Found':
      found[0].append(tic)
      found[1].append(info)
    elif status == 'Skipped':
      skipped[0].append(tic)
      skipped[1].append(info)
    else:
      missing.append(tic)

  return found, skipped, missing

def accountForPlanets():
  labelFile = 'labels.tsv'
  dataFile  = 'foundPlanets.txt'

  prevFound, prevSkipped, prevMissing = getPrev(dataFile)
  found, skipped, missing = whatsMissing(prevFound,labelFile)
  for tic,info in np.transpose(prevFound):
    if tic in found[0]:
      print(tic)
    else:
      found[0].append(tic)
      found[1].append(info)

  newSkipped = [[],[]]
  for tic,info in np.transpose(np.hstack((skipped,prevSkipped))):
    if tic in found[0]:
      continue
    elif tic in newSkipped[0]:
      continue
    else:
      newSkipped[0].append(tic)
      newSkipped[1].append(info)
  skipped = newSkipped

  newMissing = []
  for tic in np.concatenate((missing,prevMissing)):
    if tic in found[0]:
      continue
    elif tic in skipped[0]:
      continue
    elif tic in newMissing:
      continue
    else:
      newMissing.append(tic)
  missing = newMissing

  nSystemsFound = len(found[0]) + len(skipped[0]) + len(missing)
  if nSystemsFound != getNumUniqueSystems('labels.tsv'):
    # raise RunTimeError('number of systems is wrong, something has gone awry')
    print('number of systems changed something is wrong :(')

  with open(dataFile,'w') as f:
    for i in range(len(found[0])):
      print('Found', found[0][i], found[1][i], file=f)

    for i in range(len(skipped[0])):
      print('Skipped', skipped[0][i], skipped[1][i], file=f)

    for i in range(len(missing)):
      print('Missing', missing[i], ' ', file=f)

def findAstroNetScores():
  found, skipped, missing = getPrev('foundPlanets.txt')
  tics = skipped[0]
  info = skipped[1]
  reasons = []
  reasonCount=[0, 0, 0]

  for i in tqdm(range(len(tics))):
    tic = tics[i]
    sector, cam, ccd = info[i].replace("'","").strip('\n)(').split(', ')

    astroNetPath = os.path.join('/pdo/qlp-data/',sector,'ffi/run/')
    astroNetFile = 'prediction_'+cam+ccd+'.txt'
    fullAstroNetFile = os.path.join(astroNetPath,astroNetFile)
    if not os.path.exists(fullAstroNetFile):
      reasons.append('AstroNetFile DNE')
      reasonCount[0] += 1
      continue

    foundInAN = False

    with open(fullAstroNetFile,'r') as f:
      lines = f.readlines()
    for line in lines:
      seenTic, score = line.split(' ')
      if seenTic == tic:
        reasons.append('AN Score: '+str(score))
        foundInAN = True
        reasonCount[2] += 1
        break

    if foundInAN:
      continue
    else:
      reasons.append('TIC not in AN File')
      reasonCount[1] += 1

  with open('reasons.txt','w') as f:
    for i in range(len(tics)):
      print('Skipped', skipped[0][i], skipped[1][i], reasons[i], file=f)

  print(len(tics), ' Skipped Files')
  print(reasonCount[0], ' AstroNetFile DNE')
  print(reasonCount[1], ' TIC not in AN File')
  print(reasonCount[2], ' Score found')

def main():
  mode = sys.argv[1]
  if mode == 'run':
    accountForPlanets()
  elif mode == 'reason':
    findAstroNetScores()

if __name__ == "__main__":
  main()