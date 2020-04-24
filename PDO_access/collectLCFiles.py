from __future__ import print_function
import os
import numpy as np

def getAstroNetFiles(baseDir, sector, anoPath='ffi/run/', prefix='prediction_', returnPath = False
):
  """
    Returns a list of all astronet output files for a given sector
    e.g. returns ["prediction_cam3ccd1.txt", "prediction_cam3ccd2.txt", ...]
  """
  path = os.path.join(baseDir,sector,anoPath)
  try:
    pathList = os.listdir(path)
  except Exception as e:
    print(e)
    print('Sector '+sector+' has no astroNet Output Files')
    return []

  anoFiles = [item for item in pathList if prefix in item]
  if returnPath:
    return [path+anoFile for anoFile in anoFiles if 'cut' not in anoFile]
  return anoFiles

def getLCFileInfo(baseDir, sector):
  """
    Returns a list of TIC ID, Astronet Score, PDOPath to LC
    for each LC in sector which has an astronetscore
  """
  anoFiles = getAstroNetFiles(baseDir, sector, returnPath=True)

  output = []

  for anoFile in anoFiles:
    pathToData, extra = anoFile.split(sector)
    camInfo = extra.split('prediction_')[1]
    camInfo='/ccd'.join(camInfo.split('ccd'))[:-4]+'/'
    pathToData = os.path.join(pathToData,sector,'ffi/',camInfo,'LC/')

    with open(anoFile,'r') as f:
      lines = f.readlines()
      for line in lines:
        try:
          tic, score = line.strip().split(' ')
        except ValueError:
          print(line)
        output.append([tic, score, pathToData])

  return output

def writeANScoreFiles(baseDir, outDir, sector, anoFileOut='astroNetOutFiles.txt'
):
  """
    Saves the list of all astronet output files for sector
    Creates file outDir+anoFileOut
    Writes output of getAstroNetFiles
  """
  anoFiles = getAstroNetFiles(baseDir, sector, returnPath=True)
  output = os.path.join(outDir,sector)
  try:
    os.mkdir(output)
  except OSError:
    pass
  except FileExistsError:
    pass
  with open(output+anoFileOut,'w') as f:
    for anoFile in anoFiles:
      print(anoFile, file=f)

def parseANO(baseDir, outDir, sector,
  threshold=0.09,
  saveName='filesToPreProc.txt',
  belowThresold='filesBelowThreshold.txt'
):
  """
    for each LC which
      A) has an astronet score
      B) has astronet score above threshold
    writes astronet score, BLSFile path, LCFile Path to saveName
    for LCs which fail:
      Returns a list of strings of "TIC ID, score, path"
  """
  allLCInfo = getLCFileInfo(baseDir, sector)

  failures = []
  outPath = os.path.join(outDir, sector)
  try:
    os.mkdir(outPath)
  except OSError:
    pass
  except FileExistsError:
    pass
  outFile = os.path.join(outPath, saveName)
  with open(outFile, 'w') as f:
    with open(os.path.join(outPath, belowThresold),'w') as btf:
      for lcinfo in allLCInfo:
        try:
          TIC, score, path = lcinfo
        except ValueError:
          print(lcinfo)
          failures.append(' '.join(lcinfo))
          continue
        if np.float(score) > threshold:
          lcfile = path+TIC+'.h5'
          blsfile = path.replace('LC/','BLS/')+TIC+'.blsanal'
          print(lcfile, blsfile, score, file=f)
        else:
          print(lcfile, score, file=btf)
  return failures

def runOnPDO():
  baseDir = '/pdo/qlp-data/'
  outDir  = '/pdo/users/nmehrle/'
  sectors = [item+'/' for item in os.listdir(baseDir) if 'sector' in item]
  sectors = [sector for sector in sectors if 'spoc' not in sector]
