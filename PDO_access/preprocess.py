from __future__ import print_function
from collectLCFiles import writeANScoreFiles, parseANO
from processLC import processLC
import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from datetime import datetime

def parseFileList(lines, sector, outputFolder, limitTo=None, verbose=False):
  ngood  = 0
  errors = []
  fatal  = []
  prev   = []
  fatal_msgs = []
  for i, line in enumerate(tqdm(lines)):
    lcfile, blsfile, score = line.strip().split(' ')
    if limitTo is not None:
      if lcfile not in limitTo:
        continue
        
    lcname = lcfile.split('/')[-1]
    outfile = os.path.join(outputFolder, lcname)
    if os.path.exists(outfile):
      prev.append(lcfile)
      continue

    val, e = processLC(lcfile, blsfile, outfile, score, verbose=verbose)
    if val == 1:
      ngood+=1
    elif val == 0:
      errors.append(lcfile)
    elif val == -1:
      fatal.append(lcfile)
      fatal_msgs.append(e)

  return ngood, errors, fatal, prev, fatal_msgs

def saveFatal(fileName, fatal, fatal_msgs):
  with open(fileName,'w') as f:
    for i in range(len(fatal)):
      print(fatal[i], '----', fatal_msgs[i], file=f)

def runPreprocess(baseDir, outDir, sector, threshold=0.09, verbose=False):
  print(sector)
  print('--')
  fatalParse = parseANO(baseDir, outDir, sector, threshold=threshold)
  fatal_msgs = ["Error Reading Astronet Output"]*len(fatalParse)

  fileList = os.path.join(outDir, sector, 'filesToPreProc.txt')

  with open(fileList,'r') as f:
    lines = f.readlines()

  outputFolder = os.path.join(outDir, sector,'preprocessed/')
  try:
    os.mkdir(outputFolder)
  except OSError:
    pass
  
  ngood, errors, fatal, prev, parse_errors = parseFileList(lines, sector, outputFolder,verbose=verbose)
  fatal = np.hstack((fatalParse,fatal))
  fatal_msgs = np.hstack((fatal_msgs, parse_errors))

  print('Processed {} lightcurves'.format(len(lines)))
  print('  --  {} Done successfully'.format(ngood))
  print('  --  {} already done'.format(len(prev)))
  print('  --  {} Done partially'.format(len(errors)))
  print('  --  {} failed'.format(len(fatal)))

  with open(os.path.join(outDir, sector, 'preproc_error.txt'),'w') as f:
    for each in errors:
      print(each, file=f)

  saveFatal(os.path.join(outDir, sector, 'preproc_fatal.txt'), fatal, fatal_msgs)

def getStellarParamsSector(outDir, sector):
  try:
    from qlp.util.gaia import GaiaCatalog
    import tsig
    from tsig import catalog
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError('GetStellarParams must be run on PDO')

  print(sector)
  print('--')
  outputFolder = os.path.join(outDir, sector,'preprocessed/')
  fileList = os.listdir(outputFolder)

  c = tsig.catalog.TIC()
  gaia = GaiaCatalog()
  field_list = ["id", "tmag", "mass", "rad", "teff", "logg", "ra", "dec", "jmag", "kmag", "pmra", "pmdec", "plx"]

  for i, fileName in enumerate(tqdm(fileList)):
    stellarParams = {}

    ticID = fileName.split(".")[0]
    result, _ = c.query_by_id(ticID, field_list=",".join(field_list))
    for i, key in enumerate(field_list):
      stellarParams[key] = result[0][i]

    stellarParams['gaia_b_r'] = np.nan
    
    try:
      if stellarParams['rad'] is None or np.isnan(stellarParams['rad']):
        gaiaResult = gaia.query_by_loc(stellarParams['ra'], stellarParams['dec'], 0.02, stellarParams['tmag'])
        if not gaiaResult is None:
          stellarParams['rad']      = float(gaiaResult["radius_val"])
          stellarParams['teff']     = float(gaiaResult["teff_val"])
          stellarParams['gaia_b_r'] = float(gaiaResult["phot_bp_mean_mag"]) - float(gaiaResult["phot_rp_mean_mag"])
          stellarParams['pmra']     = float(gaiaResult["pmra"])
          stellarParams['pmdec']    = float(gaiaResult["pmdec"])
          stellarParams['plx']      = float(gaiaResult["parallax"])
    except Exception as e:
      print(stellarParams['rad'], type(stellarParams['rad']))
      print(ticID, e)
      pass

    outFile = h5py.File(os.path.join(outputFolder,fileName),'r+')
    stellarParamsGroup = outFile.require_group("Stellar Params")

    for k,v in stellarParams.items():
      try:
        stellarParamsGroup.create_dataset(k, (1,), data=v)
      except RuntimeError:
        del stellarParamsGroup[k]
        stellarParamsGroup.create_dataset(k, (1,), data=v)

    outFile.close()
  print('')
  print('')

def rerunFatalSector(outDir, sector, verbose=False):
  print(sector)
  print('--')

  fullfileList = os.path.join(outDir, sector, 'filesToPreProc.txt')
  with open(fullfileList,'r') as f:
    lines = f.readlines()

  fatalFiles = os.path.join(outDir, sector, 'preproc_fatal.txt')
  with open(fatalFiles,'r') as f:
    rawFatalLines = f.readlines()

  fatalLines  = []
  for fl in rawFatalLines:
    try:
      line = fl.split('----')[0].strip()
      if '_CH' in line:
        line = ''.join(line.split('_CH'))
      fatalLines.append(line)
    except IndexError:
      fatalLines.append(fl.strip())

  print('Rerunning {} failed lightcurves'.format(len(fatalLines)))

  outputFolder = os.path.join(outDir, sector,'preprocessed/')
  ngood, errors, fatal, prev, parse_errors = parseFileList(lines, sector, outputFolder, fatalLines, verbose=verbose)
  print('Processed {} lightcurves'.format(len(fatalLines)))
  print('  --  {} Done successfully'.format(ngood))
  print('  --  {} already done'.format(len(prev)))
  print('  --  {} Done partially'.format(len(errors)))
  print('  --  {} failed'.format(len(fatal)))
  print('')
  print('')

  saveFatal(fatalFiles, fatal, parse_errors)

  return ngood

# def removeDuplicates(dataPath='./', subpath='preprocessed/', duplicatePath='duplicates/'):
#   allSectors = [item+'/' for item in os.listdir(dataPath) if 'sector' in item]
#   allSectors = [sector for sector in allSectors if 'spoc' not in sector]
#   # Sort sectors in descending order
#   sector_nums = [int(sector.split('-')[1][:-1]) for sector in allSectors]
#   sorted_idx = np.argsort(sector_nums)
#   sorted_sectors = list(np.array(allSectors)[sorted_idx])[::-1]
#   foundTics    = []
#   foundSectors = []

#   fileCount = 0
#   duplicateCount = 0

#   for sector in tqdm(sorted_sectors):
#     # Path to lightcurves
#     lcPath = os.path.join(dataPath, sector, subpath)

#     # Path to duplicate Lightcurves
#     duplicatePath = os.path.join(lcPath, 'duplicates/')
#     if not os.path.exists(duplicatePath):
#       os.mkdir(duplicatePath)

#     for lcfile in tqdm(os.listdir(lcPath)):
#       # get TIC ID from lightcurve files
#       if '.h5' not in lcfile:
#         continue
#       tic = lcfile.split('.')[0]

#       if tic in foundTics:
#         # This TIC ID has been found in a later sector
#         # Move the lightcurve file to the duplicate folder
#         idx = np.where(np.array(foundTics) == tic)[0][0]
#         srcName = os.path.join(lcPath,lcfile)
#         newFileName = str(tic)+'_'+foundSectors[idx][:-1]+'.h5'
#         destName = os.path.join(duplicatePath, newFileName)
#         os.rename(srcName, destName)
#         duplicateCount+=1
#       else:
#         # New TIC ID, log that it was found
#         foundTics.append(tic)
#         foundSectors.append(sector)
#       fileCount+=1

#   print(str(fileCount) + ' files found, '+str(duplicateCount)+' duplicates, '+str(fileCount-duplicateCount)+'  originals')

def removeDuplicates(dataPath='./', subpath='preprocessed/', duplicatePath='duplicates/'):
  allSectors = [item+'/' for item in os.listdir(dataPath) if 'sector' in item]
  allSectors = [sector for sector in allSectors if 'spoc' not in sector]
  # Sort sectors in descending order
  sector_nums = [int(sector.split('-')[1][:-1]) for sector in allSectors]
  sorted_idx = np.argsort(sector_nums)
  sorted_sectors = list(np.array(allSectors)[sorted_idx])[::-1]

  higherSectors = []

  fileCount = 0
  duplicateCount = 0

  for sector in tqdm(sorted_sectors, desc='sector'):
    # Path to lightcurves
    lcPath = os.path.join(dataPath, sector, subpath)

    # Path to duplicate Lightcurves
    duplicatePath = os.path.join(lcPath, 'duplicates/')
    if not os.path.exists(duplicatePath):
      os.mkdir(duplicatePath)

    for lcfile in tqdm(os.listdir(lcPath)):
      # get TIC ID from lightcurve files
      if '.h5' not in lcfile:
        continue
      tic = lcfile.split('.')[0]

      # look through the above sectors for this tic ID
      for higherSector in higherSectors:
        higherLCPath = os.path.join(dataPath, higherSector, subpath)
        # if found, mark this one as duplicate
        if os.path.exists(os.path.join(higherLCPath, lcfile)):
          srcName = os.path.join(lcPath,lcfile)
          newFileName = str(tic)+'_'+higherSector[:-1]+'.h5'
          destName = os.path.join(duplicatePath, newFileName)
          os.rename(srcName, destName)
          duplicateCount+=1
          break
        else:
          continue
      fileCount+=1
    higherSectors.append(sector)

  print(str(fileCount) + ' files found, '+str(duplicateCount)+' duplicates, '+str(fileCount-duplicateCount)+'  originals')

def test():
  # baseDir = '/pdo/qlp-data/'
  outDir  = './'
  # sector = 'sector-15/'
  # writeANScoreFiles(baseDir, outDir, sector)
  # fatalParse = parseANO(baseDir, outDir, sector, threshold=.09)
  # for each in fatalParse:
  #   if '28230919' in each:
  #     print(each)
  # getStellarParamsSector(outDir, 'sector-14/')
  # getStellarParamsSector(outDir, 'sector-12/')
  # getStellarParamsSector(outDir, 'sector-11/')
  # getStellarParamsSector(outDir, 'sector-22/')
  # getStellarParamsSector(outDir, 'sector-21/')
  # getStellarParamsSector(outDir, 'sector-20/')
  # getStellarParamsSector(outDir, 'sector-19/')
  # getStellarParamsSector(outDir, 'sector-18/')
  # getStellarParamsSector(outDir, 'sector-17/')
  # getStellarParamsSector(outDir, 'sector-16/')
  # getStellarParamsSector(outDir, 'sector-15/')
  # getStellarParamsSector(outDir, 'sector-14/')
  # runPreprocess(baseDir, outDir, 'sector-1/',.09,verbose=True)
  rerunFatalSector(outDir, 'sector-9/',verbose=True)

def main():
  hasPreprocessed = False
  hasRerunFatal = False
  hasGotSP = False
  hasRemoveDup = False

  baseDir = '/pdo/qlp-data/'
  outDir  = './'
  ANthreshold = 0.09
  allSectors = [item+'/' for item in os.listdir(baseDir) if 'sector' in item]
  allSectors = [sector for sector in allSectors if 'spoc' not in sector]

  if len(sys.argv) == 1:
    print('Must specify functions from: preprocess, rerun, getStellarParams, removeDuplicates')
    return

  mode = sys.argv[1]

  sectors = sys.argv[2:]
  if sectors == []:
    sectors = allSectors
  for i, sector in enumerate(sectors):
    if sector[-1] != '/':
      sectors[i] = sector+'/' 

  if mode == 'preprocess':
    for sector in sectors:
      if os.path.isdir(os.path.join(outDir,sector)):
        print(sector, ' Exists. Skipping')
        continue
      writeANScoreFiles(baseDir, outDir, sector)
      runPreprocess(baseDir, outDir, sector, threshold=ANthreshold)
  elif mode == 'rerun':
    totalfixed = 0
    for sector in sectors:
      totalfixed += rerunFatalSector(outDir, sector)
    if totalfixed == 0:
      print('Rerun Fatal -- None Successful')
  elif mode == 'getStellarParams':
    for sector in sectors:
      getStellarParamsSector(outDir, sector)
  elif mode == 'removeDuplicates':
    removeDuplicates()
  elif mode == 'test':
    test()
  else:
    print('Arguments must specify functions from: preprocess, rerun, getStellarParams, removeDuplicates')

if __name__ == "__main__":
  main()