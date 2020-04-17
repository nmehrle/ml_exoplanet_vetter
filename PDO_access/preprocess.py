from __future__ import print_function
from collectLCFiles import writeANScoreFiles, parseANO
from processLC import processLC
import os, sys
import numpy as np

def parseFileList(lines, sector, outputFolder, limitTo=None):
  ngood  = 0
  errors = []
  fatal  = []
  for i, line in enumerate(lines):
    lcfile, blsfile, score = line.strip().split(' ')
    if limitTo is not None:
      if lcfile not in limitTo:
        continue
        
    lcname = lcfile.split('/')[-1]
    outfile = os.path.join(outputFolder, lcname)

    print("{} -- {} / {}\r".format(sector, i,len(lines)-1),end="")

    val = processLC(lcfile, blsfile, outfile, score)
    if val == 1:
      ngood+=1
    elif val == 0:
      errors.append(lcfile)
    elif val == -1:
      fatal.append(lcfile)

  return ngood, errors, fatal

def runPreprocess(baseDir, outDir, sector, threshold=0.09):
  print(sector)
  print('--')
  fatalParse = parseANO(baseDir, outDir, sector, threshold=0.09)
  fileList = os.path.join(outDir, sector, 'filesToPreProc.txt')
  with open(fileList,'r') as f:
    lines = f.readlines()

  outputFolder = os.path.join(outDir, sector,'preprocessed/')
  try:
    os.mkdir(outputFolder)
  except OSError:
    pass
  
  ngood, errors, fatal = parseFileList(lines, sector, outputFolder)
  fatal = np.hstack((fatalParse,fatal))

  print('Processed {} lightcurves'.format(len(lines)))
  print('  --  {} Done successfully'.format(ngood))
  print('  --  {} Done partially'.format(len(errors)))
  print('  --  {} failed'.format(len(fatal)))

  with open(os.path.join(outDir, sector, 'preproc_error.txt'),'w') as f:
    for each in errors:
      print(each, file=f)

  with open(os.path.join(outDir, sector, 'preproc_fatal.txt'),'w') as f:
    for each in fatal:
      print(each, file=f)

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

  for i, fileName in enumerate(fileList):
    print("{} -- {} / {}\r".format(sector, i,len(fileList)-1),end="")

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

def rerunFatalSector(outDir, sector):
  print(sector)
  print('--')

  fullfileList = os.path.join(outDir, sector, 'filesToPreProc.txt')
  with open(fullfileList,'r') as f:
    lines = f.readlines()

  fatalFiles = os.path.join(outDir, sector, 'preproc_fatal.txt')
  with open(fatalFiles,'r') as f:
    rawFatalLines = f.readlines()

  fatalLines = []
  for fl in rawFatalLines:
    fatalLines.append(fl.strip())

  print('Rerunning {} failed lightcurves'.format(len(fatalLines)))

  outputFolder = os.path.join(outDir, sector,'preprocessed/')
  ngood, errors, fatal = parseFileList(lines, sector, outputFolder, fatalLines)
  print('Processed {} lightcurves'.format(len(lines)))
  print('  --  {} Done successfully'.format(ngood))
  print('  --  {} Done partially'.format(len(errors)))
  print('  --  {} failed'.format(len(fatal)))
  print('')
  print('')

def removeDuplicates():
  print('todo')

def main():
  hasPreprocessed = False
  hasRerunFatal = False
  hasGotSP = False
  hasRemoveDup = False

  baseDir = '/pdo/qlp-data/'
  outDir  = './'
  ANthreshold = 0.09
  sectors = [item+'/' for item in os.listdir(baseDir) if 'sector' in item]
  sectors = [sector for sector in sectors if 'spoc' not in sector]

  if len(sys.argv) == 1:
    print('Must specify functions from: preprocess, rerunFatal, getStellarParams, removeDuplicates')
    return

  for arg in sys.argv[1:]:
      if arg == 'preprocess':
        if hasPreprocessed:
          pass
        else:
          for sector in sectors:
            writeANScoreFiles(baseDir, outDir, sector)
            runPreprocess(baseDir, outDir, sector, threshold=ANthreshold)
            hasPreprocessed=True
      elif arg == 'rerunFatal':
        if hasRerunFatal:
          pass
        else:
          for sector in sectors:
            rerunFatalSector(outDir, sector)
            hasRerunFatal=True
      elif arg == 'getStellarParams':
        if hasGotSP:
          pass
        else:
          for sector in sectors:
            getStellarParamsSector(outDir, sector)
            hasGotSP=True
      elif arg == 'removeDuplicates':
        if hasRemoveDup:
          pass
        else:
          removeDuplicates()
          hasRemoveDup=True
      else:
        print('Arguments must specify functions from: preprocess, rerunFatal, getStellarParams, removeDuplicates')
        return

if __name__ == "__main__":
  main()