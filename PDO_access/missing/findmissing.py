from __future__ import print_function
import numpy as np
import os
import socket
from tqdm import tqdm

pdo = socket.gethostname()
print(pdo)

def find():
  baseDir = '/pdo/qlp-data/'
  sectors = os.listdir(baseDir)
  sectors = [sector for sector in sectors if 'sector' in sector]

  cams = ['cam1','cam2','cam3','cam4']
  ccds = ['ccd1', 'ccd2', 'ccd3', 'ccd4']

  ticlist = np.loadtxt('missing.txt',str)
  found = []
  foundCount=0
  notfound = []

  for tic in tqdm(ticlist):
    foundtic=False
    for sector in sectors[::-1]:
      foundsector=False
      for cam in cams:
        for ccd in ccds:
          testpath = os.path.join(baseDir,sector,'ffi/',cam,ccd,'LC/',tic)
          if os.path.exists(testpath):
            found.append([pdo, testpath, tic])
            foundsector=True
            foundtic=True
            foundCount+=1
      if foundsector:
        break
    print(foundCount)
    if not foundtic:
      notfound.append(tic)


  with open('found'+pdo.split('.')[0]+'.txt','w') as f:
    for line in found:
      print(','.join(line),file=f)

  with open('notfound'+pdo.split('.')[0]+'.txt','w') as f:
    for line in notfound:
      print(line,file=f)

def why():
  sectordir='../'
  master = []
  for i in range(1,6):
    foundfile = 'foundpdo'+str(i)+'.txt'
    with open(foundfile,'r') as f:
      lines = f.readlines()
      for line in lines:
        master.append(line)
  anfailcount = 0
  othercount=0
  spoccount=0
  otherfails = []
  tics = []
  for line in master:
    _, p, tic = line.strip().split(',')
    chunks = p.split('/')
    sector = chunks[3]
    cam = chunks[5]
    ccd = chunks[6]
    
    if tic in tics:
      continue
    else:
      tics.append(tic)
    
    if 'spoc' in sector:
      spoccount+=1
      continue
    
    anfile = os.path.join(sectordir,sector,'astroNetOutFiles.txt')
    
    with open(anfile,'r') as anf:
      anlines = anf.readlines()

    anfail=True
    for each in anlines:
      if cam in each and ccd in each:
        othercount+=1
        otherfails.append([tic.strip(),each.strip()])
        anfail=False
        break
    if anfail:
      anfailcount+=1

  print('looking at '+str(len(tics))+', '+str(anfailcount)+' failed for astronet reasons, '+str(othercount)+' other')

  with open('failed.txt','w') as f:
    for each in otherfails:
      print(','.join(each),file=f)

def getotherANscores():
  with open('failed.txt','r') as f:
    lines = f.readlines()

  for line in lines:
    tic, anf=line.strip().split(',')
    with open(anf,'r') as f2:
      an_lines = f2.readlines()
    for an_line in an_lines:
      t,s = an_line.split(' ')
      if t == tic.split('.')[0]:
        print(t, s, anf)

getotherANscores()