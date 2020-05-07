import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

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

def readPickle(filepath):
  assert os.path.exists(filepath)
  with open(filepath,'rb') as f:
    data = pickle.load(f)
  
  nfiles = len(data)
  
  #views
  localviews = np.zeros((nfiles,3,61))
  globalviews = np.zeros((nfiles,3,201))
  
  std_depths     = np.zeros((nfiles,2))
  true_depths    = np.zeros((nfiles,6))
  depth_errors   = np.zeros((nfiles,6))
  astronet_score = np.zeros((nfiles,1))
  stellar_radii  = np.zeros((nfiles,1))
  magnitudes     = np.zeros((nfiles, 1))

  # Labels
  labels = np.zeros((nfiles,))
  
  for i,file in enumerate(data):
    stdcut = (np.std(file['LocalDepths']) < 1) and (np.std(file['GlobalDepths']) < 1)
    non_zero_depths = ((np.sum(file['LocalDepths'] == 0) + np.sum(file['GlobalDepths'] == 0)) == 0)
    no_nans = np.sum(np.isnan(file['LocalView'])) + np.sum(np.isnan(file['GlobalView'])) == 0

    if no_nans and non_zero_depths and stdcut:

      localviews[i] = file['LocalView']
      globalviews[i] = file['GlobalView']
      labels[i] = file['label']

      astronet_score[i] = np.array([file['AstroNetScore']])
      std_depths[i]     = np.array([np.std(file['LocalDepths']),np.std(file['GlobalDepths'])])
      true_depths[i]    = np.hstack([file['LocalDepths'],file['GlobalDepths']])
      depth_errors[i]    = np.hstack([file['LocalDepthErrors'],file['GlobalDepthErrors']])
      stellar_radii[i]  = np.array([file['StellarParams']['rad']])
      magnitudes[i]     = np.array(file['StellarParams']['tmag'])

  return labels, localviews, globalviews, std_depths, true_depths, depth_errors, astronet_score, stellar_radii, magnitudes

def loadSortedData(pickleDir, sectors, verbose=True):
  std_depths = np.zeros((0,2))
  true_depths = np.zeros((0,6))
  depth_errors = np.zeros((0,6))
  astronets = np.zeros((0,1))
  stellar_radii = np.zeros((0,1))
  all_mag    = np.zeros((0,1))

  all_localviews = np.zeros((0,3,61))
  all_globalviews = np.zeros((0,3,201))
  all_labels = np.zeros((0,))

  # append with data from each sector
  for sector in tqdm(sectors):
    filepath = pickleDir + sector + '.pickle'
    labels, localviews, globalviews, std_depth, true_depth, depth_error, astronet_score, stellar_rad, magnitude = readPickle(filepath)
    
    std_depths    = np.append(std_depths, std_depth,axis=0)
    true_depths   = np.append(true_depths,true_depth,axis=0)
    depth_errors  = np.append(depth_errors, depth_error, axis=0)
    astronets     = np.append(astronets, astronet_score, axis = 0)
    stellar_radii = np.append(stellar_radii, stellar_rad , axis = 0)
    all_mag       = np.append(all_mag, magnitude, axis=0)
    
    all_localviews = np.append(all_localviews, localviews,axis=0)
    all_globalviews = np.append(all_globalviews, globalviews,axis=0)
    all_labels = np.append(all_labels,labels)

  if verbose:
    print("X_global.shape = ", all_globalviews.shape)
    print("X_local.shape  = ", all_localviews.shape)
    print("X_depths.shape = ", true_depths.shape)
    print("y.shape        = ", all_labels.shape)

  return all_labels, all_localviews, all_globalviews, true_depths, depth_errors, std_depths, astronets, stellar_radii, all_mag

def addToDataMatrix(X, addon):
  n = np.shape(X)[0]
  ncols = np.prod(np.shape(addon)[1:])
  X = np.append(X, addon.reshape(n,ncols), axis=1)
  return X

def genDataMatrix(y, *features):
  n = len(y)
  X = np.zeros((n,0))
  for feat in features:
    X = addToDataMatrix(X, feat)
  return X, y

##########################################

def getPCs(X,y):
  pcs = X[np.where(y==1)]
  npcs = X[np.where(y==0)]
  return pcs, npcs

def magnitudeCut(X, y, magnitudes, cut):
  pcs, npcs = getPCs(X,y)
  npcIndex = np.where(y==0)
  npcMags = magnitudes[npcIndex]

  indicies = np.where(npcMags < cut)[0]
  npcs = npcs[indicies]

  X = np.vstack((pcs,npcs))
  y = np.concatenate((np.ones(len(pcs)), np.zeros(len(npcs))))
  return X,y

def overSamplePos(x,y,random_state=420):
  data = np.hstack((x,y[:,np.newaxis]))

  pos = data[data[:,-1] == 1]
  neg = data[data[:,-1] == 0]
  
  pos_oversample = resample(pos, replace=True, n_samples=len(neg),
                            random_state=random_state)
  
  data = np.append(neg, pos_oversample, axis=0)
  x = data[:,:-1]
  y = data[:,-1]
  return x,y


# -- Models
def RFModel(x_train,y_train,x_test,y_test,
            depth, n_jobs, random_state=420):
  print(n_jobs)
  clf = RandomForestClassifier(n_estimators = 100,
                               max_depth=depth,
                               n_jobs=n_jobs,
                               random_state=random_state)
  clf.fit(x_train,y_train)
  
  predictions = clf.predict(x_test)
  p_planet = clf.predict_proba(x_test)[:,1]
  return p_planet, clf.feature_importances_

def LRModel(x_train,y_train, x_test,y_test,
            random_state=420):
  reg = linear_model.LinearRegression()
  reg.fit(x_train, y_train)
    
  score = np.dot(x_test, reg.coef_)
  return score, [reg.coef_]
###

#-- Model Running
def testModel(x,y, model, *params, k=5, overSample=True, random_state=420):
  kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
  results = []
  model_info = []
  for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f'{i+1} / {k}',end=' ')
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    if overSample:
      x_train, y_train = overSamplePos(x_train, y_train, random_state=random_state)

    scores, info = model(x_train,y_train,x_test,y_test,
                             *params, random_state=random_state)

    results.append(np.array([scores,y_test]))
    model_info.append(info)
  return results, np.array(model_info)

def aggregateScores(modelOutput):
  return np.concatenate(modelOutput,1)

def getConfusion(scores, labels, threshold):
  pos      = (scores > threshold)
  real     = (labels == 1)

  true_neg  = np.sum(np.logical_and(~pos, ~real))
  true_pos  = np.sum(np.logical_and( pos,  real))
  false_neg = np.sum(np.logical_and(~pos,  real))
  false_pos = np.sum(np.logical_and( pos, ~real))

  return np.array([true_neg, true_pos, false_neg, false_pos])

def calcPrecision(confusion):
  true_neg, true_pos, false_neg, false_pos = confusion
  return true_pos / (true_pos + false_pos)

def calcRecall(confusion):
  true_neg, true_pos, false_neg, false_pos = confusion
  return true_pos / (true_pos + false_neg)

def getPRCurve(scores, labels, thresholds):
  precision = []
  recall    = []
  for threshold in thresholds:
    confusion = getConfusion(scores, labels, threshold)
    precision.append(calcPrecision(confusion))
    recall.append(calcRecall(confusion))

  return np.array(recall), np.array(precision)

def interpretModel(y, *data):
  nPos = int(np.sum(y))
  nNeg = int(np.sum(1-y))
  print(f'{nPos} - True Positive, {nNeg} - True Negative')

  print('Feature Importances')
  for each in data:
    info = each[1]
    print(np.median(info,0))

  sns.set()
  plt.figure()
  for i,each in enumerate(data):
    stack = np.hstack(each[0])
    plt.plot(*getPRCurve(*stack,np.arange(0,1,.01)), label=i)
  
  guessLevel = nPos / (nNeg + nPos)
  plt.plot((0,1),(guessLevel, guessLevel))
  plt.legend()
###