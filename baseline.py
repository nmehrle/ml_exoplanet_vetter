import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn import linear_model

datapath = './'
sectors = [f'sector-{i}' for i in np.arange(1,23)]


def loadData(sectors, datapath, lcFeatureFile='lcFeatures.npy', returnSectors=False):
  failedSectors = []
  allData = []
  for sector in sectors:
    try:
      lcfeatures = np.load(os.path.join(datapath, sector, lcFeatureFile))
      # EDITED
      b = lcfeatures[:,:19]
      a = lcfeatures[:,-1]
      a[a==0] = -1
      lcfeatures = np.concatenate((b,a[:,np.newaxis]),-1)
      ######
      allData.append(lcfeatures)
    except FileNotFoundError as e:
      failedSectors.append(sector)

  if len(failedSectors) > 0:
    temp = []
    print('Failed on --')
    for sector in sectors:
      if sector not in failedSectors:
        temp.append(sector)
      else:
        print(sector)
    sectors = temp

  allData = np.vstack(allData)
  x = allData[:,:-1]
  y = allData[:,-1]

  if returnSectors:
    return x, y, sectors

  return x, y

def overSamplePos(x,y,random_state=420):
  data = np.hstack((x,y[:,np.newaxis]))

  pos = data[data[:,-1] == 1]
  neg = data[data[:,-1] == -1]
  
  pos_oversample = resample(pos, replace=True, n_samples=len(neg),
                            random_state=random_state)
  
  data = np.append(neg, pos_oversample, axis=0)
  x = data[:,:-1]
  y = data[:,-1]
  return x,y


# -- Models
def RFModel(x_train,y_train,x_test,y_test,
            depth,random_state=420):
  clf = RandomForestClassifier(n_estimators = 100,
                               max_depth=depth,
                               random_state=random_state)
  clf.fit(x_train,y_train)
  
  predictions = clf.predict(x_test)
  p_planet = clf.predict_proba(x_test)[:,1]
  return p_planet, [clf.feature_importances_]

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

###