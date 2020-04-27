import os
import numpy as np
import h5py
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import (Linear, ReLU, Conv1d, Flatten, Conv2d, Sequential,MaxPool1d, MaxPool2d, Dropout, CrossEntropyLoss)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from skimage.io import imread, imshow
from skimage.transform import resize

import sklearn
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

from utils import make_iter,model_fit,call_model

#################################################
# Grab Data
#################################################
def get_LC(filepath):

    fin = h5py.File(filepath,'r')
    localview = fin['LocalView'].get("Aperture_%.3d" % fin['bestap'][0]).value
    fin.close()
    return localview

def get_data(TICS_folderpath,label_filepath,verbose=1):

    X = np.zeros((0,61),dtype='float')
    y = np.zeros((0,),dtype='int')
    ids = np.empty((0,),dtype='str')

    known_planets = np.genfromtxt(label_filepath,dtype='str')

    sectors = ["sector-{}".format(i) for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21]]

    for sector in sectors:
        # get the ids
        sectorids = os.listdir(os.path.join(TICS_folderpath,sector,"preprocessed"))
        if verbose: print("\n%s" % sector)

        for i,file in enumerate(sectorids):

            if verbose: print("--{:5d} / {:5d}\r".format(i,len(sectorids)),end='')
            filepath = os.path.join(TICS_drive,sector,"preprocessed",file)

            # get the lightcurve
            try:
                nX = get_LC(filepath)
            except:
                if verbose: print("bad file: ", filepath)
            # print(nX.shape, X.shape)
            X = np.vstack([X,nX])

            # get the label
            if file in known_planets:
                y = np.append(y,[1],axis=0)
            else:
                y = np.append(y,[0],axis=0)


        # get the id
        ids = np.append(ids,sectorids,axis=0)

    return X,y,ids

def generate_data(TICS_drive,generate_input,overwrite,verbose=0):

    if generate_input:
        
        # File containing names of known planets
        planets_file = os.path.join(TICS_drive,"planets.txt")

        # Generate input data
        Xfile = os.path.join(TICS_drive,"Xdata_cnn.npy")
        yfile = os.path.join(TICS_drive,"ydata_cnn.npy")
        idfile = os.path.join(TICS_drive,"ids_cnn.npy")

        if (os.path.exists(Xfile)==False) or \
            (os.path.exists(yfile)==False) or \
            (os.path.exists(idfile)==False) or overwrite:
            if verbose: print("generating input data")
            X,y,ids = get_data(TICS_drive, planets_file, verbose=verbose)
            np.save(Xfile,X)
            np.save(yfile,y)
            np.save(idfile,ids) 
        else:
            if verbose: print("not generating new data: overwrite off...")

    if verbose: print("loading existing input data...")
    X = np.load(os.path.join(TICS_drive,"Xdata_cnn.npy"))
    y = np.load(os.path.join(TICS_drive,"ydata_cnn.npy"))
    ids = np.load(os.path.join(TICS_drive,"ids_cnn.npy"))
    if verbose: print("X.shape, y.shape, ids.shape = ", X.shape, y.shape, ids.shape)

    return X,y,ids

def split_data(X,y,ids,n_fold):

    skf = StratifiedKFold(n_splits=5,shuffle=True)

    # only use first fold for now
    train_i, test_i = next(skf.split(X, y))

    return (X[train_i,:],y[train_i],ids[train_i]), (X[test_i,:],y[test_i],ids[test_i])

#################################################
# Run neural nets
#################################################
def make_deterministic():
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

def weight_reset(l):
    if isinstance(l, Conv2d) or isinstance(l, Linear):
        l.reset_parameters()

def run_pytorch(train_iter, val_iter, test_iter, layers, epochs,weights,verbose=True):

    # Model specification
    model = Sequential(*layers)

    # Define the optimization
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(weight=torch.tensor([50,.5]))

    
    # Fit the model
    train_m, vali_m = model_fit(model, train_iter, epochs=epochs, 
                                optimizer=optimizer, criterion=criterion,
                                validation_iter=val_iter,
                                verbose=verbose,history=None)
    if verbose: print()
    
    (train_loss, train_acc) = train_m
    (vali_loss, val_acc) = vali_m
    
    # Evaluate the model on test data, if any
    if test_iter is not None:
        test_loss, test_acc = model_evaluate(model, test_iter, criterion)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, val_acc, test_acc

def run_pytorch_fc(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    '''
    train, test = input data
    layers = list of PyTorch layers, e.g. [Linear(in_features=784, out_features=10)]
    epochs = number of epochs to run the model for each training trial
    trials = number of evaluation trials, resetting weights before each trial
    '''
    if deterministic:
        make_deterministic()
    (X_train, y1), (X_val, y2) = train, test

    # weights
    weights = torch.tensor(sklearn.utils.class_weight.compute_class_weight('balanced',[0,1],y1),dtype=torch.float64)

    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        for l in layers:
            weight_reset(l)
        # Make Dataset Iterables
        train_iter, val_iter = make_iter(X_train, y1, batch_size=32), make_iter(X_val, y2, batch_size=32)
        # Run the model
        model, vacc, tacc = \
            run_pytorch(train_iter, val_iter, None, layers, epochs, torch.tensor([50,.5]), verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))

    # WARNING: must cast input to FloatTensor not Double Tensor
    predict = model(torch.FloatTensor(X_val)).data.numpy()

    return sklearn.metrics.precision_recall_curve(y2, np.amax(predict,axis=1) , pos_label=1)

def run_pytorch_cnn(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    
    if deterministic: make_deterministic()
    
    # Load the dataset
    (X_train, y1), (X_val, y2) = train, test
    # Add a final dimension indicating the number of channels (only 1 here)
    
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], 1, m, m))
    X_val = X_val.reshape((X_val.shape[0], 1, m, m))

    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        for l in layers:
            weight_reset(l)
        # Make Dataset Iterables
        train_iter, val_iter = make_iter(X_train, y1, batch_size=32), make_iter(X_val, y2, batch_size=32)
        # Run the model
        model, vacc, tacc = \
            run_pytorch(train_iter, val_iter, None, layers, epochs, torch.tensor([1,1]), verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))

    predict = model(torch.FloatTensor(X_val)).data.numpy()

    return sklearn.metrics.precision_recall_curve(y2, np.amax(predict,axis=1) , pos_label=1)

if __name__ == "__main__":

    # runtime options
    verbose = 1
    generate_input = True
    overwrite = False  

    # TICS folder which has binned lightcurves
    TICS_drive   = "/Volumes/halston_lim/School_Documents/SP_2019-2020/6.862/TICS/"

    # generate data and split
    X,y,ids = generate_data(TICS_drive,generate_input,overwrite,verbose=verbose)
    (X_train,y_train, ids_train), (X_valid, y_valid, ids_valid) = split_data(X,y,ids,5)
    train = (X_train,y_train)
    test  = (X_valid, y_valid)

    # layers = [ \
    #     # d = 61
    #     nn.Conv1d(in_channels=1, out_channels=32,kernel_size=3), \
    #     # d = 59
    #     nn.ReLU(), \
    #     # d = 59
    #     nn.MaxPool1d(kernel_size=2), \
    #     # d = 59
    #     nn.Conv1d(in_channels=32, out_channels=64,kernel_size=3), \
    #     # d = 61
    #     nn.ReLU(), \
    #     nn.MaxPool1d(kernel_size=2), \
    #     nn.Flatten(), \
    #     nn.Linear(in_features=3648, out_features=128), \
    #     nn.Dropout(p=0.5), \
    #     nn.Linear(in_features=128,out_features=2)]

    # test = run_pytorch_cnn(train, test , layers, 1, verbose=verbose,trials=1)

    layers = [ \
        nn.Linear(in_features=61, out_features=512), \
        nn.ReLU(), \
        nn.Linear(in_features=512,out_features=256), \
        nn.ReLU(), \
        nn.Linear(in_features=256,out_features=2)]
    test = run_pytorch_fc(train, test , layers, 1, verbose=verbose,trials=1)



    plt.plot(test[0],test[1])
    plt.savefig('test.png',bbox_inches='tight')
