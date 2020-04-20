import os
import numpy as np
import itertools
import math as m
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import (Linear, ReLU, Conv1d, Flatten, Conv2d, Sequential,MaxPool1d, MaxPool2d, Dropout, CrossEntropyLoss)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision.datasets import MNIST

from skimage.io import imread, imshow
from skimage.transform import resize

from utils_hw8 import make_iter
from utils_hw8 import model_fit,call_model


#################################################
# Grab Data
#################################################
def shifted(X, shift):
    n = X.shape[0]
    m = X.shape[1]
    size = m + shift
    X_sh = np.zeros((n, size, size))
    plt.ion()
    for i in range(n):
        sh1 = np.random.randint(shift)
        sh2 = np.random.randint(shift)
        X_sh[i, sh1:sh1+m, sh2:sh2+m] = X[i, :, :]
        # If you want to see the shifts, uncomment
        #plt.figure(1); plt.imshow(X[i])
        #plt.figure(2); plt.imshow(X_sh[i])
        #plt.show()
        #input('Go?')
    return X_sh
  
def get_MNIST_data(shift=0):
    train = MNIST(root='./mnist_data', train=True, download=True, transform=None)
    val = MNIST(root='./mnist_data', train=False, download=True, transform=None)
    (X_train, y1), (X_val, y2) = (train.data.numpy(), train.targets.numpy()), \
                                  (val.data.numpy(), val.targets.numpy())
    if shift:
        X_train = shifted(X_train, shift)
        X_val = shifted(X_val, shift)
    return (X_train, y1), (X_val, y2)

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

def run_pytorch(train_iter, val_iter, test_iter, layers, epochs,
                verbose=True, history=None):

    # Model specification
    model = Sequential(*layers)

    # Define the optimization
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()
    
    # Fit the model
    train_m, vali_m = model_fit(model, train_iter, epochs=epochs, 
                                optimizer=optimizer, criterion=criterion,
                                validation_iter=val_iter,
                                history=history, verbose=verbose)
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


def run_pytorch_fc_mnist(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    '''
    train, test = input data
    layers = list of PyTorch layers, e.g. [Linear(in_features=784, out_features=10)]
    epochs = number of epochs to run the model for each training trial
    trials = number of evaluation trials, resetting weights before each trial
    '''
    if deterministic:
        make_deterministic()
    (X_train, y1), (X_val, y2) = train, test
    # Flatten the images
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m * m))
    X_val = X_val.reshape((X_val.shape[0], m * m))

    val_acc, test_acc = 0, 0
    for trial in range(trials):
        # Reset the weights
        for l in layers:
            weight_reset(l)
        # Make Dataset Iterables
        train_iter, val_iter = make_iter(X_train, y1, batch_size=32), make_iter(X_val, y2, batch_size=32)
        # Run the model
        model, vacc, tacc = \
            run_pytorch(train_iter, val_iter, None, layers, epochs, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))


def run_pytorch_cnn_mnist(train, test, layers, epochs, verbose=True, trials=1, deterministic=True):
    if deterministic:
        make_deterministic()
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
            run_pytorch(train_iter, val_iter, None, layers, epochs, verbose=verbose)
        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))

if __name__ == "__main__":
    train, validation = get_MNIST_data()
    layers = [ \
        nn.Linear(in_features=28*28, out_features=512), \
        nn.ReLU(), \
        nn.Linear(in_features=512,out_features=256), \
        nn.ReLU(), \
        nn.Linear(in_features=256,out_features=10)]

    run_pytorch_fc_mnist(train, validation, layers, 1, verbose=True)
