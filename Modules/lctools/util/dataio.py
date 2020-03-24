#!/usr/bin/env python
# openfile.py this file is used for data io control
from types import *
import math
import os
import string
import numpy as np


def readcolumn(var,col,infile,datformat='float',div=None,fill=False):
    with open(infile,mode='r') as fin:
        data=fin.readline().split(div)
        if (datformat=='float'):
    	    while((not len(data)==0) and (not data == [''])):			
    		    if(not (data[0].startswith('#'))):
    			    try:
    				    var.append(float(data[col-1]))
    			    except ValueError:
    				    if(fill):
    					    var.append(np.nan)
    				    else:
    					    print data 
    					    raise 
    			    except IndexError:
    				    if(fill):
    					    var.append(np.nan)
    				    else:
    					    raise 
    		    data=fin.readline().split(div)
    
        if (datformat=='str'):
    	    while((not len(data)==0) and (not data == [''])):			
    		    if(not (data[0].startswith('#'))):
    			    try:
    				    var.append(data[col-1])
    			    except IndexError:
    				    if(fill):
    					    var.append(None)
    				    else:
    					    raise IndexError
    
    		    data=fin.readline().split(div)


def readtableline(infile, nline=1):
    # read the first nth line of a table with header
    candparam = {}
    i =1
    with open(infile, mode='r') as fin:
        headerline = fin.readline()
        content = fin.readline()  
        while not content=='' and i<nline:
            content = fin.readline()

        if headerline.startswith('#'):
            for i in xrange(len(headerline.split())-1):
                key = headerline.split()[i+1].split('_')[1]
                val = content.split()[i]
                candparam[key]=eval(val)
    return candparam

