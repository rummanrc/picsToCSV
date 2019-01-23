import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
#from sklearn import datasets, svm, metrics
#fetch original mnist dataset
#from sklearn.datasets import fetch_mldata

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
'''

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (28, 28))
    return resized


def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(1,51):
        print('Load folder 1{:02d}'.format(j))
        #path = os.path.join('dataset', 'training_set', '1{:02d}'.format(j), '*.png')
        path = os.path.join('iso240', 'training_set', '1{:02d}'.format(j),'*')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)

    return X_train, y_train
X,y = load_train()
X = np.array(X)
dataset_size = len(X)
X = X.reshape(dataset_size,-1)
df=pd.DataFrame(X)
#df.to_csv("test01.csv",index=False,header=False)
ye=np.array(y)
newdf=df
newdf.loc[:,784]=pd.Series(ye,index=newdf.index)
newdf.to_csv("75kWithClass.csv",index=False,header=False)
