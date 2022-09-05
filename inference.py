#!/usr/bin/python3
# inference.py
# Xavier Vasques 13/04/2021


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing



def inference():

    MODEL_DIR = os.environ["MY_MODEL"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    print(MODEL_PATH_NN)
    clf_lda = load(MODEL_PATH_LDA)
    print("LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    print(MODEL_PATH_NN)
    clf_nn = load(MODEL_PATH_NN)
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))


if __name__ == '__main__':
    inference()
