#!/usr/bin/python3
# tain.py
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
from joblib import dump
from sklearn import preprocessing



initial_count=0  

def count_number(): 
    global initial_count
    dir = "./data"
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
           initial_count += 1
    return initial_count       

def train():
    
    # Load directory paths for persisting model
    
    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    
    #Container內的模型儲存路徑
    #MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)	    
    #MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    
    #將模型儲存在mapping的資料夾
    LOCAL_DIR = os.environ["LOCAL_PATH"]
    #SAVE_AT_LOCAL_LDA = os.path.join(LOCAL_DIR, MODEL_FILE_LDA)
    
      
    # Load, read and normalize training data
    training = "./data/train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    #clf_lda = LinearDiscriminantAnalysis()
    #clf_lda.fit(X_train, y_train)
    
    # Save model
    from joblib import dump
    #處存在Container內
    #dump(clf_lda, MODEL_PATH_LDA)
    #dump(clf_lda, SAVE_AT_LOCAL_LDA)
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)
    
       
    # Secord model
    from joblib import dump, load
    #dump(clf_NN, MODEL_PATH_NN)
    
    #得到回傳值(資料夾內檔案數量)
    count_of_file=count_number()
    NAME_with_count=str(count_of_file)+"_"+MODEL_FILE_NN
    SAVE_AT_LOCAL_NN =os.path.join(LOCAL_DIR,  NAME_with_count)
    dump(clf_NN, SAVE_AT_LOCAL_NN)
        
if __name__ == '__main__':
    train()
