from config import config_lac as config
from siamese import SIAMESERN
from utils import read_file
from cache_map import cache_fx
from sklearn import preprocessing

import seaborn as sn
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


APP = 'lac'

if __name__ == '__main__':

    
    #the dataset is composed of multidimensional data, each data (i.e., parameter) is used to report 
    #an attribute in data space evaluated. Firstly, normalize such input data to avoid overfitting 
    #during the training.
    dataset = pd.read_csv(config.TRAIN, header=None)
    
    
    #the siamese neural network structure is dynamically configured. The nhidden is a parameter that 
    #indicates the number of neurons on each hidden layer, epochs indicate the number of interactions 
    #during the training, the lr reports the progression factor applied, the ext and app are internal 
    #parameters that compose outputs files, the dropout is used to avoid overfitting, and, finally, 
    #the batch_size is used to configure the size batch used during accuracy evaluate.
    model = SIAMESERN(nhidden=[1024]*2, epochs=200, lr=1e-4, ext=APP, app=APP, dropout=0.2, batch_size=32)
    model.train(dataset) 