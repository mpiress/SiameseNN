import csv
from utils import read_file
from collections import Counter
import pandas as pd

#base genérica
from keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
        

    
def _get_real_train_and_test(self, path):
    le = preprocessing.LabelEncoder()

    dataset = read_file(path, has_header=False, norm=False)

    df = dataset.sample(frac=1)
    for _ in range(self.__k):
        df = pd.concat([df, dataset.sample(frac=1)])
        
    X  = []
    Y0 = []
    for t1, t2 in zip(df.values[0::2], df.values[1::2]):
        y  = cache_fx(t1, t2, self.__cache)
        Y0 += [y]*2
        X.append(t1)
        X.append(t2)

    le.fit(Y0)
    Y0 = le.transform(Y0)
    ros = RandomOverSampler(random_state=42)
    X, Y0 = ros.fit_resample(X, Y0)
    
    data = {}
    for t,k in zip(X,Y0):
        if k not in data:
            data[k] = []
        data[k].append(t)
    
    trainX = []
    trainY = []
    testX  = []
    testY  = []
    for k in data:
        tsize  = int(len(data[k]) * 0.05)
        tsize  = tsize if tsize % 2 == 0 else tsize + 1
        trainX += data[k][tsize:]
        trainY += [k]*len(data[k][tsize:])
        testX  += data[k][0:tsize]
        testY  += [k]*len(data[k][0:tsize])
        
    return trainX, trainY, testX, testY


