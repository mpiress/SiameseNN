"""!

@PyMid is devoted to manage multitask algorithms and methods.

@authors Michel Pires da Silva (michelpires@dcc.ufmg.br)

@date 2014-2018

@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyMid is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    PyMid is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import numpy as np
import pandas as pd
import os, time, json, math, csv

from matplotlib import pyplot
from collections import Counter

from cache_map import cache_fx

#base genérica
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn import preprocessing


class SIAMESERN():


    def __init__(self, nhidden=[], epochs=200, opt=None, lr=1e-4, ext=None, app=None, dropout=0.2, batch_size=32):
        
        self.__epochs       = epochs
        self.__nhidden      = nhidden
        
        self.__input        = 0
        self.__out          = 0
        self.__classes      = []
        self.__opt          = opt
        self.__lr           = lr
        self.__ext          = ext
        self.__app          = app
        self.__dropout      = dropout
        self.__batch_size   = batch_size
        
        self.__merge        = 0
        

        np.random.seed(42)
        
        

    def __eval_abs(self, t1, t2):
        return tf.keras.backend.abs(t1 - t2)

    
    def __nnmodel(self):

        T1  = tf.keras.layers.Input((self.__input,), name='t1')
        T2  = tf.keras.layers.Input((self.__input,), name='t2')
        
        initializer = tf.keras.initializers.he_uniform(seed=42)
        bias_init   = 'zeros'
        
        self.__model = tf.keras.Sequential()
        
        
        for h in self.__nhidden[0:-1]:
            self.__model.add(tf.keras.layers.Dense(h, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init))
            self.__model.add(tf.keras.layers.Dropout(rate=self.__dropout, seed=42))
            
        self.__model.add(tf.keras.layers.Dense(self.__nhidden[-1], activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init))
        
        #encoding inputs based on T1 and T2 
        ET1 = self.__model(T1)
        ET2 = self.__model(T2)

        # Add a customized layer to compute the absolute difference between the encodings
        L1 = tf.keras.layers.Lambda(lambda tensors:self.__eval_abs(tensors[0], tensors[1]))
        L1F = L1([ET1, ET2])
        
        # Add a dense layer with a softmax unit to generate the similarity score
        pred = tf.keras.layers.Dense(self.__out, activation='softmax', use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init)(L1F)
        
        self.__model = tf.keras.Model(inputs=[T1, T2], outputs=pred)
        
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.__lr, amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['acc'])

    
    def erase_model(self):
        del(self.__model)

    
    def __prepare_input(self, dataset):
        
        X     = []
        y     = []
        train = []
        for i1, i2 in zip(dataset[0::2].index, dataset[1::2].index):
            t1 = dataset[dataset.index == i1].values[0]
            t2 = dataset[dataset.index == i2].values[0]
            value = cache_fx(t1, t2, self.__app)
            X.append((i1, i2))
            y.append(value)
            
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        for x,y in zip(X, y):
            train.append({'t1':x[0], 't2':x[1], 'similarity':y})

        output = 'buffer/'+self.__app+'_train.csv'
        df = pd.DataFrame(train)
        df.to_csv(output, index=None)

        return df


    # load the dataset
    def __load_dataset(self, dataset):
        X = []
        y = []
        
        filename = 'buffer/'+self.__app+'_train.csv'
        # load the dataset as a pandas DataFrame        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = self.__prepare_input(dataset)

        for line in df.itertuples():
            X.append((line[1], line[2]))
            y.append(int(line[3]))
            
        
        return X, y

    def __preprocessing(self, X_train, X_test):
        
        train = []
        df = pd.DataFrame(X_train)
        for line in df.itertuples():
            train.append(line[1])
            train.append(line[2])
        
        test = []
        df = pd.DataFrame(X_test)
        for line in df.itertuples():
            test.append(line[1])
            test.append(line[2])

        return train, test
        

    def __prepare_dataset(self, dataset):

        output = 'buffer/'+self.__app+'_train.csv'

        X, y = self.__load_dataset(dataset)
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, y_train = ros.fit_resample(X_train, y_train)
        print('Classes:', Counter(y_train))

        
        X = []
        y = []
        for a, b in zip(X_train, y_train):
            t1 = dataset[dataset.index == a[0]].values[0]
            t2 = dataset[dataset.index == a[1]].values[0]
            X.append(t1)
            X.append(t2)
            y.append(b)

        X_train = np.array(X)
        y_train = np.array(y)

        X = []
        y = []
        for a, b in zip(X_test, y_test):
            t1 = dataset[dataset.index == a[0]].values[0]
            t2 = dataset[dataset.index == a[1]].values[0]
            X.append(t1)
            X.append(t2)
            y.append(b)

        X_test = np.array(X)
        y_test = np.array(y)        

        return X_train, y_train, X_test, y_test
        

    def train(self, dataset):
        checkpoint = os.path.dirname(os.path.abspath(__file__)) + os.sep + "buffer/model_"+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+".h5"
        history    = None

        if not os.path.exists(checkpoint):
            trainX, trainY, testX, testY = self.__prepare_dataset(dataset)
            
            self.__classes = list(sorted(Counter(trainY).keys()))
            self.__out     = len(self.__classes)
            self.__input   = len(trainX[0])
            
            
            self.__nnmodel()
                
            ts = time.time()
            #rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', min_delta=0.001, cooldown=0, min_lr=0)
            #sea = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
            cp = tf.keras.callbacks.ModelCheckpoint(checkpoint, verbose=0, save_weights_only=True, period=self.__epochs)
            history = self.__model.fit({'t1':np.array(trainX[0::2]), 't2':np.array(trainX[1::2])}, trainY, epochs=self.__epochs, callbacks=[cp],\
                                       shuffle=True, validation_data=({'t1':np.array(testX[0::2]), 't2':np.array(testX[1::2])}, testY), use_multiprocessing=True, batch_size=self.__batch_size)
            te = time.time() 
            data = history.history
            data['Runtime'] = te - ts
            
            pyplot.xlabel('epochs ('+str(self.__ext)+')')
            pyplot.ylabel('accuracy')
            pyplot.plot(data['acc'], label='train')
            pyplot.plot(data['val_acc'], label='test')
            path = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'buffer/'
            pyplot.legend()
            pyplot.savefig(path + 'accuracy_'+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+'.png')
            pyplot.close()

            pyplot.xlabel('epochs ('+str(self.__ext)+')')
            pyplot.ylabel('loss')
            pyplot.plot(data['loss'], label='train')
            pyplot.plot(data['val_loss'], label='test')
            path = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'buffer/'
            pyplot.legend()
            pyplot.savefig(path + 'loss_'+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+'.png')
            pyplot.close()

            with open(path + 'nndata_'+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+'.pkl', 'w') as fp:
                json.dump(str(data), fp)
            
            with open(path + 'classes_'+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+'.pkl', 'w') as fp:
                writer = csv.writer(fp, delimiter=',')
                writer.writerow(self.__classes)

        else:
            path = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'buffer/'
            with open(path + 'classes_'+str(self.__nhidden[0])+'_'+str(self.__ext)+"_"+str(self.__batch_size)+'.pkl', 'r') as fp:
                reader = csv.reader(fp, delimiter=',')
                classes = next(reader)
            
            self.__classes = [int(c) for c in classes]
            self.__out     = len(self.__classes)
            self.__input   = len(dataset.values[0])
            self.__nnmodel()
            self.__model.load_weights(checkpoint)
            
                

    def predict(self, t1, t2): 
        pred = self.__model.predict({'t1':np.array(t1), 't2':np.array(t2)})
        resp = [self.__classes[np.argmax(p)] for p in pred]
        return resp

