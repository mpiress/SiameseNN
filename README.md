# SiameseNN

The siamese neural network (SNN) was firstly presented in 2004 to compare digital signatures and identify a similarity range between two signatures provided as input. In this implementation, we propose a generic architecture using such an approach to compare inputs extracted from multidimensional data sets. 

In the generic implementation, we adopt a structure composed of two hidden layers, each with 1024 neurons, a function of loss rate in 1-4e and dropout in 0.2. These parameters are modified according to demand, rewriting the main.py file. 

The output of the SNN proposed reports a similarity rate between two inputs. Such similarity is used to show the overlap degree among inputs provided. To train our approach, a set of multidimensional data input is introduced, which is reported into the main.py. For greater reliability and shorter training runtime, is very important to provide normalized data as input for such stage. Such normalization patterns should be used during the test stage too.
