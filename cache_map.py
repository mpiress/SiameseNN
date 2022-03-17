'''
Created on 25 de set de 2019

@author: michel
'''
import time
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances

def cache_fx_nscale (t1, t2):

    t1 = set([(t1[0], t1[1], t1[2], t1[3], t1[4],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13], t1[12], t1[5],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13], t1[12], t1[5], t1[7], t1[8],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13], t1[12], t1[5], t1[7], t1[8], t1[6],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13], t1[12], t1[5], t1[7], t1[8], t1[6], t1[9], t1[14],),\
        (t1[0], t1[1], t1[2], t1[3], t1[4], t1[13], t1[12], t1[5], t1[7], t1[8], t1[6], t1[9], t1[14], t1[10], t1[11],)])

    t2 = set([(t2[0], t2[1], t2[2], t2[3], t2[4],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13], t2[12], t2[5],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13], t2[12], t2[5], t2[7], t2[8],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13], t2[12], t2[5], t2[7], t2[8], t2[6],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13], t2[12], t2[5], t2[7], t2[8], t2[6], t2[9], t2[14],),\
        (t2[0], t2[1], t2[2], t2[3], t2[4], t2[13], t2[12], t2[5], t2[7], t2[8], t2[6], t2[9], t2[14], t2[10], t2[11],)])


    return round(len(t1 & t2)/len(t1),1)
    

def cache_fx_lac (t1, t2):
    
    t1 = list(enumerate(t1[0:-1]))
    t2 = list(enumerate(t2[0:-1]))
    
    t1 = set(t1)
    t2 = set(t2)
    sizeof = len(t1)

    return round(len(t1 & t2)/sizeof,1)

       
def cache_fx(t1, t2, app):
    if app == 'lac':
        return cache_fx_lac(t1, t2)
    elif app == 'nscale':
        return cache_fx_nscale(t1, t2)
    
        

