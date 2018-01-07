#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 02:53:18 2018

@author: arvy
"""

import numpy as np
import pandas as pd

data=pd.read_csv('/home/arvy/Documents/ML/datasets/energydata_complete.csv')
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
train_test_split
data=load_boston()
X=data.data
y=data.target

y=y.reshape([506,1])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

def cal(x,w):
    z=np.dot(x,w.T)
    return z

def cost(z,y):
    j=np.sum(((y-z)**2))/1012;
    return j

def optimize(w,x,y,z):
    t=(z-y)
    w=w + 0.001*np.dot(t.T,x)
    return w

def pred(test,w):
    p=np.dot(test,w.T)
    print(p)
    
w=np.array([[100,100,100,100,100,100,100,100,100,100,100,100,100]])

for i in range(0,5):
    z=cal(X_train,w)
    
    j=cost(z,y_train)
    print(j)
    w=optimize(w,X_train,y_train,z)
    print(w)



"""x=np.array([[0,0,0],[1,1,1],[2,2,2]])
x=np.concatenate((np.ones([3,1]),x),axis=1)
print(x)
y=np.array([[0],[1],[2]])
w=np.array([[3,3,3,3]])
for i in range(0,20):
    z=cal(x,w)
    print(z)
    
    j=cost(z,y)
    print(j)
    w=optimize(w,x,y,z)
    print(w)


pred([[1,5,5,5]],w)"""
