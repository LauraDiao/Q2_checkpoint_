## Import Libraries
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import os
import sys
from fractions import Fraction

## Functions to create features and EDA
def max_bytes(x,y):
    maxbytes = pd.DataFrame([x,y]).T.groupby(0).sum().max().values[0]
    return maxbytes
## Function to return accuraucy based on 10% difference 
def accuracy(a,b):
    scale = a*.1
    if b <= a + scale and b >= a -scale:
        return 1
    else:
        return 0

def aggregation_by_second_plot(feature, method, data,loss,value1,value2):
    plt.rcParams["figure.figsize"] = (10,10)
    if loss == 'packet':
        plot1 = data.loc[data.loss_label==value1].reset_index().drop('index',axis=1)[:10].groupby('Time').agg(method)[feature].reset_index().drop('Time',axis=1)
        plot2 = data.loc[data.loss_label==value2].reset_index().drop('index',axis=1)[:10].groupby('Time').agg(method)[feature].reset_index().drop('Time',axis=1)
        plt.plot(plot1,label= str(Fraction(value1).limit_denominator()))
        plt.plot(plot2,label= str(Fraction(value2).limit_denominator()))
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel(' '.join(feature.split('_')))
        plt.title(' '.join(feature.split('_')).title() + ' per Second (Packet Loss)')
    elif loss == 'latency':
        plot1 = data.loc[data.latency_label==value1].reset_index().drop('index',axis=1)[:10].groupby('Time').agg(method)[feature].reset_index().drop('Time',axis=1)
        plot2 = data.loc[data.latency_label==value2].reset_index().drop('index',axis=1)[:10].groupby('Time').agg(method)[feature].reset_index().drop('Time',axis=1)
        plt.plot(plot1,label= str(Fraction(value1).limit_denominator()))
        plt.plot(plot2,label= str(Fraction(value2).limit_denominator()))
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel(' '.join(feature.split('_')))
        plt.title(' '.join(feature.split('_')).title() + ' per Second (Latency)')


