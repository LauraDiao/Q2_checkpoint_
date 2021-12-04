import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Data cleaning to turn columns into list of integers
def return_int(x):
    return [int(i) for i in x.split(';')[:-1]]

def longest_seq(aList):
    "find longest sequence in packet dirs "
    maxCount = 1
    actualCount = 1
    for i in range(len(aList)-1):
        if aList[i] == aList[i+1]:
            actualCount += 1
        else:
            actualCount = 1
        if actualCount > maxCount:
            maxCount = actualCount
    return maxCount


def agg10(t_df):
    #print(t_df.columns)
    indexcol = ['total_bytes','max_bytes','proto', "1->2Bytes",'2->1Bytes','1->2Pkts','2->1Pkts',
                'total_pkts', 'loss', 'latency']
    df = pd.DataFrame([t_df[:10]['total_bytes'].mean(),
                       t_df[:10]['max_bytes'].std(), 
                       t_df[:10]['Proto'].value_counts().idxmax(), # most frequent protocol
                       t_df[:10]['1->2Bytes'].mean(),
                       t_df[:10]['2->1Bytes'].mean(),
                       t_df[:10]['1->2Pkts'].mean(),
                       t_df[:10]['2->1Pkts'].mean(),
                       t_df[:10]['total_pkts'].mean(),
                       t_df.loss.unique()[0],
                       t_df.latency.unique()[0]],
                      index = indexcol).T
    
    for i in range(10, t_df.shape[0],10):
        df = pd.concat([df, pd.DataFrame([t_df[:10]['total_bytes'].mean(),
                                           t_df[:10]['max_bytes'].std(), 
                                           t_df[:10]['Proto'].value_counts().idxmax(), # most frequent protocol
                                           t_df[:10]['1->2Bytes'].mean(),
                                           t_df[:10]['2->1Bytes'].mean(),
                                           t_df[:10]['1->2Pkts'].mean(),
                                           t_df[:10]['2->1Pkts'].mean(),
                                           t_df[:10]['total_pkts'].mean(),
                                           t_df.loss.unique()[0],
                                           t_df.latency.unique()[0]],
                                        index = indexcol).T]
                                        ,ignore_index=True)
    return df

def onehot_(df): 
    d_proto = pd.get_dummies(df['Proto']).rename(columns={6: "TCP", 58: "IPv6"})
    d_p1 = pd.get_dummies(df['Port1'])
    d_p2 = pd.get_dummies(df['Port2'])
    temp_df = pd.concat([df, d_proto, d_p1, d_p2], axis=1)
    return temp_df


def time(dataframe): 
    # scales seconds 
    mini = dataframe['Time'].min()
    temp1 = dataframe.assign(Second = lambda x: x['Time'] - mini)

    return temp1


def main2(temp_df):
    transformed = temp_df #time(temp_df)
    label = 'latency'

    # print(transformed.columns)
    s =[ 'Second', label]
    
    p_sum_agg = transformed.groupby(s)['total_pkts'].agg(['count', 'sum']).reset_index()
    # print(p_sum_agg.columns)
    
    b_agg = transformed.groupby(s)['total_bytes'].agg(['count', 'sum']).reset_index()
    
    p_agg =  transformed.groupby(s)['total_pkts'].agg(['count', 'sum']).reset_index()
    # print("all aggregations made")
    return [p_sum_agg, b_agg, p_agg]

def genfeat(df): 
    tdf = df
    tdf['total_bytes'] = tdf['1->2Bytes'] + tdf['2->1Bytes'] # combining bytes
    tdf['total_pkts'] = tdf['1->2Pkts'] + tdf['2->1Pkts'] # combining packets
    tdf['packet_sizes'] = tdf['packet_sizes'].astype('str').apply(return_int) # converting list of packet sizes to type int
    tdf['pkt sum'] = tdf['packet_sizes'].apply(lambda x: sum(x)) # summing packets
    tdf['packet_dirs'] = tdf['packet_dirs'].astype('str').apply(return_int) # converting to type int
    tdf['longest_seq'] = tdf['packet_dirs'].apply(longest_seq) # finding longest sequence
    tdf['packet_times'] = tdf['packet_times'].apply(return_int) # converting to int
    #tdf = onehot_(tdf)
    def maxbyte(x):
        x = pd.DataFrame([x[0],x[1]]).T.groupby(0).sum().max().values[0]
        return x
    mx_byte = tdf[['packet_times', 'packet_sizes']].apply(maxbyte, axis =1) 
    tdf['max_bytes'] = mx_byte
    # print("max bytes generated")
    return tdf        


