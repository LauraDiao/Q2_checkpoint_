from helper import *
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

def test_feat(df, cols, p): 
    # col is feauture comb
    # p is for loss or latency
    # 1: loss  # 2 : latency
    X = df[cols]
    
    if p == 1: 
        y = df.loss
    if p == 2: 
        y = df.latency
        
    # randomly split into train and test sets, test set is 80% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc1 = mean_squared_error(y_test, y_pred)
    
    # calculate loss
    # retreive metrics
    #print("Decision Tre Accuracy:", acc1, '\n')
    
    clf2 = RandomForestRegressor(n_estimators=10)
    clf2 = clf2.fit(X_train,y_train)
    y_pred2 = clf2.predict(X_test)
    acc2= mean_squared_error(y_test, y_pred2)
    #print("Random Forest Accuracy:", acc2, '\n')
    
    clf3 = ExtraTreesRegressor(n_estimators=10)
    clf3 = clf3.fit(X_train,y_train)
    y_pred3 = clf3.predict(X_test)
    acc3= mean_squared_error(y_test, y_pred3)
    #print("Extra Trees Accuracy:", acc3, '\n')
    
    pca = PCA() 
    X_transformed = pca.fit_transform(X_train) 
    cl = DecisionTreeRegressor() 
    cl.fit(X_transformed, y_train)
    newdata_transformed = pca.transform(X_test)
    y_pred4 = cl.predict(newdata_transformed)
    acc4 = mean_squared_error(y_test, y_pred4)
    #print("PCA Accuracy:", acc4, '\n')
    
    return [acc1, acc2, acc3, acc4 ]


def getAllCombinations(object_list):
    lst = ['total_bytes', 'max_bytes', 'proto', '1->2Bytes','2->1Bytes', '1->2Pkts', '2->1Pkts', 'total_pkts', 'loss']
    uniq_objs = set(lst)
    combinations = []
    for obj in uniq_objs:
        for i in range(0,len(combinations)):
            combinations.append(combinations[i].union([obj]))
        combinations.append(set([obj]))
    print("all combinations generated")
    return combinations

def test_mse(all_comb):

    filedir = os.path.join(os.getcwd(), "outputs", "combined_t_latency.csv")
    df = pd.read_csv(filedir)
    all_comb2 = pd.Series(all_comb).apply(lambda x: list(x))
    dt = []
    rf = []
    et = []
    pca = []
    dt2 = []
    rf2 = []
    et2 = []
    pca2 = []
    for i in all_comb2:
        # 1 = loss
        # 2 = latency
        acc_loss = test_feat(df, i, 1)
        acc_latency = test_feat(df, i, 2)
        #print(accs)
        dt.append(acc_loss[0])
        rf.append(acc_loss[1])
        et.append(acc_loss[2])
        pca.append(acc_loss[3])
        
        dt2.append(acc_latency[0])
        rf2.append(acc_latency[1])
        et2.append(acc_latency[2])
        pca2.append(acc_latency[3])        
    
    dict1 = pd.DataFrame({'feat': all_comb, 'dt': dt, 'rf': rf, 'et': et, 'pca': pca})
    dict2 = pd.DataFrame({'feat2': all_comb, 'dt2': dt2, 'rf2': rf2, 'et2': et2, 'pca2': pca2})
    
    feat_df = pd.concat([dict1, dict2], axis=1).drop(['feat2'], axis=1)
    path = os.path.join(os.getcwd() , "outputs")
    feat_df.to_csv(os.path.join(path, "feat_df.csv"), index = False)
    
    # return feat_df

def best_performance():
    print("finding best performance")
    filedir = os.path.join(os.getcwd(), "outputs", "feat_df.csv")
    df = pd.read_csv(filedir)
    sorted = df.sort_values(by=['dt', 'rf', 'et', 'pca','dt2', 'rf2', 'et2', 'pca2'], ascending = False)
    
    #combined.to_csv("combined_latency.csv")
    # sorted.to_csv(os.path.join(path, "sorted_df.csv"), index = False)
    return sorted[:1]

