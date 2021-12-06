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

def test_feat(cond, df, cols, p, df_u): 
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    # col is feauture comb
    # p is for loss or latency
    # 1: loss  # 2 : latency
    #print(df.columns)
    X = df[cols]
    X2 = df_u[cols]

    if p == 1: 
        y = df.loss
        y2 = df_u.loss
    if p == 2: 
        y = df.latency
        y2 = df_u.latency
        
    # randomly split into train and test sets, test set is 80% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1)

    if unseen == 'unseen': 
        X_test = X2
        y_test = y2
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc1 = mean_squared_error(y_test, y_pred)
    
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


def getAllCombinations( cond_):
    lst =  ['total_bytes','max_bytes','proto', "1->2Bytes",'2->1Bytes'
                ,'1->2Pkts','2->1Pkts','total_pkts','number_ms', 'pkt_ratio','time_spread', 'pkt sum','longest_seq'
                ,'total_pkt_sizes']
    lst1 = ["max bytes", "longest_seq", "total_bytes"]
    lst2 = ["total_pkts", "number_ms", "byte_ratio"]
    if cond_ == 1:
        lst = lst1
    if cond_ == 2:
        lst = lst2 
    uniq_objs = set(lst)
    combinations = []
    for obj in uniq_objs:
        for i in range(0,len(combinations)):
            combinations.append(combinations[i].union([obj]))
        combinations.append(set([obj]))
    print("all combinations generated")
    return combinations

def test_mse(cond, all_comb1, all_comb2):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    filedir_unseen = os.path.join(os.getcwd(), "outputs", unseen + "combined_t_latency.csv")
    df_unseen = pd.read_csv(filedir_unseen)
    filedir = os.path.join(os.getcwd(), "outputs", "combined_t_latency.csv")
    
    df = pd.read_csv(filedir)
    all_comb1 = pd.Series(all_comb1).apply(lambda x: list(x))
    all_comb1 = pd.Series(all_comb2).apply(lambda x: list(x))
    dt = []
    rf = []
    et = []
    pca = []
    for i in all_comb1:
        acc_loss = test_feat(cond, df, i, 1, df_unseen)
        dt.append(acc_loss[0])
        rf.append(acc_loss[1])
        et.append(acc_loss[2])
        pca.append(acc_loss[3])

    dt2 = []
    rf2 = []
    et2 = []
    pca2 = []
    for i in all_comb2:
        # 1 = loss
        # 2 = latency
        acc_latency = test_feat(cond, df, i, 2, df_unseen)
        #print(accs)
        dt2.append(acc_latency[0])
        rf2.append(acc_latency[1])
        et2.append(acc_latency[2])
        pca2.append(acc_latency[3])        
    
    dict1 = pd.DataFrame({'feat': all_comb1, 'dt': dt, 'rf': rf, 'et': et, 'pca': pca})
    dict2 = pd.DataFrame({'feat2': all_comb2, 'dt2': dt2, 'rf2': rf2, 'et2': et2, 'pca2': pca2})
    
    #feat_df = pd.concat([dict1, dict2], axis=1).drop(['feat2'], axis=1)
    path = os.path.join(os.getcwd() , "outputs")
    dict1.to_csv(os.path.join(path, unseen + "feat_df1.csv"), index = False)
    dict2.to_csv(os.path.join(path, unseen + "feat_df2.csv"), index = False)
    
    # return feat_df

def best_performance(cond):
    unseen = ''
    if cond == 'unseen': 
        unseen = 'unseen'
    #print("finding best loss performance")
    filedir1 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df1.csv")
    df1 = pd.read_csv(filedir1)
    print( "\n")
    print("Loss Performance sorted from lowest to highest", "\n")
    print(df1.sort_values(by=['dt', 'rf', 'et', 'pca'], ascending = True)[:5], "\n")
    #print("Loss Performance sorted from highest to lowest")
    #print(df1.sort_values(by=['dt', 'rf', 'et', 'pca'], ascending = False)[:5])
    

    #print("finding best latency performance")
    filedir2 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df2.csv")
    df2 = pd.read_csv(filedir2)
    print( "\n")
    print("Latency Performance sorted from lowest to highest", "\n")
    print(df2.sort_values(by=['dt2', 'rf2', 'et2', 'pca2'], ascending = True)[:5], "\n")
    #print("Latency Performance sorted from highest to lowest")
    #print(df2.sort_values(by=['dt2', 'rf2', 'et2', 'pca2'], ascending = False)[:5])
    
    #combined.to_csv("combined_latency.csv")
    # sorted.to_csv(os.path.join(path, "sorted_df.csv"), index = False)


