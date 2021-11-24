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

def test_feat(cols, df): 
    combined = df
    X = combined[cols]
    y = combined.latency
    # randomly split into train and test sets, test set is 80% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    
    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc1 = mean_squared_error(y_test, y_pred)
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

def test_mse(all_comb, df):
    all_comb2 = pd.Series(all_comb).apply(lambda x: list(x))
    dt = []
    rf = []
    et = []
    pca = []
    counter = 0
    for i in all_comb2:
        accs = test_feat(i, df)
        #print(accs)
        dt.append(accs[0])
        rf.append(accs[1])
        et.append(accs[2])
        pca.append(accs[3])
        counter +=1
        #print("combination:" + str(counter))

    dictin = {'feat': all_comb, 'dt': dt, 'rf': rf, 'et': et, 'pca': pca}
   
    feat_df = pd.DataFrame(dictin)
    return feat_df 

def best_performance(df):
    print("finding best performance")
    sorted = df.sort_values(by=['dt', 'rf', 'et', 'pca'], ascending = False)
    path = os.path.join(os.getcwd() , "outputs")
    #combined.to_csv("combined_latency.csv")
    sorted.to_csv(os.path.join(path, "sorted_df.csv"), index = False)
    return sorted[:1]

