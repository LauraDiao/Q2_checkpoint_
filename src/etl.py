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

from helper import *


def readfilerun(run):
    "reads in files from each run"
    #path =  os.path.join(os.getcwd())
    testdir = os.path.join(os.getcwd(), "test", "run" + str(run))
    dir_files = os.listdir(testdir)
    #print(dir_files) # listing all files in directory

    df_lst = []
    rawdf_lst = []
    subsetsec = []
    # loop over the list of csv files
    for f in dir_files:
        #print(f)
        filedir = os.path.join(os.getcwd(), "test", "run" + str(run), f)
        #dftest = open(filedir)
        df = pd.read_csv(filedir)
        
        # print the location and filename | print('Location:', f)
        filename = f.split("\\")[-1].strip("-iperf.csv")
        #print('File Name:', filename)
      
        # GET LOSS regex to get everything after last  "-"
        los = str(re.search('([^\-]+$)', filename).group(0))
        
        # GET LATENCY regex to everything between "_" and "-"
        lat = (re.search("_(.*?)-", f).group(0)).strip("_").strip("-")
#         print("run" + str(run) + ", loss: " + los, "latency: "+ lat)
#         print()
        
        # labelling the df for the loss
        tempdf1 = df.assign(loss = int(los))
        # labelling the df for latency
        tempdf2 = tempdf1.assign(latency = int(lat))
        # labelling the df for run 
        tempdf3 = tempdf2.assign(iteration = run)
        
        # data processing 
        df_cols = genfeat(tempdf3)
        
        time_scaled = time(df_cols)
        rawdf_lst.append(time_scaled)
        
        df_mid = time_scaled.iloc[20:40]
        subsetsec.append(df_mid)
        
        f_df = agg10(df_cols)
        df_lst.append(f_df)
        
    
    # concatenating all of the dataframes agg over 10 from one run
    newfeat = pd.concat(df_lst ,ignore_index=True)#.reset_index(drop = True)
    
    #new_filename = "run"+ str(run) + "_merged.csv"
    #new.to_csv(new_filename, index=False)
    
    # raw data
    newdf = pd.concat(rawdf_lst ,ignore_index=True)#.reset_index(drop = True)
    subset_newdf = pd.concat(subsetsec ,ignore_index=True)
    
    return newdf, subset_newdf, newfeat


def gen(runs):
    data = []
    datasubset = []
    transformed = []
    for i in runs: 
        data_i, datasubset_i, transformed_i = readfilerun(i)

        data.append(data_i)
        datasubset.append(datasubset_i)
        transformed.append(transformed_i)
        print("run " + str(i) + " generated")

    path = os.path.join(os.getcwd() , "outputs")

    combined_data = pd.concat(data , ignore_index=True)#.reset_index(drop = True)
    combined_data.to_csv(os.path.join(path, "combined_all_latency.csv"), index = False)
    
    combined_subset = pd.concat(datasubset , ignore_index=True)
    combined_subset.to_csv(os.path.join(path, "combined_subset_latency.csv"), index = False)
    
    combined_t = pd.concat(transformed, ignore_index=True)  
    combined_t.to_csv(os.path.join(path, "combined_t_latency.csv"), index = False)
    return combined_t