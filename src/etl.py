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
        tdf = tempdf3
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
        
        # input is 10 seconds of network traffic, so to predict for a period of time, 
        # need to aggregate features over 10 seconds
        f_df = agg10(tdf)
        
        df_lst.append(f_df)
    
    # concatenating all of the dataframes from one run
    new = pd.concat(df_lst ,ignore_index=True)#.reset_index(drop = True)
    #new_filename = "run"+ str(run) + "_merged.csv"
    #new.to_csv(new_filename, index=False)
        
    return new


def gen(runs):
    dfs = []
    for i in runs: 
        dfs.append(readfilerun(i))

    combined = pd.concat(dfs , ignore_index=True)#.reset_index(drop = True)
    #combined.to_csv("combined_latency.csv")
    return combined