import numpy as np
import json
import sys
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

from helper import *

def getAllCombinations( cond_):
    lst =  ['total_bytes','max_bytes', "1->2Bytes",'2->1Bytes'
                ,'1->2Pkts','2->1Pkts','total_pkts','number_ms', 'pkt_ratio','time_spread', 'pkt sum','longest_seq'
                ,'total_pkt_sizes', 'mean_tdelta', 'max_tdelta']# 'proto',
    lst1 = ["max_bytes", "longest_seq", "total_bytes" ] # more later
    lst2 = ["total_pkts", "number_ms", "byte_ratio"]
    latency_lst = [ "byte_ratio", 'max_tdelta', 'pkt_ratio', 'time_spread_min', ]
    loss_lst = ['total_pkts', 'pkt sum', 'total_pkt_sizes', '2->1Bytes', 'total_pkts_max', 'number_ms', 'mean_tdelta'] #  'total_bytes', '2->1Pkts', 
    
    if cond_ == 1:
        lst = loss_lst
    if cond_ == 2:
        lst = latency_lst
    uniq_objs = set(lst)
    combinations = []
    for obj in uniq_objs:
        for i in range(0,len(combinations)):
            combinations.append(combinations[i].union([obj]))
        combinations.append(set([obj]))
    print("all combinations generated")
    return combinations