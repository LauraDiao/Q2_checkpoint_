import numpy as np
import json
import sys
import pandas as pd
import os
import glob
import re
import time 
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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
from helper import *
from eda import *
from train import *
from etl import *

def main(targets):
    init_()

    transform_config = json.load(open('config/transform.json'))
    eda_config = json.load(open('config/eda.json'))
    all_config = json.load(open("config/all.json"))

    test_unseen = 'unseen'
    test_seen = 'seen'
    
    cond1 = True
    cond2 = False

    if 'data' in targets:
        """generating feat from unseen and seen data"""
        readfilerun('data/raw/train_r', 'data/temp/tempdata_r') # TODO uncomment
        gen(test_seen, 'tempdata_r', **transform_config)
        readfilerun('data/raw/train_c', 'data/temp/tempdata_c')
        gen(test_unseen, 'tempdata_c', **transform_config)

    if 'eda' in targets:  
        # readfiledrun and gen for seen data, refer to data target
        print('plotting seen data')
        main_eda(test_seen, [200, 300], **eda_config)
        print('plotting un seen data')
        main_eda(test_unseen, [200, 300], **eda_config)
        print("EDA saved to outputs/eda/ folder")

    if 'train' in targets:
        "trains tests in this target"
        readfilerun('data/raw/train_r', 'data/temp/tempdata_r') # TODO uncomment
        gen(test_seen, 'tempdata_r', **transform_config)
                
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on seen data: ")
        test_mse(test_seen, comb1, comb2)
        best_performance(test_seen)
        

    if "inference" in targets: 
        readfilerun('data/raw/train_c', 'data/temp/tempdata_c')
        gen(test_unseen, 'tempdata_c', **transform_config)
        
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on unseen data: ")
        test_mse(test_unseen, comb1, comb2)
        best_performance(test_unseen)
            
    if "test" in targets: 
        """ runs all targets on sample data"""

        readfilerun('data/raw/train_r', 'data/temp/tempdata_r') # TODO uncomment
        gen(test_seen, 'tempdata_r', **transform_config)
        readfilerun('data/raw/train_c', 'data/temp/tempdata_c')
        gen(test_unseen, 'tempdata_c', **transform_config)
       
        print('plotting seen data')
        main_eda(test_seen, [200, 300], **eda_config)
        print('plotting un seen data')
        main_eda(test_unseen, [200, 300], **eda_config)
        print("EDA saved to outputs/eda/ folder")
        
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on seen data: ")
        test_mse(test_seen, comb1, comb2)
        best_performance(test_seen)
        
        print("Testing on unseen data: ")
        test_mse(test_unseen, comb1, comb2)
        best_performance(test_unseen)
        
    if 'all' in targets: 
        """ runs all targets on all data"""
        readfilerun('data/raw/train_r', 'data/temp/tempdata_r') # TODO uncomment
        gen(test_seen, 'tempdata_r', **all_config)
        readfilerun('data/raw/train_c', 'data/temp/tempdata_c')
        gen(test_unseen, 'tempdata_c', **all_config)
       
        print('plotting seen data')
        main_eda(test_seen, [200, 300], **eda_config)
        print('plotting un seen data')
        main_eda(test_unseen, [200, 300], **eda_config)
        print("EDA saved to outputs/eda/ folder")
        
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on seen data: ")
        test_mse(test_seen, comb1, comb2)
        best_performance(test_seen)
        
        print("Testing on unseen data: ")
        test_mse(test_unseen, comb1, comb2)
        best_performance(test_unseen)
        

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
