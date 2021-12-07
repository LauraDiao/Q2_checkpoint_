import sys
import json
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


sys.path.insert(0, 'src')

from etl import *
from helper import *
from train import *
from eda import *

def main(targets):

    transform_config = json.load(open('config/transform.json'))
    columns = json.load(open('config/columns.json'))
    eda_config = json.load(open('config/eda.json'))
    all_config = json.load(open("config/all.json"))

    test_unseen = 'unseen'
    test_seen = 'seen'

    if 'data' in targets:
        """generating feat from unseen and seen data"""
        print("transforming seen data")
        gen(test_seen, **transform_config)
        print("transforming unseen data")
        gen(test_unseen, **transform_config)

    if 'eda' in targets:  
        print("transforming seen data")
        gen(test_seen, **transform_config)
        main_eda(test_seen, **eda_config)
        print("EDA saved to outputs/eda/ folder")

    if 'train' in targets:
        "trains tests in this target"
        print("transforming seen data")
        gen(test_seen, **transform_config)
                
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on seen data: ")
        test_mse(test_seen, comb1, comb2)
        best_performance(test_seen)
                        
    if "inference" in targets: 
        print("transforming seen data")
        gen(test_seen, **transform_config)
        print("transforming unseen data")
        gen(test_unseen, **transform_config)
        
        comb1 = getAllCombinations(1)
        comb2 = getAllCombinations(2)
        
        print("Testing on unseen data: ")
        test_mse(test_unseen, comb1, comb2)
        best_performance(test_unseen)
            
    if "test" in targets: 
        """ runs all targets on sample data"""
        print("transforming seen data")
        gen(test_seen, **transform_config)
        print("transforming unseen data")
        gen(test_unseen, **transform_config)
        main_eda(test_seen, **eda_config)
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
        print("transforming seen data")
        gen(test_seen, **all_config)
        print("transforming unseen data")
        gen(test_unseen, **all_config)
        main_eda(test_seen, **eda_config)
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
