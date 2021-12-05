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
    
    test_unseen = 'unseen'
    test_seen = 'seen'

    if 'data' in targets:
        """generating feat from unseen and seen data"""
        gen(test_seen, **transform_config)
        gen(test_unseen, **transform_config)

    if 'eda' in targets:  
        gen(test_seen, **transform_config)
        main_eda(test_seen, **eda_config)

    if 'generate' in targets:
        gen(test_seen, **transform_config)
        gen(test_unseen,**transform_config)
        
        combs = getAllCombinations(**columns)
        
        print("Testing on seen data: ")
        test_mse(test_seen, combs)
        best = best_performance(test_seen)
        print("Performance for seen data: ", "\n", best, "\n")
        
        print("Testing on unseen data: ")
        test_mse(test_unseen, combs)
        best2 = best_performance(test_unseen)
        print("Performance for unseen data: ", "\n", best2, "\n")
        

    if 'tune' in targets: 
        print("tune?")
    
    # if 'all' in targets: 
        

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
