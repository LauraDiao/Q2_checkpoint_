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
    out_path = os.path.join(os.getcwd() , "outputs")
    #train_config = json.load(open('config/train.json'))
    transform_config = json.load(open('config/transform.json'))
    columns = json.load(open('config/columns.json'))
    eda_config = json.load(open('config/eda.json'))
    
    if 'data' in targets:
        gen(**transform_config)

    if 'eda' in targets:  
        gen(**transform_config)
        main_eda(**eda_config)

    if 'generate' in targets:
        gen(**transform_config)
        combs = getAllCombinations(**columns)
        test_mse(combs)
        best = best_performance()
        print("Found Best Performance: ")
        print(best)

    if 'tune' in targets: 
        print("tune?")
    
    if 'all' in targets: 
        print("running target " + "data")
        gen(**transform_config)

        print("running target " + "eda")
        main_eda(**eda_config)

        print("running target " + "generate")
        combs = getAllCombinations(**columns)
        test_mse(combs)
        best = best_performance()
        print("Found Best Performance: ")
        print(best)

        print("running target" + 'tune')


if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
