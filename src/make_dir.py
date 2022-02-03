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

def init_():
    temp_path = "data/temp"
    temp_c_path = "data/temp/tempdata_c"
    temp_r_path = "data/temp/tempdata_r"
    
    if not os.path.isdir('data/temp'):
        os.mkdir(temp_path)
        os.mkdir(temp_c_path)
        os.mkdir(temp_r_path)
        
    img_path = "outputs/eda"
    
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
        os.mkdir(img_path)
        
    out_path = "data/out"
    
    if not os.path.isdir('data/out'):
        os.mkdir(out_path)
