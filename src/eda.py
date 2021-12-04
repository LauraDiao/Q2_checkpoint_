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
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")

def main_eda(lst, filen1, filen2 ):
    fpath1 = os.path.join(os.getcwd() , "outputs", filen1)
    df_1 = pd.read_csv(fpath1)
    plottogether(lst, df_1, filen1.strip(".csv"))

    fpath2 = os.path.join(os.getcwd() , "outputs", filen2)
    df_2 = pd.read_csv(fpath2)
    plotloss(df_2)

def plot_main4(df_1, l1, df_2, l2, picname):
    
    #separating all of the aggregates for each loss

    tp_sum_agg, tb_agg, tp_agg = main2(df_1)

    tp_sum_agg2, tb_agg2, tp_agg2 = main2(df_2)

    label = 'latency'
  
    fig, axes = plt.subplots(3, 2,figsize=(18, 10))#, sharex=True)
    #print()
    sns.lineplot(ax=axes[0, 0], x = 'Second', y = 'sum', data = tp_sum_agg, hue = label)
    axes[0, 0].set_title("Packets Per Second over latency for run " + l1)
    sns.lineplot(ax=axes[0, 1], x = 'Second', y = 'sum', data = tp_sum_agg2, hue = label)
    axes[0, 1].set_title("Packets Per Second over latency for run " + l2)
    
    sns.lineplot(ax=axes[1, 0], x = 'Second', y = 'sum', hue = label , data = tb_agg)
    axes[1, 0].set_title("Bytes over packet latency for run " + l1)
    sns.lineplot(ax=axes[1, 1], x = 'Second', y = 'sum', hue = label , data = tb_agg2)
    axes[1, 1].set_title("Bytes over packet latency for run " + l2)

    sns.lineplot(ax=axes[2, 0], x = 'Second', y = 'sum', hue = label , data = tp_agg)
    axes[2, 0].set_title("Pkts over packet loss for run " + l1)
    sns.lineplot(ax=axes[2, 1], x = 'Second', y = 'sum', hue = label , data = tp_agg2)
    axes[2, 1].set_title("Pkts over packet loss for run " + l2)
    plt.subplots_adjust(hspace = 0.8)
    #savefig
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda","latency_trends_" + picname + ".png")
    fig.savefig(saveto)
    print("end of main4")

def plottogether(lst, df_e, picname): 
    leftrun = lst[0]
    rightrun = lst[1]
    values = df_e['iteration'].unique()
    # print(values)
    subset1 = df_e[df_e['iteration'] == leftrun]
    subset2 = df_e[df_e['iteration'] == rightrun]
    # print(subset1.shape)
    # print(subset2.shape)
    plot_main4(subset1, str(leftrun), subset2, str(rightrun), picname)

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
        source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def plotloss(df):
    warnings.filterwarnings("ignore")

    fig, axes = plt.subplots(3, 3, figsize=(10, 15))

    X = df.drop(['latency'], axis=1)
    y = df.latency

    title = "Learning Curves (DecisionTree)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = DecisionTreeRegressor()
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    title = r"Learning Curves (RandomForest)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = RandomForestRegressor(n_estimators=10)
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    title = r"Learning Curves (ExtraTrees)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = ExtraTreesRegressor(n_estimators=10)
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 2], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    #plt.show()
    #savefig
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda","learning_curves.png")
    fig.savefig(saveto)