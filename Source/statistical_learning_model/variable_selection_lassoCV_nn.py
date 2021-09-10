# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:23:14 2021

@author: Ryo
"""
# %% modules
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os 

import random

from matplotlib import pyplot as plt
from adjustText import adjust_text

from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
# =============================================================================
# %% FUNCTIONS
# =============================================================================
# utils funcs
def blliCI_willson(theta, alpha, n):
    # get normal quantile
    z = norm.ppf(1-alpha/2)
    z2 = z**2
    # constant
    ci_const = (1/(1 + z2/n)) * (theta + z2/(2*n))
    ci_openess = (z/(1 + z2/n)) * np.sqrt(theta * (1 - theta)/n + z2/(4*n**2))
    
    ci_lower = ci_const - ci_openess 
    ci_upper = ci_const + ci_openess
    
    return(ci_lower, ci_upper) 
    
# get reproducible results in keras
def reset_random_seeds(seed, do_print=False):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
   if(do_print):
       print("\n Reset random state with seed: " + str(seed))
       
   return()

# tidy dataframes
def get_dummies_results(df, var='y', prefix='y'):
    # working data frame
    df_work = df.copy()
    
    # one hot encode results
    y = pd.get_dummies(df_work[var], prefix=prefix, drop_first=False, dtype=int)
    names_y = y.columns.values
    
    # append to dataframe
    df_work[names_y] = y
    
    # reorder columns
    columns_names = df_work.columns.values
    columns_names = np.concatenate([names_y, columns_names[:-3]])
    df_work = df_work[columns_names]
    
    # drop result
    df_work.drop(columns=var, inplace=True)
    
    return(df_work, list(names_y))

def split_trainvaltest(df, queries, objective_var='y', var='x'):    
    # working dataframe
    df_work = df.copy()    
    
    # get train, validation and test
    dbb = {}    
    for k,v in queries.items():
        df_subset = df_work.query(v)
        
        # save objective and covariables in dataframe
        dbb[k] = {'y': df_subset[objective_var], 'X': df_subset[var]}
        
    return(dbb)
    
def tidydf_1(df, queries, objective_var='y', var='x'): 
    # working data frame
    df_work = df.copy()
    
    # get dummies
    df_work, names_y = get_dummies_results(df_work, var=objective_var, prefix=objective_var)
    
    # mutate matchweek to normalize it (lost generalization in functions)
    df_work['matchweek'] = df_work.groupby('season')['matchweek'].\
        transform(lambda x: (x - min(x))/(max(x) - min(x)))   
    
    # get train, validation and test
    dbb = split_trainvaltest(df_work, queries, objective_var=names_y, var=var)  
    
    return(dbb)

def tidydf_2(df, queries, objective_var='y', var='x'):
    # normalized columns
    COLS_BN = [
        'position_table_home', 'total_pts_home', 'npxGD_ma_home',
        'npxGD_var_home',  
        'position_table_away', 'total_pts_away', 'npxGD_ma_away',
        'npxGD_var_away', 
        'ova_home', 'att_home', 'mid_home', 'def_home', 'transfer_budget_home',
        'ip_home', 'saa_home', 'ova_away', 'att_away', 'mid_away',
        'def_away', 'transfer_budget_away', 'ip_away', 'saa_away'        
    ]
    
    # working data frame
    df_work = df.copy()
    
    # get dummies
    df_work, names_y = get_dummies_results(df_work, var=objective_var, prefix=objective_var)
    
    # normalize some entries by matchweek
    df_work[COLS_BN] = df_work.groupby('matchweek')[COLS_BN].apply(lambda x: (x-x.mean())/x.std())
    
    # get train, validation and test
    dbb = split_trainvaltest(df_work, queries, objective_var=names_y, var=var)  
    
    return(dbb)

def tidydf_3(df, queries, objective_var='y', var='x'):
    # normalized columns
    COLS_BN = [
       'position_table_ad',
       'total_pts_ad', 'npxGD_ma_ad', 'ip_ad', 'saa_ad'
    ]
    
    # working data frame
    df_work = df.copy()
    
    # get dummies
    df_work, names_y = get_dummies_results(df_work, var=objective_var, prefix=objective_var)
    
    # normalize some entries by matchweek
    df_work[COLS_BN] = df_work.groupby('matchweek')[COLS_BN].apply(lambda x: (x-x.mean())/x.std())
    
    # get train, validation and test
    dbb = split_trainvaltest(df_work, queries, objective_var=names_y, var=var)  
    
    return(dbb)
    
# keras shallow model
def CVlasso(dbb, nsplit_cv=5, alpha=0.05, ntest_size=1, lambda_val=0, metric='accuracy', seed=42, verbose=True):  
    # merge train and validation in one dataframe
    X_list = [v2 for k1, dics in dbb.items() if k1 != 'test' for k2, v2 in dics.items() if k2 == "X"]
    Y_list = [v2 for k1, dics in dbb.items() if k1 != 'test' for k2, v2 in dics.items() if k2 == "y"]
    
    X_data = pd.concat(X_list)
    Y_data = pd.concat(Y_list)
    
    # save 
    array_errors = np.zeros((nsplit_cv, 2))
    
    # split data in nsplit_cv-folds
    if verbose:
        print("fold: ", end='')

    tscv = TimeSeriesSplit(n_splits=nsplit_cv, test_size=ntest_size, gap=0)
    i = 0
    for index_train, index_val in tscv.split(X_data):        
        # save train/val database
        dbb_val = {
            'train': {
                'X': X_data.iloc[index_train],
                'y': Y_data.iloc[index_train]            
            },
            'validation': {
                'X': X_data.iloc[index_val],
                'y': Y_data.iloc[index_val]            
            }     
        }
        
        # get a look in the number of iterations
        if verbose:
            print(str(i+1) + ", ", end="")
        
        # train model and get validation loss and accuracy
        _, err_val, _ = get_model(dbb_val, lambda_val=lambda_val, metric_stop=metric, seed=seed) 
        
        # append metrics
        array_errors[i] = np.array(err_val)
        
        # update index
        i += 1
        

    # get statistics of the metrics
    # loss
    mean_loss = np.mean(array_errors[:, 0], axis=0)
    std_loss = np.std(array_errors[:, 0], axis=0)
    
    # acc
    mean_acc = np.mean(array_errors[:, 1], axis=0)
    ci_acc_low, ci_acc_up = blliCI_willson(mean_acc, alpha=alpha, n=nsplit_cv)   
    
    # return    
    loss_stats = np.array((mean_loss, std_loss))
    acc_stats = np.array((mean_acc, ci_acc_low, ci_acc_up))    
    
    return(loss_stats, acc_stats)

def get_model(dbb, lambda_val=0, metric_stop='accuracy', seed=42):
    # init randomness
    reset_random_seeds(seed=seed, do_print=False)
    
    # get train database
    X_train=dbb.get('train')['X']
    Y_train=dbb.get('train')['y']
    
    # number of classes, covars and observations
    n_class = Y_train.shape[1]
    n_covars, n_obs = X_train.shape
    
    # create model
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            # initialization
            units= n_class, 
            activation='softmax',
            kernel_initializer=keras.initializers.GlorotNormal(), 
            # lasso punishment
            kernel_regularizer=keras.regularizers.l1(lambda_val),
            # kwargs
            dtype=np.float64
        )
    )
    
    # compile model
    model.compile(
            optimizer='Nadam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # early stopping
    if metric_stop == 'accuracy':        
        # train model
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            mode='min', 
            min_delta=0,
            patience=10,
            restore_best_weights=True 
        )        
    else: 
        # train model
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            min_delta=0.01,
            patience=10,
            restore_best_weights=True 
        )
    
    # train model
    model.fit(
        x=X_train,
        y=Y_train,
        epochs=1000,
        batch_size=n_obs, #complete gradient
        shuffle=False,
        verbose=0,
        callbacks=[early_stop],
        validation_data=(dbb.get('validation')['X'], dbb.get('validation')['y']),
        validation_batch_size=len(dbb.get('validation')['X'])    #complete gradient               
    )
    
    # get errors (loss and accuracy)
    err_train = model.evaluate(X_train, Y_train, verbose=0)
    err_val = model.evaluate(dbb.get('validation')['X'], dbb.get('validation')['y'], verbose=0)
    theta = model.get_weights()    
    
    return(err_train, err_val, theta)

def CVlasso_models(dbb, lambdas=np.power(10, np.linspace(1, -5)), nsplit_cv=5, alpha=0.05, ntest_size=1, metric='accuracy', verbose=True, seed=48):
    # print loops
    if verbose:
        print("temporal CV of " + str(nsplit_cv) + " folds\n\n")
    
    # save accuracy and loss
    df_lossstats = pd.DataFrame(columns=("mean_loss", "std_loss"))
    df_accstats = pd.DataFrame(columns=("mean_acc", "ci_lower", "ci_upper"))
    
    n_lambdas = len(lambdas)
    
    # train and get best lambda model
    for i, lambd in enumerate(lambdas):
        # print number of iteration
        if verbose:
            print('\n\ntraining model no.' + str(i+1) + ' of ' + str(n_lambdas))
        
        # train and test model for each lambda in temporal cross validation
        df_lossstats.loc[i], df_accstats.loc[i] = CVlasso(
            dbb, 
            nsplit_cv=nsplit_cv, 
            alpha=alpha, 
            ntest_size=ntest_size, 
            lambda_val=lambd, 
            metric=metric, 
            seed=seed
        )   
        
    df_lossstats["lambdas"] = lambdas
    df_accstats["lambdas"] = lambdas
        
    # get best lambda
    imax = df_accstats['mean_acc'].argmax()
    best_acc = df_accstats.loc[imax, 'mean_acc']
    lambda_opt = lambdas[imax]
    
    # get best sparser model lambda in same CI 
    # for bigger lambda
    imax_sparse = np.where(best_acc <= df_accstats.loc[imax:,"ci_upper"])[0][-1] + imax
    lambda_sparse = lambdas[imax_sparse]
        
    dic_out = {
        'stats': {
            'loss': df_lossstats,
            'acc': df_accstats
        },
        'lambdas': {
            'optimal': lambda_opt,
            'sparser': lambda_sparse
        }
    }
    return(dic_out)

# plots
def pltCV_lossacc(df_loss, df_acc, lambs):
    # get lambdas
    lambdas = df_loss['lambdas'].copy()
    
    #plot
    # plot loss
    fig, ax1 = plt.subplots()

    mu = df_loss['mean_loss']
    sigma = df_loss['std_loss']
    ax1.plot(lambdas, mu, color='black')
    ax1.fill_between(lambdas, mu + sigma, mu - sigma, facecolor='gray', alpha=0.5)
    
    ax1.set_xscale('log')
    ax1.set_ylabel('Pérdida')
    ax1.set_xlabel('$\lambda$')
    ax1.tick_params(axis='y')
    ax1.legend()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # plt accuracy
    theta = df_acc["mean_acc"]
    thetaCI_l = df_acc['ci_lower']
    thetaCI_u = df_acc['ci_upper']
    thetaCI = np.array(thetaCI_l, thetaCI_u).transpose()
    
    ax2.errorbar(
        lambdas,
        theta, 
        yerr=thetaCI, 
        capsize=3,
        elinewidth=1,
        color ='black', 
        ecolor='gray', 
        fmt='o'
    )
    
    ax2.set_xscale('log')
    ax2.set_ylabel('Precisión')
    ax2.tick_params(axis='y')
    
    # merge plots
    fig.tight_layout()  
    
    # add lambdas
    plt.axvline(x=lambs['optimal'], color='lightgray', linestyle='--')
    plt.axvline(x=lambs['sparser'], color='lightgray', linestyle='--')
    
    plt.show()
    
    return()

def pltW_objectiveY(dic, lambdas, lambda_model, threshold_factor=1):
    # get cube of the weights of the output (unique) layer
    W_lambda_list = [
        v for theta in list(dic.values()) 
            for k,v in theta.items() if k == 'W'
    ]
    W_lambda3D = np.stack(W_lambda_list, axis=2)
    
    n_covars, n_y, n_lambdas = W_lambda3D.shape
    NAMES_Y = ('visita', 'empate', 'casa')
    
    
    # plot each covariable by type of objective var
    for i in range(n_y):
        W_yi =  W_lambda3D[:, i, :].transpose()
        
        # plot weight decayment
        fig = plt.figure()
        ax = fig.add_subplot(111)
        texts = []
        # append number of the covariable
        ilambda = np.argmax(lambdas >= lambda_model)
        # Wi_std = np.std(W_yi[ilambda])
        # threshold = threshold_factor * Wi_std  
        threshold = threshold_factor   
                # if it is an "important" variable (with respect to their mean)
        for j in range(n_covars):
            if np.abs(W_yi[ilambda, j]) >= threshold:
                ax.plot(lambdas, W_yi[:, j], color = 'black', alpha=0.9)
                texts.append(plt.text(0.7*lambdas[0], W_yi[0, j], s=str(j+1), fontsize=8))
            else:
                ax.plot(lambdas, W_yi[:, j], color = 'gray', alpha=0.5)    
                
        adjust_text(texts, only_move={'points':'y', 'text':'y'}, autoalign='y', force_points=0)
        
        # show lambda of the model
        plt.axvline(x=lambda_model, linestyle='--', color='gray')        
        
        ax.set_xscale('log')
        plt.xlim(0.5*lambdas[0], 1)
        plt.ylabel('$W$')
        plt.xlabel('$\lambda$')
        plt.title('Decaimiento de los pesos para y: ' + NAMES_Y[i])
        plt.show()   
        
    return()
    
def pltW_covars(dic, lambdas, lambda_model, vars_names=[]):
    # format in plots
    LINESTYLES = ['--', ':', '-']
    NAMES_Y = ('visita', 'empate', 'casa')
    
    # get cube of the weights of the output (unique) layer
    W_lambda_list = [
        v for theta in list(dic.values()) 
            for k,v in theta.items() if k == 'W'
    ]
    W_lambda3D = np.stack(W_lambda_list, axis=2)
    
    n_covars, n_y, n_lambdas = W_lambda3D.shape
    
    # plot each covariable weight for all y classes
    for j in range(n_covars):
        W_x = W_lambda3D[j]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # plot each layer
        for i in range(n_y): # 3 y's (classes): away, draw, home
            ax.plot(lambdas, W_x[i], label=NAMES_Y[i], color = 'black', linestyle=LINESTYLES[i])
        
        # show lambda of the model
        plt.axvline(x=lambda_model, linewidth=0.4, color='gray')  
        
        ax.set_xscale('log')
        ax.legend()
        plt.xlim(lambdas[0], 1)
        plt.ylabel('$W_{{%s}}$' % str(j+1))
        plt.xlabel('$\lambda$')
        plt.title('Pesos de la variable: ' + vars_names[j] + ' (var '+ str(j+1) + ')')
        plt.show()
        
    return()


# =============================================================================
# %% MAIN
# =============================================================================
#%% setup ---------------------------------------------------------------------
# init params
SEED_VALUE = 8

# read data
os.chdir('C:\\Users\\Ryo\\Documents\\Estudios\\ITAM\\Tesis\\sportsAnalytics\\tesis\\Repository')
df_myscale = pd.read_csv('Data\\Main_DBB\\model_myscale.csv')


# %% model original
### init params
vars_model = [
    'matchweek', 'position_table_home', 'total_pts_home',
    'npxGD_ma_home', 'npxGD_var_home', 'big_six_home',
    'promoted_team_home', 'position_table_away', 'total_pts_away',
    'npxGD_ma_away', 'npxGD_var_away', 'big_six_away',
    'promoted_team_away', 'ova_home','att_home', 'mid_home', 'def_home', 
    'transfer_budget_home', 'ip_home', 'saa_home',
    'ova_away','att_away', 'mid_away', 'def_away',
    'transfer_budget_away', 'ip_away', 'saa_away', 'proba_h', 'proba_d','proba_a'
   ]

queries_model = {
    'train':  'season < 20',
    'validation': 'season >= 20 & matchweek < 0.45', # matchweek 18 (standarized)
    'test': 'season >= 20 & matchweek >= 0.45'         # normalized(18; max=38, min=2) = (18-2)/(38-2)
    }

#### shallow nn
# get data base
dbb1 = tidydf_1(df_myscale, queries=queries_model, objective_var='result', var=vars_model)

# build models with lasso regression
lambdas = np.power(10, np.linspace(-3, 0, num=5))
dic_CVLasso = CVlasso_models(dbb1, lambdas=lambdas, nsplit_cv=17, alpha=0.5, ntest_size=10, seed=SEED_VALUE)


# %% plot
df_lossCV = dic_CVLasso.get('stats')['loss']
df_accCV = dic_CVLasso.get('stats')['acc']
best_lambdas = dic_CVLasso.get('lambdas')

pltCV_lossacc(df_lossCV, df_accCV, best_lambdas)























