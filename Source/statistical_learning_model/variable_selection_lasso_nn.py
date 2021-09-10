# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:54:22 2021

@author: Roman Alberto Velez Jimenez

Explore the possible best combination of variables and interactions
for the neural network by creating a shallow neural network (multiple multivariable
 regression) that is the same than a generalizaed linear model.

The main idea is to explore the power of prediction from each variable by a 
penalization technique called Lasso.
"""
# %% modules
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
import os 
import random

from matplotlib import pyplot as plt
from adjustText import adjust_text

# =============================================================================
# %% FUNCTIONS
# =============================================================================
# get reproducible results in keras
# look: https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-
# results-in-keras-even-though-i-set-the-random-seeds
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

def lasso_models(dbb, lambdas=np.power(10, np.linspace(1, -5)), metric='accuracy', verbose=True, seed=48):
    # save accuracy and loss
    array_errors = np.zeros((len(lambdas), 5))
    dic_weights = {}
    n_lambdas = len(lambdas)
    
    # train and get best lambda model
    for i, lambd in enumerate(lambdas):
        # print number of iteration
        if i % 5 == 0 and verbose:
            print('training model no.' + str(i+1) + ' of ' + str(n_lambdas))
        
        # train and test model for each lambda
        err_train, err_val, theta = get_model(dbb, lambda_val=lambd, metric_stop=metric, seed=seed)    
        
        # append values
        # row by: lambda, loss train, loss val, accuracy train, accuracy validation
        array_errors[i] = np.array([lambd, err_train[0], err_val[0], err_train[1], err_val[1]])
        dic_weights['lambda_' + str(i+1)] = {'W': theta[0], 'b': theta[1]}
        
    # get best lambda
    df_errors = pd.DataFrame(
        array_errors, 
        columns=['lambdas', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
    )

    # max or min lambda if metric is accuracy or loss
    imax = df_errors['val_acc'].argmax()
    lambda_opt = lambdas[imax]

        
    dic_out = {
        'df_lambdas': df_errors,
        'dic_thetas': dic_weights,
        'lambda_best': lambda_opt
    }
    return(dic_out)


# plots
def plt_lossacc(df):
    # get lambdas
    lambdas = df['lambdas'].copy()
    
    #plot
    # plot loss
    fig, ax1 = plt.subplots()

    ax1.plot(lambdas, df['train_loss'], label='train', color = 'black')
    ax1.plot(lambdas, df['val_loss'], label='validation', color = 'gray')
    
    ax1.set_xscale('log')
    ax1.set_ylabel('Pérdida')
    ax1.set_xlabel('$\lambda$')
    ax1.tick_params(axis='y')
    ax1.legend()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # plt accuracy
    ax2.plot(lambdas, df['train_acc'], label='train', color = 'black', linestyle='dashed')
    ax2.plot(lambdas, df['val_acc'], label='validation', color = 'gray', linestyle='dashed')
    
    ax2.set_xscale('log')
    ax2.set_ylabel('Precisión')
    ax2.tick_params(axis='y')
    ax2.legend()
    
    # merge plots and show
    fig.tight_layout()  
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
    ### initial values --------------------------------------------------------
SEED_VALUE = 8

    ### read ------------------------------------------------------------------
# set working dir (bad practice but util)
os.chdir('C:\\Users\\Ryo\\Documents\\Estudios\\ITAM\\Tesis\\sportsAnalytics\\tesis\\Repository')

# read data
df_myscale = pd.read_csv('Data\\Main_DBB\\model_myscale.csv')
df_original = pd.read_csv('Data\\Main_DBB\\model_original.csv')
df_interactions = pd.read_csv('Data\\Main_DBB\\model_interactions.csv')


# =============================================================================
# %% PLAYGROUND
# =============================================================================
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
lambdas = np.power(10, np.linspace(-4, 1, num=10))
dic_modelsLasso = lasso_models(dbb1, lambdas=lambdas, metric='loss', seed=8)

#### plots
# accuracy & loss
plt_lossacc(dic_modelsLasso['df_lambdas'])

# W's
pltW_objectiveY(dic_modelsLasso['dic_thetas'], lambdas, lambda_model=dic_modelsLasso['lambda_best'], threshold_factor=0.01)
pltW_covars(dic_modelsLasso['dic_thetas'], lambdas, lambda_model=dic_modelsLasso['lambda_best'], vars_names=vars_model)

# # %%look names of variables
# for i, name in enumerate(vars_model):
#     print('Var: ' + str(i+1) + ', ' + name)

# %% model original with batch normalization
### init params
vars_model2 = [
    'position_table_home', 'total_pts_home',
    'npxGD_ma_home', 'npxGD_var_home', 'big_six_home',
    'promoted_team_home', 'position_table_away', 'total_pts_away',
    'npxGD_ma_away', 'npxGD_var_away', 'big_six_away',
    'promoted_team_away', 'ova_home','att_home', 'mid_home', 'def_home', 
    'transfer_budget_home', 'ip_home', 'saa_home',
    'ova_away','att_away', 'mid_away', 'def_away',
    'transfer_budget_away', 'ip_away', 'saa_away', 'proba_h', 'proba_d','proba_a'
   ]

queries_model2 = {
    'train':  'season < 20',
    'validation': 'season >= 20 & matchweek < 18',
    'test': 'season >= 20 & matchweek >= 18'
    }

#### shallow nn
# get data base
dbb2 = tidydf_2(df_original, queries=queries_model2, objective_var='result', var=vars_model2)

# build models with lasso regression
dic_modelsLasso2 = lasso_models(dbb2, lambdas=lambdas, metric='loss', seed=8)

#### plots
# accuracy & loss
plt_lossacc(dic_modelsLasso2['df_lambdas'])

# W's
pltW_objectiveY(dic_modelsLasso2['dic_thetas'], lambdas, lambda_model=dic_modelsLasso2['lambda_best'], threshold_factor=0.01)
pltW_covars(dic_modelsLasso2['dic_thetas'], lambdas, lambda_model=dic_modelsLasso2['lambda_best'], vars_names=vars_model2)

# %% model with interactions
### init params
vars_model3 = [
       'big_six_ad', 'promoted_team_ad', 'position_table_ad',
       'total_pts_ad', 'npxGD_ma_ad', 'ip_ad', 'saa_ad', 'proba_h',
       'proba_d', 'proba_a'
   ]

queries_model3 = queries_model2

#### shallow nn
# get data base
dbb3 = tidydf_3(df_interactions, queries=queries_model3, objective_var='result', var=vars_model3)

# build models with lasso regression
dic_modelsLasso3 = lasso_models(dbb3, lambdas=lambdas, metric='loss', seed=8)

#### plots
# accuracy & loss
plt_lossacc(dic_modelsLasso3['df_lambdas'])

# W's
pltW_objectiveY(dic_modelsLasso3['dic_thetas'], lambdas, lambda_model=dic_modelsLasso3['lambda_best'], threshold_factor=0.01)
pltW_covars(dic_modelsLasso3['dic_thetas'], lambdas, lambda_model=dic_modelsLasso3['lambda_best'], vars_names=vars_model3)
























































