# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:40:06 2021

@author: Roman Alberto Velez Jimenez

Main library for the training and testing of the neural networks for the
prediction of the EPL.

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

import os
import random

from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.ticker as mtick

from sklearn import metrics as sk_metrics
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# functions
# =============================================================================

#%% utils
def _reset_random_seeds(seed):
    """
    Reset random seed for Numpy, TensorFlow and Random

    Parameters
    ----------
    seed : int

    Returns
    -------
    None.

    """
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return()

def get_dummies_results(df, var='y', prefix='y'):
    """
    One hot encoding

    Parameters
    ----------
    df : Dataframe
        Dataframe with the variable to modify
    var : char, optional
        Name of the column of the variable to dummies. The default is 'y'.
    prefix : char, optional
        Prefix of the dummies. The default is 'y'.

    Returns
    -------
    Dataframe df with the dummies of 'var' with the prefix 'prefix'

    """

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
    """
    Split a dataframe in train/validation/test

    Parameters
    ----------
    df : Dataframe
    queries : dictionary
        SQL like query to the database. The key name is the name of the
        dataframe that would be saved.
    objective_var : char, optional
        Name of the objective variable. The default is 'y'.
    var : list, string, optional
        List/stringo f the covariables column(s). The default is 'x'.

    Returns
    -------
    Dictionary with the

    """
    # working dataframe
    df_work = df.copy()

    # get train, validation and test
    dbb = {}
    for k,v in queries.items():
        df_subset = df_work.query(v)

        # save objective and covariables in dataframe
        dbb[k] = {'y': df_subset[objective_var], 'X': df_subset[var]}

    return(dbb)

def _printProgressBar (
        iteration, total, prefix = 'progress:',
        suffix = 'complete', decimals = 1, length = 10,
        fill = '█', printEnd = "\r"
        ):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    Function made by Greenstick
    https://stackoverflow.com/a/34325723
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def blliCI_willson(theta, n, alpha=0.05):

    """
    Wilson Confidence Interval for a Blli parameter

    Parameters
    ----------
    theta : float
        Estimated parameter
    n : int
        Number of observations
    alpha : float, optional
        Degree of significance. The default is 0.05.

    Returns
    -------
    tuple of floats. Lower and upper 1-alpha % confidence interval.

    """
    # get normal quantile
    z = norm.ppf(1-alpha/2)
    z2 = z**2
    # constant
    ci_const = (1/(1 + z2/n)) * (theta + z2/(2*n))
    ci_openess = (z/(1 + z2/n)) * np.sqrt(theta * (1 - theta)/n + z2/(4*n**2))

    ci_lower = ci_const - ci_openess
    ci_upper = ci_const + ci_openess

    return(ci_lower, ci_upper)

def tidy_bdd(df, queries, objective_var='y', var='x'): 
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

#%% neural networks
def build_model(dbb, lambda_val=0, metric_stop='accuracy', seed=42, return_model=False):
    """
    Build keras shallow lasso neural network. Train and validate data in the
    database given (dbb).

    The kernel initializer is GlorotNormal.
    The validation threshold is 0.01 in loss with 10 cycles of patience. It restores
    the best weights.

    Parameters
    ----------
    dbb : dict
        Dictionary of dictionaries. First dictionary should have a 'train'
        and 'validation' keys, which are the databases where the model should
        fit and validate (test) the data. Each dictionary should have a 'X' and 'y' keys,
        which are the covariables and the objective variables, respectively.
    lambda_val : float, optional
        Lasso penalization magnitude. The default is 0.
    metric_stop : str, optional
        The neural network stops training in the next 10 epochs where there
        isn't an observed improvement. The default is 'accuracy'.
    seed : int, optional
        Random seed generator. The default is 42.

    Returns
    -------
    - err_train: array
        [0]: Loss
        [1]: Accuracy

    - err_val: array
        [0]: Loss
        [1]: Accuracy

    - theta: dict (covars x layers)
        Weight matrix of the neural network per layer

    """
    # init randomness
    _reset_random_seeds(seed=seed)

    # get train database
    X_train=dbb['train']['X']
    Y_train=dbb['train']['y']

    # number of classes, covars and observations
    n_class = Y_train.shape[1] # num columns
    n_obs, n_covars = X_train.shape

    # create model
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            units= n_class,
            activation='softmax',
            # initialization of weights
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
            min_delta=0,
            patience=10,
            restore_best_weights=True
        )
    else:
        # train model
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
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
        validation_data=(dbb['validation']['X'], dbb['validation']['y']),
        validation_batch_size=len(dbb['validation']['X'])    #complete gradient
    )

    # get errors (loss and accuracy)
    err_train = model.evaluate(X_train, Y_train, verbose=0)
    err_val = model.evaluate(dbb['validation']['X'], dbb['validation']['y'], verbose=0)
    theta = model.get_weights()
    
    if return_model:
        return(model)
    else:
        return(err_train, err_val, theta)

def eval_LassoTCV(
        dbb, nsplit_cv=5, ntest_cv=1,
        lambda_val=0, metric='accuracy',
        seed=42, verbose=True
        ):
    """
    Evaluate dbb (train and validation) in a simple temporal cross validation

    Parameters
    ----------
    dbb : dict
        Dictionary of dictionaries. First dictionary should have a 'train'
        and 'validation' keys, which are the databases where the model should
        fit and validate (test) the data. Each dictionary should have a 'X' and 'y' keys,
        which are the covariables and the objective variables, respectively.
    nsplit_cv : int, optional
        Number of cross validation splits. The default is 5.
    ntest_cv : int, optional
        Number of cross validations evaluated. The default is 1 (all the database).
    lambda_val : float, optional
         Lasso penalization magnitude. The default is 0.
    metric : TYPE, optional
        The neural network stops training in the next 10 epochs where there
        isn't an observed improvement. The default is 'accuracy'.
    seed : int, optional
         Random seed generator. The default is 42.
    verbose : boolean, optional
        If True then it will print progress in evaluation. The default is True.

    Returns
    -------
    - loss_stats: Dataframe
        Dataframe with sample mean & standard deviation

    - acc_stats: Dataframe
        Dataframe with sample mean, standard deviation & Willson confidence intervals
        for bernoulli distribution

    """

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

    tscv = TimeSeriesSplit(n_splits=nsplit_cv, test_size=ntest_cv, gap=0)
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
        _, err_val, _ = build_model(dbb_val, lambda_val=lambda_val, metric_stop=metric, seed=seed)

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
    std_acc = np.std(array_errors[:, 1], axis=0)
    
    # return
    loss_stats = np.array((mean_loss, std_loss))
    acc_stats = np.array((mean_acc, std_acc))

    return(loss_stats, acc_stats)

def get_lambdaLassoTCV(
        dbb, lambdas=np.power(10, np.linspace(1, -5)), nsplit_cv=5,
        ntest_cv=1,  metric='accuracy', verbose=True, seed=48
        ):
    """
    Get best lambda for lasso penalization in a temporal cross validation. As well,
    get the last lambda within one standard deviation and the last lambda which its
    confidence interval includes the best lambda.

    Parameters
    ----------
    dbb : dict
        Dictionary of dictionaries. First dictionary should have a 'train'
        and 'validation' keys, which are the databases where the model should
        fit and validate (test) the data. Each dictionary should have a 'X' and 'y' keys,
        which are the covariables and the objective variables, respectively.
    lambdas : array, optional
        The search space grid of the best posible lambda.
        The default is np.power(10, np.linspace(1, -5)).
    nsplit_cv : int, optional
        Number of cross validation splits. The default is 5.
    ntest_cv : int, optional
        Number of cross validations evaluated. The default is 1 (all the database).
    lambda_val : float, optional
         Lasso penalization magnitude. The default is 0.
    metric : TYPE, optional
        The neural network stops training in the next 10 epochs where there
        isn't an observed improvement. The default is 'accuracy'.
    seed : int, optional
         Random seed generator. The default is 42.
    verbose : boolean, optional
        If True then it will print progress in evaluation. The default is True.

    Returns
    -------
    Dictionary. The keys are 'stats' and 'lambdas' which includes the dataframes
    of the validation loss and accuracy per lambda and the

    """

    # print loops
    if verbose:
        print("temporal CV of " + str(nsplit_cv) + " folds\n\n")

    # save accuracy and loss
    df_lossstats = pd.DataFrame(columns=("mean_loss", "std_loss"))
    df_accstats = pd.DataFrame(columns=("mean_acc", "std_acc"))

    n_lambdas = len(lambdas)

    # train and get best lambda model
    for i, lambd in enumerate(lambdas):
        # print number of iteration
        if verbose and i % 5 == 0:
            print('\n\ntraining model no.' + str(i+1) + ' of ' + str(n_lambdas))

        # train and test model for each lambda in temporal cross validation
        df_lossstats.loc[i], df_accstats.loc[i] = eval_LassoTCV(
            dbb,
            nsplit_cv=nsplit_cv,
            ntest_cv=ntest_cv,
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

    # get best sparser model lambda at one std
    acc_1std = df_accstats.loc[imax:,"mean_acc"] + df_accstats.loc[imax:,"std_acc"]
    imax_sparse_std = np.where(best_acc <= acc_1std)[0][-1] + imax
    lambda_sparse_std = lambdas[imax_sparse_std]

    dic_out = {
        'stats': {
            'loss': df_lossstats,
            'acc': df_accstats
        },
        'lambdas': {
            'optimal': lambda_opt,
            'sparser_1std': lambda_sparse_std
        }
    }
    return(dic_out)

def get_lassoWeights(
        dbb, lambdas=np.power(10, np.linspace(1, -5)), metric='accuracy',
        verbose=True, seed=48
        ):
    """
    Compute the train and validation accuracy, loss and weights per value of lambda
    for ALL the train set

    Parameters
    ----------
    dbb : dict
        Dictionary of dictionaries. First dictionary should have a 'train'
        and 'validation' keys, which are the databases where the model should
        fit and validate (test) the data. Each dictionary should have a 'X' and 'y' keys,
        which are the covariables and the objective variables, respectively.
    lambdas : array, optional
        The search space grid of the best posible lambda.
        The default is np.power(10, np.linspace(1, -5)).
    metric : TYPE, optional
        The neural network stops training in the next 10 epochs where there
        isn't an observed improvement. The default is 'accuracy'.
    seed : int, optional
         Random seed generator. The default is 42.
    verbose : boolean, optional
        If True then it will print progress in evaluation. The default is True.

    Returns
    -------
    Dictionary. A dataframe of the error stats per lambda. A dictionary of the
    weights per layer per lambda value.

    """

    # save accuracy and loss
    array_errors = np.zeros((len(lambdas), 5))
    dic_weights = {}
    n_lambdas = len(lambdas)

    # train and get best lambda model
    if verbose:
        _printProgressBar(0, n_lambdas)

    for i, lambd in enumerate(lambdas):
        # print number of iteration
        if verbose:
            _printProgressBar(i + 1, n_lambdas)

        # train and test model for each lambda
        err_train, err_val, theta = build_model(dbb, lambda_val=lambd, metric_stop=metric, seed=seed)

        # append values
        # row by: lambda, loss train, loss val, accuracy train, accuracy validation
        array_errors[i] = np.array([lambd, err_train[0], err_val[0], err_train[1], err_val[1]])
        dic_weights['lambda_' + str(i+1)] = {'W': theta[0], 'b': theta[1]}

    # get best lambda
    df_errors = pd.DataFrame(
        array_errors,
        columns=['lambdas', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
    )


    dic_out = {
        'df_lambdas': df_errors,
        'dic_thetas': dic_weights
    }
    return(dic_out)

#%% neural models
def build_model_elastic(hp):
    """
    Build Elastic Net deep models with hyperparameter tunning
    Actually, the hyperparameters for tunning are:
        1. penalization (lambda) per layer
        2. convexity between lasso and ridge (lambda)
        3. number of neurons per layer 
        4. learning rate

    Parameters
    ----------
    hp : hyperparam selector
        

    Returns
    -------
    tunners grid
    """
    # init randomness
    _reset_random_seeds(8)
    n_covars = 21
    n_l = 4
    
    #### hyper params ####
    # lambda penalization
    dic_lambdas = {
        'layer' + str(i+1): hp.Float(
          name='elastic_lambda_l' + str(i+1),
          min_value=1e-4,
          max_value=1e-2
        ) 
        for i in range(n_l)
    }
    
    # alpha (convex combination between lasso & ridge)
    dic_alphas = {
        'layer' + str(i+1): hp.Float(
          name='elastic_alpha_l' + str(i+1),
          min_value=0,
          max_value=1,
          default=0.5
        ) for i in range(n_l)
    }
    
    # number of neurons
    # create a funnel architecture
    dic_units = dict()
    for i in range(n_l - 1):
      if i == 0:
        dic_units['layer1'] = hp.Int(
          name='units_l1',
          min_value=n_covars,
          max_value=n_covars*2,
          step=4,
          default=n_covars # initial value
        )
      else:
        dic_units['layer' + str(i+1)] = hp.Int(
          name='units_l' + str(i+1),
          min_value=dic_units['layer' + str(i)],
          max_value=dic_units['layer' + str(i)]*2,
          step=4,
          default=dic_units['layer' + str(i)] # initial value
        )
    
    # learning rate
    lr_r = hp.Choice(
        name = "lr", 
        values = [1e-2, 1e-3, 1e-4]
      )
    
    #### create model ####
    model = keras.Sequential()
    # layers
    for i in range(n_l - 1):
      ikey = str(i+1)
      model.add(
          keras.layers.Dense(
              units=dic_units['layer' + ikey],
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l1_l2(
                  l1=dic_lambdas['layer' + ikey] * dic_alphas['layer' + ikey],     # Hastie & Tibshiranie formulation of elastic net 
                  l2=dic_lambdas['layer' + ikey] * (1 - dic_alphas['layer' + ikey]) / 2  # Hastie & Tibshiranie formulation of elastic net 
                ),
              dtype=np.float64
          )
      )
    
    # final layer
    model.add(
        keras.layers.Dense(
          units=3, # three final classes
          activation='softmax',
          kernel_initializer=keras.initializers.GlorotNormal(),
          kernel_regularizer=keras.regularizers.l1_l2(
              l1=dic_lambdas['layer' + str(n_l)] * dic_alphas['layer' + str(n_l)],
              l2=dic_lambdas['layer' + str(n_l)] * (1 - dic_alphas['layer' + str(n_l)]) / 2  
            ),
          dtype=np.float64          
        )
    )
    
    # compile model
    model.compile(
          optimizer=keras.optimizers.Nadam(learning_rate=lr_r),
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )
    
    return(model)

def build_model_lasso(hp):
    """
    Lasso deep models with hyperparameter tunning
    Actually, the hyperparameters for tunning are:
        1. penalization (lambda) per layer
        2. number of neurons per layer 
        3. learning rate

    Parameters
    ----------
    hp : hyperparam selector
        

    Returns
    -------
    tunners grid
    """
    # init randomness
    _reset_random_seeds(8)
    n_covars = 21
    n_l = 4
    
    #### hyper params ####
    # lambda penalization
    dic_lambdas = {
        'layer' + str(i+1): hp.Float(
          name='elastic_lambda_l' + str(i+1),
          min_value=1e-4,
          max_value=1e-2
        ) 
        for i in range(n_l)
    }
    
    # number of layers
    # create a funnel architecture
    dic_units = dict()
    for i in range(n_l - 1):
      if i == 0:
        dic_units['layer1'] = hp.Int(
          name='units_l1',
          min_value=n_covars,
          max_value=n_covars*2,
          step=4,
          default=n_covars # initial value
        )
      else:
        dic_units['layer' + str(i+1)] = hp.Int(
          name='units_l' + str(i+1),
          min_value=dic_units['layer' + str(i)],
          max_value=dic_units['layer' + str(i)]*2,
          step=4,
          default=dic_units['layer' + str(i)] # initial value
        )
    
    # learning rate
    lr_r = hp.Choice(
        name = "lr", 
        values = [1e-2, 1e-3, 1e-4]
      )
    
    #### create model ####
    model = keras.Sequential()
    # layers
    for i in range(n_l - 1):
      ikey = str(i+1)
      model.add(
          keras.layers.Dense(
              units=dic_units['layer' + ikey],
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l1(dic_lambdas['layer' + ikey]),
              dtype=np.float64
          )
      )
    
    # final layer
    model.add(
        keras.layers.Dense(
          units=3, # three final classes
          activation='softmax',
          kernel_initializer=keras.initializers.GlorotNormal(),
          kernel_regularizer=keras.regularizers.l1(dic_lambdas['layer' + ikey]),
          dtype=np.float64          
        )
    )
    
    # compile model
    model.compile(
          optimizer=keras.optimizers.Nadam(learning_rate=lr_r),
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )
    
    return(model)

def build_model_ridge(hp):
    """
    Ridge deep models with hyperparameter tunning
    Actually, the hyperparameters for tunning are:
        1. penalization (lambda) per layer
        2. number of neurons per layer 
        3. learning rate

    Parameters
    ----------
    hp : hyperparam selector
        

    Returns
    -------
    tunners grid
    """
    # init randomness
    _reset_random_seeds(8)
    n_covars = 21
    n_l = 4
    
    
    #### hyper params ####
    # lambda penalization
    dic_lambdas = {
        'layer' + str(i+1): hp.Float(
          name='elastic_lambda_l' + str(i+1),
          min_value=1e-4,
          max_value=1e-2
        ) 
        for i in range(n_l)
    }
    
    # number of neurons
    # create a funnel architecture
    dic_units = dict()
    for i in range(n_l - 1):
      if i == 0:
        dic_units['layer1'] = hp.Int(
          name='units_l1',
          min_value=n_covars,
          max_value=n_covars*2,
          step=4,
          default=n_covars # initial value
        )
      else:
        dic_units['layer' + str(i+1)] = hp.Int(
          name='units_l' + str(i+1),
          min_value=dic_units['layer' + str(i)],
          max_value=dic_units['layer' + str(i)]*2,
          step=4,
          default=dic_units['layer' + str(i)] # initial value
        )
    
    # learning rate
    lr_r = hp.Choice(
        name = "lr", 
        values = [1e-2, 1e-3, 1e-4]
      )
    
    #### create model ####
    model = keras.Sequential()
    # layers
    for i in range(n_l - 1):
      ikey = str(i+1)
      model.add(
          keras.layers.Dense(
              units=dic_units['layer' + ikey],
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(dic_lambdas['layer' + ikey]),
              dtype=np.float64
          )
      )
    
    # final layer
    model.add(
        keras.layers.Dense(
          units=3, # three final classes
          activation='softmax',
          kernel_initializer=keras.initializers.GlorotNormal(),
          kernel_regularizer=keras.regularizers.l2(dic_lambdas['layer' + ikey]),
          dtype=np.float64          
        )
    )
    
    # compile model
    model.compile(
          optimizer=keras.optimizers.Nadam(learning_rate=lr_r),
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )
    
    return(model)

def build_model_dropout(hp):
    """
    Build Drop-Out deep models with hyperparameter tunning
    Actually, the hyperparameters for tunning are:
        1. dropout rate per layer
        2. number of neurons per layer 
        3. learning rate

    Parameters
    ----------
    hp : hyperparam selector
        

    Returns
    -------
    tunners grid
    """
    # init randomness
    _reset_random_seeds(8)
    n_covars = 21
    n_l = 4
    
    #### hyper params ####
    # dropout
    dic_dropout_rate = {
        'layer' + str(i+1): hp.Float(
          name='rate_l' + str(i+1),
          min_value=1e-6,
          max_value=0.5,
          sampling='log'
          ) 
        for i in range(n_l - 1)
    }
    
    # number of layers
    # create a funnel architecture
    dic_units = dict()
    for i in range(n_l - 1):
      if i == 0:
        dic_units['layer1'] = hp.Int(
          name='units_l1',
          min_value=n_covars,
          max_value=n_covars*2,
          step=4,
          default=n_covars # initial value
        )
      else:
        dic_units['layer' + str(i+1)] = hp.Int(
          name='units_l' + str(i+1),
          min_value=dic_units['layer' + str(i)],
          max_value=dic_units['layer' + str(i)]*2,
          step=4,
          default=dic_units['layer' + str(i)] # initial value
        )
    
    # learning rate
    lr_r = hp.Choice(
        name = "lr", 
        values = [1e-2, 1e-3, 1e-4]
      )
    
    #### create model ####
    model = keras.Sequential()
    
    # layers
    for i in range(n_l - 1):
      ikey = str(i+1)
      # model layer
      model.add(
          keras.layers.Dense(
              units=dic_units['layer' + ikey],
              activation='relu',
              kernel_initializer='he_normal',
              dtype=np.float64
          )    
      )
      # model dropout
      model.add(
          keras.layers.Dropout(rate=dic_dropout_rate['layer' + ikey])        
      )
    
    
    # final layer
    model.add(
        keras.layers.Dense(
          units=3, # three final classes
          activation='softmax',
          kernel_initializer=keras.initializers.GlorotNormal(),
          dtype=np.float64          
        )
    )
    
    # compile model
    model.compile(
          optimizer=keras.optimizers.Nadam(learning_rate=lr_r),
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )
    
    return(model)

def build_model_bn(hp):
    """
    Batch Normalization _before_ the activation function, as argued by
    F. Chollet and paper authors: S. Ioffe & C. Szegedy
    Actually, the hyperparameters for tunning are:
        1. dropout rate per layer
        2. number of neurons per layer 
        3. learning rate

    Parameters
    ----------
    hp : hyperparam selector
        

    Returns
    -------
    tunners grid
    
    """
    # init randomness
    _reset_random_seeds(8)
    n_covars = 21
    
    #### hyper params ####
    # number of layers
    n_l = hp.Int(
        name='n_l',
        min_value=2, # not a shallow nn
        max_value=10,
        default=5      
        )
    
    # create a funnel architecture
    dic_units = dict()
    for i in range(n_l - 1):
      if i == 0:
        dic_units['layer1'] = hp.Int(
          name='units_l1',
          min_value=n_covars,
          max_value=n_covars*2,
          step=4,
          default=n_covars # initial value
        )
      else:
        dic_units['layer' + str(i+1)] = hp.Int(
          name='units_l' + str(i+1),
          min_value=dic_units['layer' + str(i)],
          max_value=dic_units['layer' + str(i)]*2,
          step=4,
          default=dic_units['layer' + str(i)] # initial value
        )
  
    # learning rate
    lr_r = hp.Choice(
        name = "lr", 
        values = [1e-2, 1e-3, 1e-4]
      )
  
    #### create model ####
    # init
    model = keras.Sequential()
    # batch normalization
    model.add(keras.layers.BatchNormalization())
    
    # layers
    for i in range(n_l - 1):
        ikey = str(i+1)
        
        # compute Z score
        model.add(
            keras.layers.Dense(
                units=dic_units['layer' + ikey],
                kernel_initializer='he_normal',
                use_bias=False, # there isn't need of a bais term because is centered
                dtype=np.float
            )            
        )
        
        # compute A score
        model.add(keras.layers.Activation('elu'))
        
        # batch normalization
        model.add(keras.layers.BatchNormalization())
  
    # final layer
    model.add(
        keras.layers.Dense(
          units=3, # three final classes
          activation='softmax',
          kernel_initializer=keras.initializers.GlorotNormal(),
          dtype=np.float64          
        )
    )
  
    # compile model
    model.compile(
          optimizer=keras.optimizers.Nadam(learning_rate=lr_r),
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )
  
    return(model)

def fit_model(dbb, model, rho, seed=42):
    """
    fit deep learning model with the given database (dbb)

    Parameters
    ----------
    dbb : dictionary
    model : keras.Models
    seed : int, optional

    Returns
    -------
    None
    """
    # init randomness
    _reset_random_seeds(seed=seed)

    # get train database
    X_train=dbb['train']['X']
    Y_train=dbb['train']['y']

    # compile model
    model.compile(
            optimizer=keras.optimizers.Nadam(lr=rho),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # early stopping
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
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[early_stop],
        validation_data=(dbb['validation']['X'], dbb['validation']['y']),
        validation_batch_size=8
    )

    return()

def eval_model(dbb, model, rho=1e-1, nsplit_cv=5, ntest_cv=1, verbose=True, seed=42):
    """
    fit & evaluate model 

    Parameters
    ----------
    dbb : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    nsplit_cv : TYPE, optional
        DESCRIPTION. The default is 5.
    ntest_cv : TYPE, optional
        DESCRIPTION. The default is 1.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    seed : TYPE, optional
        DESCRIPTION. The default is 42.

    Returns
    -------
    None.

    """
    # merge train and validation in one dataframe
    X_list = [v2 for dics in dbb.values() for k2, v2 in dics.items() if k2 == "X"]
    Y_list = [v2 for dics in dbb.values() for k2, v2 in dics.items() if k2 == "y"]

    X_data = pd.concat(X_list)
    Y_data = pd.concat(Y_list)

    # save loss & acc
    array_errors = np.zeros((nsplit_cv, 2))

    # split data in nsplit_cv-folds
    if verbose:
        print("fold: ", end='')

    tscv = TimeSeriesSplit(n_splits=nsplit_cv, test_size=ntest_cv, gap=0)
    i = 0    
    for index_train, index_val in tscv.split(X_data):
        # get a look in the number of iterations
        if verbose:
            print(str(i+1) + ", ", end="")
            
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
        
        # shell model
        model_copy = keras.models.clone_model(model)
        
        # fit model
        fit_model(dbb_val, model_copy, rho, seed)
        
        # evaluate time series cross validation data
        # append metrics
        array_errors[i] = model_copy.evaluate(
            dbb_val['validation']['X'],
            dbb_val['validation']['y']            
        )

        # update index
        i += 1


    # get statistics of the metrics
    # loss
    mean_loss = np.mean(array_errors[:, 0], axis=0)
    std_loss = np.std(array_errors[:, 0], axis=0)

    # acc
    mean_acc = np.mean(array_errors[:, 1], axis=0)
    std_acc = np.std(array_errors[:, 1], axis=0)

    # return
    loss_stats = np.array((mean_loss, std_loss))
    acc_stats = np.array((mean_acc, std_acc))

    return(loss_stats, acc_stats)

#%% plots
def plt_lossaccCV(df_loss, df_acc, lambs, folder_name='main'):
    """
    
    Plot loss and accuracy with their respective dispersion intervals

    Parameters
    ----------
    df_loss : Dataframe
        Dataframe having 'mean_loss' & 'std_loss'
    df_acc : Dataframe
        Dataframe having 'mean_loss', 'std_loss', & 'std_acc'
    lambs : Dictionary
        Contains keys 'optimal' & 'sparser_1std'

    Returns
    -------
    plot

    """
    # plot    
    fig, ax1 = plt.subplots()
    
    # plot loss
    mu = df_loss['mean_loss']
    sigma = df_loss['std_loss']
    ax1.plot(lambdas, mu, color='black')
    ax1.fill_between(lambdas, mu + sigma, mu - sigma, facecolor='darkgrey', alpha=0.5)
    
    ax1.set_xscale('log')
    ax1.set_ylabel('Pérdida', fontsize=20)
    ax1.set_xlabel('$\lambda$', fontsize=20)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    
    ax1.legend()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # plt accuracy
    theta = df_acc["mean_acc"]
    # 1 std intervals (as Hastie & Tibshirani)
    thetaCI_l = df_acc['std_acc'] 
    thetaCI_u = df_acc['std_acc'] 
    thetaCI = np.vstack((thetaCI_l, thetaCI_u))
    
    ax2.errorbar(
        lambdas,
        theta,
        yerr=thetaCI,
        capsize=5,
        elinewidth=1,
        color ='black',
        ecolor='black',
        fmt='o'
    )
    
    ax2.set_xscale('log')
    ax2.set_ylabel('Precisión', fontsize=20)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    
    # merge plots
    fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
    
    # add lambdas
    plt.axvline(x=lambs['optimal'], color='dimgray', linestyle='--', linewidth = 0.9)
    
    # add title 
    plt.title("Pérdida y Precisión por Penalización $\lambda$\n", fontsize=30)
    
    # save plot #
    # generate figure folder
    if not os.path.exists("Figures_Colab"):
      os.makedirs("Figures_Colab")
    
    # generate lasso folder
    if not os.path.exists("Figures_Colab/Lasso_Selection"):
      os.makedirs("Figures_Colab/Lasso_Selection")

    # generate sub-lasso folder
    final_folder = "Figures_Colab/Lasso_Selection/" + folder_name
    if not os.path.exists(final_folder):
      os.makedirs(final_folder)
    
    # save figure
    plt.savefig(final_folder + "TCV_acc_both.png", bbox_inches='tight')    
    
    # show figure #
    plt.show()

    return()

def pltW_objectiveY(dic, lambdas, lambda_model, threshold_factor=1, folder_name='main'):
    """
    Plot of coefficient decayment for each lambda value per objective variable

    Parameters
    ----------
    dic : Dictionary
        Dictionary with the weights of each layer of the NN per lambda value.
    lambdas : array
        Search space grid used for finding the best lambda
    lambda_model : float
        The lambda that is going to be used in the model
    threshold_factor : float, optional
        Minimum absolute value of the weight to be plotted. The default is 1.

    Returns
    -------
    Plot

    """
    # init params
    NAMES_Y = ('visita', 'empate', 'casa') # a, d, h (order)
    
    # get cube of the weights of the output (unique) layer
    W_lambda_list = [
    v for theta in list(dic.get('dic_thetas').values())
        for k,v in theta.items() if k == 'W'
    ]
    W_lambda3D = np.stack(W_lambda_list, axis=2)
    
    # get values of the 3 dimensions
    n_covars, n_y, n_lambdas = W_lambda3D.shape
    
    # plot each covariable by type of objective var
    for i in range(n_y):
        W_yi =  W_lambda3D[:, i, :].transpose()
        
        # plot weight decayment
        fig = plt.figure()
        ax = fig.add_subplot(111)
        texts = []
        # append number of the covariable
        ilambda = np.argmax(lambdas >= lambda_model)
        Wi_std = np.std(W_yi[ilambda])
        
        # threshold value of Wij to be plotted 
        threshold = threshold_factor * Wi_std
      
        for j in range(n_covars):
            # if it is an "important" variable
            if np.abs(W_yi[ilambda, j]) >= threshold:
                ax.plot(lambdas, W_yi[:, j], color = 'black', alpha=0.9)
                texts.append(plt.text(0.7*lambdas[0], W_yi[0, j], s=str(j+1), fontsize=15))
            else:
                ax.plot(lambdas, W_yi[:, j], color = 'gray', alpha=0.5)
  
        adjust_text(
          texts, 
          only_move={'points':'y', 'text':'y'}, 
          autoalign='y', 
          force_points=0
        )
    
        # show lambda of the model
        plt.axvline(x=lambda_model, linestyle='--', color='gray')
          
        # show 0 threshold
        plt.axhline(y=0, linewidth=0.9, color='gray')
          
        # pretty graph
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=15)
        plt.ylabel('$W$', fontsize=20)
        plt.xlabel('$\lambda$', fontsize=20)
        plt.title('Decaimiento de los pesos para y: ' + NAMES_Y[i], fontsize=30)
        
        # reescale plot
        fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
          
        # save plot #
        # generate figure folder
        if not os.path.exists("Figures_Colab"):
          os.makedirs("Figures_Colab")

        # generate lasso folder
        if not os.path.exists("Figures_Colab/Lasso_Selection"):
          os.makedirs("Figures_Colab/Lasso_Selection")
          
        folder_graphs = "Figures_Colab/Lasso_Selection/By_Objective"
        if not os.path.exists(folder_graphs):
          os.makedirs(folder_graphs)
        
        # generate sub-lasso folder
        final_folder = "Figures_Colab/Lasso_Selection/By_Objective/" + folder_name
        if not os.path.exists(final_folder):
          os.makedirs(final_folder)        
    
        # save figure
        filename = final_folder + "/" + NAMES_Y[i] + "_weights_decay.png"
          
        plt.savefig(filename, bbox_inches='tight')
          
        # show plot
        plt.show()
        print("\n\n")


    return()

def pltW_covars(dic, lambdas, lambda_model, vars_names=[], folder_name='main'):
    """
    Plot of coefficient decayment for each lambda value per covariable variable

    Parameters
    ----------
    dic : Dictionary
        Dictionary with the weights of each layer of the NN per lambda value.
    lambdas : array
        Search space grid used for finding the best lambda
    lambda_model : float
        The lambda that is going to be used in the model
    vars_names : list, optional
        Names of the covariable names in the dataset. The default is [].

    Returns
    -------
    Plot

    """
    # format in plots
    LINESTYLES = ['--', ':', '-']
    NAMES_Y = ('visita', 'empate', 'casa')

    # get cube of the weights of the output (unique) layer
    W_lambda_list = [
    v for theta in list(dic.get('dic_thetas').values())
        for k,v in theta.items() if k == 'W'
    ]
    W_lambda3D = np.stack(W_lambda_list, axis=2)
    
    # get values of the 3 dimensions
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
    
        # show 0 threshold
        plt.axhline(y=0, linewidth=0.9, color='gray')
    
        # pretty graph
        ax.set_xscale('log')
        ax.legend()
        # plt.xlim(lambdas[0], 1e-1)
        plt.ylabel('$W_{{%s}}$' % str(j+1))
        plt.xlabel('$\lambda$')
        plt.title('Pesos de la variable: ' + vars_names[j] + ' (var '+ str(j+1) + ')')
    
        # save plot #
        # generate figure folder
        if not os.path.exists("Figures_Colab"):
          os.makedirs("Figures_Colab")
    
        # generate lasso folder
        if not os.path.exists("Figures_Colab/Lasso_Selection"):
          os.makedirs("Figures_Colab/Lasso_Selection")
    
        folder_graphs = "Figures_Colab/Lasso_Selection/By_Covars"
        if not os.path.exists(folder_graphs):
          os.makedirs(folder_graphs)
          
          
        # generate sub-lasso folder
        final_folder = "Figures_Colab/Lasso_Selection/By_Covars/" + folder_name
        if not os.path.exists(final_folder):
          os.makedirs(final_folder)    
    
        # save figure
        filename = final_folder + "/" + vars_names[j] + "_decay.png"
    
        plt.savefig(filename, bbox_inches='tight')
    
        plt.show()
        print("\n\n")


    return()

def plt_bias(dic, lambdas, lambda_model, folder_name='main'):
    # format in plots
    LINESTYLES = ['--', ':', '-']
    NAMES_Y = ('visita', 'empate', 'casa')

    # get cube of the weights of the output (unique) layer
    b_lambda_list = [
    v for theta in list(dic.get('dic_thetas').values())
        for k,v in theta.items() if k == 'b'
    ]
    b_array = np.stack(b_lambda_list, axis=1)
    
    # get values of the 3 dimensions
    n_y, n_lambdas = b_array.shape
    
    # plot 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot each objective variable bais
    for i in range(n_y):
        ax.plot(lambdas, b_array[i], label=NAMES_Y[i], color = 'black', linestyle=LINESTYLES[i])

    # show lambda of the model
    plt.axvline(x=lambda_model, linewidth=0.4, color='gray')

    # show 0 threshold
    plt.axhline(y=0, linewidth=0.9, color='gray')

    # pretty graph
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=20)
    # plt.xlim(lambdas[0], 1e-1)
    plt.ylabel('$b$', fontsize=20)
    plt.xlabel('$\lambda$', fontsize=20)
    plt.title('Magnitud de los Sesgos', fontsize=30)
    
    # reescale plot
    fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])

    # save plot #
    # generate figure folder
    if not os.path.exists("Figures_Colab"):
      os.makedirs("Figures_Colab")

    # generate lasso folder
    if not os.path.exists("Figures_Colab/Lasso_Selection"):
      os.makedirs("Figures_Colab/Lasso_Selection")

    folder_graphs = "Figures_Colab/Lasso_Selection/Bias"
    if not os.path.exists(folder_graphs):
      os.makedirs(folder_graphs)
      
      
    # generate sub-lasso folder
    final_folder = "Figures_Colab/Lasso_Selection/Bias/" + folder_name
    if not os.path.exists(final_folder):
      os.makedirs(final_folder)    

    # save figure
    filename = final_folder + "bias_decay.png"

    plt.savefig(filename, bbox_inches='tight')

    plt.show()        
    
    
    return()

# %% performance
def conf_matrx(y_real, y_hat, labels):
    # labels in order of y_real columns
    
    # argmax per row
    y_max = np.max(y_hat, axis=1).reshape(-1, 1) # column vector
    y_predict = (y_hat == y_max).astype(int) # boolean to int
       
    # confusion matrix
    y_real_label = [labels[ind] for ind in np.argmax(y_real, axis=1)]
    y_predict_label = [labels[ind] for ind in np.argmax(y_predict, axis=1)]
    CONFMATm = sk_metrics.confusion_matrix(y_real_label, y_predict_label, labels=labels)
    
    # adorn
    total_real = np.sum(y_real, axis=0) # sum columns
    total_predict = np.sum(y_predict, axis=0)
    
    # array 2 dataframe
    df_cm = (
        pd.DataFrame(CONFMATm, index=labels, columns=labels)
        .assign(total_model = total_real)
        .append(
            pd.DataFrame(dict(zip(labels, total_predict)), index=['total_predict']), 
            ignore_index=False            
            )        
        )
        
    return(df_cm)



# =============================================================================
#%% main 
# =============================================================================

yr = np.array([[0, 0, 1],
       [0, 0, 1],
       [1, 0, 0],
])

yh = np.array([[0.53873701, 0.23627209, 0.22499089],
 [0.03501112, 0.10946997, 0.85551891],
 [0.34705663, 0.31666682, 0.33627655]])

labels = list('adh')

cm = conf_matrx(yr, yh, labels)






# %%1. get & tidy dataframe
# init values
SEED_VALUE = 8

# read data
df_r = pd.read_csv("Data/Main_DBB/model_myscale.csv")

# generate trains / validation / test database (three tables)
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
    'validation': 'season >= 20 & matchweek < 0.45',    # matchweek 18 (standarized)
    'test': 'season >= 20 & matchweek >= 0.45'          # normalized(18; max=38, min=2) = (18-2)/(38-2) = 0.444...
    }

# main databaes
dbb = tidy_bdd(df_r, queries=queries_model, objective_var='result', var=vars_model)

# look raw data data
dbb['train']['X']


# =============================================================================
# variable selection
# =============================================================================
#%% 1. lasso models
# lambda's grid
lambdas = np.power(10, np.linspace(-3.5, 1, num=3))

# get number of cross validation sets
n_games_per_matchweek = 10 # there are 10 game per matchweek
n_cv = dbb['validation']['y'].shape[0] // n_games_per_matchweek

# get best lambda's
dict_best_lambdas = get_lambdaLassoTCV(
    dbb=dbb,
    lambdas=lambdas,
    nsplit_cv=n_cv,                 # number of matchweeks to evaluate
    ntest_cv=n_games_per_matchweek, # number of observations in test set in the tcv
    metric='loss',                  # EarlyStopping metric
    seed=SEED_VALUE
)

#%% 2. Plot acc & loss
# plot acc & loss
#plot
df_loss = dict_best_lambdas['stats']['loss']
df_acc = dict_best_lambdas['stats']['acc']
lambs = dict_best_lambdas['lambdas']

fig, ax1 = plt.subplots()

# plot loss
mu = df_loss['mean_loss']
sigma = df_loss['std_loss']
ax1.plot(lambdas, mu, color='black')
ax1.fill_between(lambdas, mu + sigma, mu - sigma, facecolor='darkgrey', alpha=0.5)

ax1.set_xscale('log')
ax1.set_ylabel('Pérdida')
ax1.set_xlabel('$\lambda$')
ax1.tick_params(axis='y')
ax1.tick_params(axis='x')
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# plt accuracy
theta = df_acc["mean_acc"]
# 1 std intervals (as Hastie & Tibshirani)
thetaCI_l = df_acc['std_acc'] 
thetaCI_u = df_acc['std_acc'] 
thetaCI = np.vstack((thetaCI_l, thetaCI_u))

ax2.errorbar(
    lambdas,
    theta,
    yerr=thetaCI,
    capsize=5,
    elinewidth=1,
    color ='black',
    ecolor='gray',
    fmt='o'
)

ax2.set_xscale('log')
ax2.set_ylabel('Precisión')
ax2.tick_params(axis='y')
ax2.tick_params(axis='x')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

# merge plots
# fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])

# add lambdas
plt.axvline(x=lambs['optimal'], color='dimgray', linestyle='--', linewidth = 0.9)
plt.axvline(x=lambs['sparser_1std'], color='dimgray', linestyle='--', linewidth = 0.9)

# add title 
plt.title("Pérdida y Precisión por Penalización $\lambda$\n", fontsize=20)

# show figure #
plt.show()

#%% 3. weights decayment

# %% 4. final weights
names_y = ['visita', 'empate', 'local']

# train model
shallow_nn = build_model(
    dbb=dbb,
    lambda_val=0,
    metric_stop='loss',
    seed=SEED_VALUE,
    return_model=True
)

# look model accuracy

# look weights 
dict_shallow_weights = shallow_nn.get_weights()
df_shallow_weights = (
    pd.concat((
        pd.DataFrame(dict_shallow_weights[0], index=names_y, columns=vars_model),
        pd.DataFrame(dict_shallow_weights[1], index='intercept', columns=vars_model)       
    ))        
)


#%% 
# =============================================================================
# %% model selection
# =============================================================================











