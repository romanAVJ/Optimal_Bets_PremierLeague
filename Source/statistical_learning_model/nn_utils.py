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
import os 
import random

from matplotlib import pyplot as plt
from adjustText import adjust_text

from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit


#### utils
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

#### statistics
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

#### neural networks
def build_model(dbb, lambda_val=0, metric_stop='accuracy', seed=42):
    """
    Build keras shallow lasso neural network. Train and validate data in the
    database given (dbb).
    
    The kernel initializer is GlorotNormal
    The validation threshold is 0.01 in loss with 10 cycles of patience. It restores
    the best weights   

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
    _reset_random_seeds(seed=seed, do_print=False)
    
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

# CV models
def eval_LassoTCV(
        dbb, nsplit_cv=5, ntest_cv=1, 
        alpha=0.05, lambda_val=0, metric='accuracy', 
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
    alpha : float (0,1), optional
        (1-aplha)% confidence interval. The default is 5%
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
    ci_acc_low, ci_acc_up = blliCI_willson(mean_acc, alpha=alpha, n=nsplit_cv)   
    
    # return    
    loss_stats = np.array((mean_loss, std_loss))
    acc_stats = np.array((mean_acc, std_acc, ci_acc_low, ci_acc_up))    
    
    return(loss_stats, acc_stats)

def get_lambdaLassoTCV(
        dbb, lambdas=np.power(10, np.linspace(1, -5)), nsplit_cv=5, 
        ntest_cv=1, alpha=0.05,  metric='accuracy', verbose=True, seed=48
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
    alpha : float (0,1), optional
        (1-aplha)% confidence interval. The default is 5%
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
    df_accstats = pd.DataFrame(columns=("mean_acc", "std_acc", "ci_lower", "ci_upper"))
    
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
            alpha=alpha, 
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
    
    # get best sparser model lambda in same CI 
    imax_sparse_ci = np.where(best_acc <= df_accstats.loc[imax:,"ci_upper"])[0][-1] + imax
    lambda_sparse_ci = lambdas[imax_sparse_ci]
    
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
            'sparser_ci': lambda_sparse_ci,
            'sparser_1std': lambda_sparse_std
        }
    }
    return(dic_out)

# Train/Validation/Test
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

# plots
def plt_lossaccCV(df_loss, df_acc, lambs):
    """
    Plot loss and accuracy with their respective dispersion intervals

    Parameters
    ----------
    df_loss : Dataframe
        Dataframe having 'mean_loss' & 'std_loss'         
    df_acc : Dataframe
        Dataframe having 'mean_loss', 'std_loss', 'ci_lower', 'ci_upper' & 'std_acc'
    lambs : Dictionary
        Contains keys 'optimal', 'sparser_ci' & 'sparser_1std'

    Returns
    -------
    plot

    """
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
    thetaCI_l = df_acc['ci_lower'] # SHOULD CHANGE TO 1STD
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
    plt.axvline(x=lambs['sparser_ci'], color='lightgray', linestyle='--') # should change!!
    
    plt.show()
    
    return()

def pltW_objectiveY(dic, lambdas, lambda_model, threshold_factor=1):
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
        for j in range(n_covars):
            # if it is an "important" variable
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

