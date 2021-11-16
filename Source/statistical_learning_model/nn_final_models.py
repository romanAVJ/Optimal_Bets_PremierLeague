# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:08:07 2021

@author: Ryo
"""
# get libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
import os 
import random

from matplotlib import pyplot as plt
from adjustText import adjust_text

from sklearn.model_selection import TimeSeriesSplit
import Source.statistical_learning_model.nn_utils as nn_utils_ravj

# =============================================================================
# UTILS
# =============================================================================


# =============================================================================
# Builiding models
# =============================================================================
# model 1:  Elastic Net
def build_model_elastic(hp):
  # init randomness
  nn_utils_ravj._reset_random_seeds(8)
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


# model 2:  Lasso
def build_model_lasso(hp):
  # init randomness
  nn_utils_ravj._reset_random_seeds(8)
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


# model 3:  Ridge
def build_model_ridge(hp):
  # init randomness
  nn_utils_ravj._reset_random_seeds(8)
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


# model 4:  DropOut
def build_model_dropout(hp):
  # init randomness
  nn_utils_ravj._reset_random_seeds(8)
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


# model 5:  Batch Normalization
def build_model_bn(hp):
    """
    Batch Normalization _before_ the activation function, as argued by
    F. Chollet and paper authors: S. Ioffe & C. Szegedy
    
    Parameters
    ----------
    hp : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.
    
    """
    # init randomness
    nn_utils_ravj._reset_random_seeds(8)
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


# =============================================================================
# build models
# =============================================================================

def fit_model(dbb, model, seed=42):
    # init randomness
    nn_utils_ravj._reset_random_seeds(seed=seed)

    # get train database
    X_train=dbb.get('train')['X']
    Y_train=dbb.get('train')['y']

    # compile model
    model.compile(
            optimizer=keras.optimizers.Nadam(lr=1e-2),
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
        validation_data=(dbb.get('validation')['X'], dbb.get('validation')['y']),
        validation_batch_size=len(dbb.get('validation')['X'])    #complete gradient
    )

    return()

def eval_model(dbb, model, nsplit_cv=5, ntest_cv=1, verbose=True, seed=42):
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
        fit_model(dbb_val, model_copy, seed)
        
        # evaluate time series cross validation data
        # append metrics
        array_errors[i] = model_copy.evaluate(
            dbb_val.get('validation').get('X'),
            dbb_val.get('validation').get('y')            
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


















