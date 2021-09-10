# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:06:40 2021

@author: Roman Alberto Velez Jimenez

Create Neural Network model to get probabilities and predictions for the
EPL
"""
# %% modules
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from matplotlib import pyplot as plt
from tensorflow import keras
import os 
import random

# =============================================================================
# %% FUNCTIONS
# =============================================================================
# get reproducible results in keras
# look: https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
def reset_random_seeds(seed, do_print=False):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
   if(do_print):
       print("\n Reset random state with seed: " + str(seed))
       
   return()

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


def tidydf_model1(df, queries, objective_var, var):
    # working data frame
    df_work = df.copy()
    
    # get dummies
    df_work, names_y = get_dummies_results(df_work, var=objective_var, prefix=objective_var)
    
    # mutate matchweek to normalize it (lost generalization in functions)
    df_work['matchweek'] = df_work.groupby('season')['matchweek'].transform(lambda x: (x - min(x))/(max(x) - min(x)))
    
    
    # get train, validation and test
    df_train = df_work.query(queries['q_train'])
    df_val = df_work.query(queries['q_validation'])
    df_test = df_work.query(queries['q_test'])
    
    # split by objective and covariables
    database = {
        'train': {
            'y': df_train[names_y],
            'X': df_train[var]
        },
        'validation': {
            'y': df_val[names_y],
            'X': df_val[var]
        },
        'test': {
            'y': df_test[names_y],
            'X': df_test[var]
        }        
    }    
    
    return(database)
    
# =============================================================================
# %% MAIN
# =============================================================================
# set working dir (bad practice but util)
os.chdir('C:\\Users\\Ryo\\Documents\\Estudios\\ITAM\\Tesis\\sportsAnalytics\\tesis\\Repository')

# read data
df_r = pd.read_csv('Data\\Main_DBB\\stat_model_variables.csv')

# seed
seed_value = 8

# =============================================================================
# %% PLAYGROUND
# =============================================================================
# %% model
vars_model1 = [
    'matchweek', 'position_table_home', 'total_pts_home',
    'npxGD_ma_home', 'npxGD_var_home', 'big_six_home',
    'promoted_team_home', 'position_table_away', 'total_pts_away',
    'npxGD_ma_away', 'npxGD_var_away', 'big_six_away',
    'promoted_team_away', 'ova_home', 'att_home', 'mid_home',
    'def_home', 'transfer_budget_home', 'ip_home', 'saa_home',
    'ova_away', 'att_away', 'mid_away', 'def_away',
    'transfer_budget_away', 'ip_away', 'saa_away', 'psch', 'pscd',
    'psca'
   ]

queries_model1 = {
    'q_train':  'season < 20',
    'q_validation': 'season >= 20 & matchweek < 0.45', # matchweek 18 (standarized)
    'q_test': 'season >= 20 & matchweek >= 0.45'         # normalized(18; max=38, min=2) = (18-2)/(38-2)
    }

dbb1 = tidydf_model1(df_r, queries=queries_model1, objective_var='result', var=vars_model1)

# train dataset
X_train=dbb1.get('train')['X']
Y_train=dbb1.get('train')['y']
n_train = len(X_train)

### neural network model 1 ###
# initialize #
reset_random_seeds(seed_value, do_print=False)
model1 = keras.Sequential(
    layers=[
        keras.layers.Dense(10, activation='relu', dtype= np.float64, use_bias=True, name='input_layer'),
        keras.layers.Dense(10, activation='relu', dtype= np.float64, use_bias=True, name='hidden_layer1'),
        keras.layers.Dense(10, activation='relu', dtype= np.float64, use_bias=True, name='hidden_layer2'),
        keras.layers.Dense(10, activation='relu', dtype= np.float64, use_bias=True, name='hidden_layer3'),
        keras.layers.Dense(3, activation='softmax', dtype= np.float64, use_bias=True, name='output_layer') # 3 classes to predict
    ],
    name='model1_dnn'
)

# compile #
lr_model1 = 0.001

model1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_model1), # should use all batch
    loss='categorical_crossentropy', # y is one-hot represented
    metrics=['accuracy']    
)

# train #
history1 = model1.fit(
    x=X_train,
    y=Y_train,
    epochs=5000,
    batch_size=n_train, # complete gradient (Goodfellow suggest this)
    verbose=0,
)

# evaluate model #
# look model
model1.summary()   

# plot loss
loss_model1 = history1.history['loss']
epochs = range(len(loss_model1))
plt.plot(epochs, loss_model1, 'b', label='train_loss')
plt.show()

# look weights and biases sizes
# get 
weights_model_r = model1.get_weights()
weights_model = {}
for i in range(0, len(weights_model_r), 2):
    W, b = weights_model_r[i], weights_model_r[i+1]
    weights_model['layer_' + str(i // 2)] = {'W': W, 'b': b}
    
# matrix norm (supreme norm to see big weights)
table_weights_model1 = pd.DataFrame(
    {
        'layer': [re.findall('\d+', s).pop() for s in list(weights_model.keys())],
        'W_norm': [
            np.linalg.norm(v, ord=np.inf)                   # supreme norm for W
                for Ws in list(weights_model.values())      # get W and b for each layer
                    for k,v in Ws.items() if k == 'W'       # get W
         ],
        'b_norm': [
            np.linalg.norm(v, ord=np.inf)                   # supreme norm for b
                for Ws in list(weights_model.values())      # get W and b for each layer
                    for k,v in Ws.items() if k == 'b'       # get b
         ]
    }    
)
print(table_weights_model1)


# test #
# get test error
model1.evaluate(
    x=dbb1.get('test')['X'],
    y=dbb1.get('test')['y'],
    batch_size=len(dbb1.get('test')['X'])
)


# %% model 2 (multinomial distribution)
# initialize #
reset_random_seeds(seed_value, do_print=False)
model2 = keras.Sequential(
    layers=[
        keras.layers.Dense(3, activation='softmax', dtype= np.float64, use_bias=True, name='output_layer') # 3 classes to predict
    ],
    name='model2_multinomial'
)

# compile #
lr_model = 0.001

model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_model), # should use all batch
    loss='categorical_crossentropy', # y is one-hot represented
    metrics=['accuracy']    
)

# train #
history1 = model2.fit(
    x=X_train,
    y=Y_train,
    epochs=5000,
    batch_size=n_train, # complete gradient (Goodfellow suggest this)
    verbose=0,
)

# evaluate model #
# look model
model2.summary()   

# plot loss
loss_model2 = history1.history['loss']
epochs = range(len(loss_model2))
plt.plot(epochs, loss_model2, 'b', label='train_loss')
plt.show()

# look weights and biases sizes
# get 
weights_model_r = model2.get_weights()
weights_model = {}
for i in range(0, len(weights_model_r), 2):
    W, b = weights_model_r[i], weights_model_r[i+1]
    weights_model['layer_' + str(i // 2)] = {'W': W, 'b': b}
    
# matrix norm (supreme norm to see big weights)
table_weights_model2 = pd.DataFrame(
    {
        'layer': [re.findall('\d+', s).pop() for s in list(weights_model.keys())],
        'W_norm': [
            np.linalg.norm(v, ord=np.inf)                   # supreme norm for W
                for Ws in list(weights_model.values())      # get W and b for each layer
                    for k,v in Ws.items() if k == 'W'       # get W
         ],
        'b_norm': [
            np.linalg.norm(v, ord=np.inf)                   # supreme norm for b
                for Ws in list(weights_model.values())      # get W and b for each layer
                    for k,v in Ws.items() if k == 'b'       # get b
         ]
    }    
)
print(table_weights_model2)


# test #
# get test error
model2.evaluate(
    x=dbb1.get('test')['X'],
    y=dbb1.get('test')['y'],
    batch_size=len(dbb1.get('test')['X'])
)

























