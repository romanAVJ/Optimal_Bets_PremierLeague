# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:28:02 2021

@author: Ryo

Simultanuous independent Kelly Criterion Optimization restricted
"""
import pandas as pd
import numpy as np
import os
import warnings
import re

import scipy.optimize as optimize
from scipy.sparse import csc_matrix
from itertools import product, cycle, islice
from pathlib import Path

# =============================================================================
#%% functions
# =============================================================================
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
    
    return(df_work)

def df2array(df, vars_subset=[], var_filter='x', value_filter=None):
    # working dataframe
    df_work = df.copy()
    
    # filter 
    df_subset = df_work[df[var_filter] == value_filter][vars_subset]
    
    # to array
    array_subset = df_subset.to_numpy().flatten(order='C')  # append rows
    
    return(array_subset)

def ids_array(df, times=3, var_id='id', var_filter='x', value_filter=None):
    # working dataframe
    df_work = df.copy()
    
    # filter 
    df_subset = df_work[df[var_filter] == value_filter][var_id]
    
    # 1dim
    array_id_raw = np.repeat(df_subset.to_numpy().squeeze(), repeats=times) # repeat
    array_id = np.sort(array_id_raw, axis=None) # sort squeezed array
    
    return(array_id)

def get_posibility_matrix(m, n):
    # generate all inedx posible combinations
    posibilities = list(product(range(m), repeat=n)) 
    matrix_index_indicator = np.array(posibilities, dtype=np.int32)
    
    # get indexes for sparse matrix
        # add m for each index to section the flatten matrix index
    narray_indexer = np.cumsum(np.repeat(m, n)) - m
    matrix_index = matrix_index_indicator + narray_indexer
    
    # generate indexes and indptr for sparse matrix
    indices_event = matrix_index.flatten(order='C') # append rows
    
    aux_indptr = np.concatenate((np.array([0]), np.repeat(n, len(posibilities))))
    indptr = np.cumsum(aux_indptr) # sectionate the indices

    # generate sparse matrix
    array1d_sparse = np.ones(indptr[-1], dtype=int)
    A_sparse = csc_matrix((array1d_sparse, indices_event, indptr))
    
    return(A_sparse.toarray())

def get_posibility_matrix_one(n):
    # m = 2 by default
    m = 2
    
    #  generate all posible binomial combinations
    posibilities = list(product(range(m), repeat=n)) 
    A = np.transpose(np.array(posibilities, dtype=np.int32))
    return(A)
    
    
def get_probas_independent_simultanuous(M_outcomes, probas, tol):
    # get tilde probability matrix
    Pm = np.matmul(np.diag(probas), M_outcomes)
    
    # get proba matrix
    Pm[Pm == 0] = 1
    
    # get each posible combination probability
    proba_posibilities = np.prod(Pm, axis=0) # multiply rows 
    
    # look that the sum is closely to 1
    sum_probas = np.sum(proba_posibilities)
    if np.abs(sum_probas - 1) >  tol:
        warnings.warn(
            f"""The sum of proabilities was {sum_probas} 
            and is out of +/-{tol} tolaration"""
            )
    
    return(proba_posibilities)

def set_probas(omega, probas):
    # compare if happen or not an outcome and return probability
    p = np.where(omega > 0, probas, 1 - probas) 
    
    return(p)


def get_probas_independent(M_outcomes, probas, tol=1e-6):
    # get probabilities for each element in omega
    Pm = np.apply_along_axis(
        func1d=set_probas,
        axis=0,
        arr=M_outcomes,
        probas=probas # *args
    )
    
    # get probabilities for each posible event
    proba_posibilities = np.prod(Pm, axis=0) # by rows
    
    # look that the sum is closely to 1
    sum_probas = np.sum(proba_posibilities)
    if np.abs(sum_probas - 1) >  tol:
        warnings.warn(
            f"""The sum of proabilities was {sum_probas} 
            and is out of +/-{tol} tolaration"""
            )
    
    return(proba_posibilities)
    
def neg_log_growth_simultanuous(bets, Wt, probs, tol=1e-6):
    # get expected log return
    Gl = np.dot(probs, np.log(1 + np.matmul(Wt, bets) - np.sum(bets)))
    
    return(- Gl)

def neg_log_growth_simultanuous_grad(bets, Wt, probs):
    xwin_return = Wt - 1
    xreturn = 1 + np.matmul(Wt, bets) - np.sum(bets)
    xwin_return_gradient = np.divide(xwin_return, xreturn[:, np.newaxis]) # divide each row by xreturn
    
    Gl_grad = np.dot(np.transpose(xwin_return_gradient), probs)

    return(-Gl_grad)

def kelly_stake(games, tol=1e-6, ftol=1e-12):
    # parameters
        # get m- outcomes and r-games
    m = games['num_events']
    r = games['num_games']
    
        # get posibility matrix (all posible combinations of independent games)
    OMEGAm = get_posibility_matrix(m=m, n=r)
    
        # get transposed wins matrix
    Wt = np.matmul(np.transpose(OMEGAm), np.diag(games['odds']))
    
        # get probability matrix
    proba_posibilities = get_probas_independent_simultanuous(OMEGAm, games['probas'], tol=tol)
    
    # optimization
        # initial feasable point
    bet0 = np.zeros(m * r)
    
        # constraints
    ineq_cons_budget = {
        'type': 'ineq',
        'fun': lambda x: 1 - np.sum(x),
        'jac': lambda x: - np.ones(x.shape)
    }
    
        # bounds 
    bounds_gl = optimize.Bounds(np.zeros(m*r), np.ones(m*r))
    
        # quadratic programming optimization 
    opt = optimize.minimize(
        fun=neg_log_growth_simultanuous, # maximize the negative log growth
        x0=bet0,                           # initial point
        args=(Wt, proba_posibilities),   # arguments for the function and the gradient
        method='SLSQP',                  
        jac=neg_log_growth_simultanuous_grad, # gradient
        bounds=bounds_gl,
        constraints=ineq_cons_budget,
        options={'ftol': ftol, 'disp': False}    
    )    
    
    # tidy results
    dict_result = {
        'log_growth': - opt['fun'],
        'bets': opt['x']        
    }
    
    if not opt['success']:
        warnings.warn("Optimization wasn't succesful")        
    
    return(dict_result)

def kelly_stake_one(games, tol=1e-6, ftol=1e-12):
    # parameters
        # get m- outcomes and r-games
    m = games['num_events']
    r = games['num_games']
    
        # get posibility matrix (all posible combinations of independent games)
    OMEGAm = get_posibility_matrix_one(n=r)
    
        # get transposed wins matrix
    Wt = np.matmul(np.transpose(OMEGAm), np.diag(games['odds']))
    
        # get probability matrix
    proba_posibilities = get_probas_independent(OMEGAm, games['probas'], tol=tol)
    
    # optimization
        # initial feasable point
    bet0 = np.zeros(r)
    
        # constraints
    ineq_cons_budget = {
        'type': 'ineq',
        'fun': lambda x: 1 - np.sum(x),
        'jac': lambda x: - np.ones(x.shape)
    }
    
        # bounds 
    bounds_gl = optimize.Bounds(np.zeros(r), np.ones(r))
    
        # quadratic programming optimization 
    opt = optimize.minimize(
        fun=neg_log_growth_simultanuous, # maximize the negative log growth
        x0=bet0,                           # initial point
        args=(Wt, proba_posibilities),   # arguments for the function and the gradient
        method='SLSQP',                  
        jac=neg_log_growth_simultanuous_grad, # gradient
        bounds=bounds_gl,
        constraints=ineq_cons_budget,
        options={'ftol': ftol, 'disp': False}    
    )    
    
    # tidy results
    dict_result = {
        'log_growth': - opt['fun'],
        'bets': opt['x']        
    }
    
    if not opt['success']:
        warnings.warn("Optimization wasn't succesful")       
    
    return(dict_result)




# =============================================================================
# init
# =============================================================================
PRODUCT_FOLDER = "Results/bets/"

# write to csv/feather
Path(PRODUCT_FOLDER).mkdir(parents=True, exist_ok=True)



# =============================================================================
#%% main
# =============================================================================
# read dataframe
df_prediction = pd.read_csv("Results\\statistical_estimates\\ridge_prediction.csv")
df_odds = pd.read_csv("Data\\Main_DBB\\stake_odds.csv")

# join frames
df_work = (
    # left join
    pd.merge(
        left=df_prediction,
        right=df_odds,
        how='inner',
        on=['season', 'matchweek', 'hometeam', 'awayteam']
    )
    # get dummies fo result
    .pipe(
        func=get_dummies_results,
        var='result',
        prefix='result'        
    )
    # generate key_index
    .assign(
        id_game = range(1, len(df_odds) + 1)
    )
    # select and arrange columns
    [[
      'id_game', 'season', 'matchweek', 'date','hometeam', 'awayteam', 
      'result_H', 'result_D', 'result_A',       
      'hatproba_home', 'hatproba_draw', 'hatproba_away', 
       'maxo_h', 'maxo_d', 'maxo_a', 'market_tracktake', 
       'diff_home', 'diff_draw', 'diff_away'
       ]]       
)

# subset test set
matchweek_test = tuple(range(18, 39))
df_s = df_work[(df_work['season'] == 20) & (df_work['matchweek'].isin(matchweek_test))].copy()
 
# get subsets 
cols_probas = ['hatproba_home', 'hatproba_draw', 'hatproba_away']
cols_odds = ['maxo_h', 'maxo_d', 'maxo_a']
cols_results = ['result_H', 'result_D', 'result_A']

# get main dictionary
dict_games = dict()
for mtchwk in matchweek_test:
    dict_games['matchweek_' + str(mtchwk)] = {
        'id_game': ids_array(df_s, 3, 'id_game', 'matchweek', mtchwk),
        'matchweek': mtchwk,
        'odds': df2array(df_s, cols_odds, 'matchweek', mtchwk), 
        'probas': df2array(df_s, cols_probas, 'matchweek', mtchwk),
        'results': df2array(df_s, cols_results, 'matchweek', mtchwk),
        'num_events': 3, # 3 posible outcomes
        }
    dict_games['matchweek_' + str(mtchwk)]['num_games'] =\
        np.sum(dict_games['matchweek_' + str(mtchwk)]['results'])
    dict_games['matchweek_' + str(mtchwk)]['results_categoric'] =\
        tuple(islice(cycle(['home', 'draw', 'away']), 10*3)) # 10 games X 3 ouctomes|

# %% get test dataframe in long format
df_s_long = pd.DataFrame(
    columns=[
        'id_game', 'matchweek', 'results', 'results_categoric', 
        'odds', 'probas'       
     ]    
)
for mtchwk, games in dict_games.items():
    # append
    df_aux = pd.DataFrame(games)
    df_s_long = pd.concat((df_s_long, df_aux))

df_s_long[['id_game', 'matchweek', 'results']] =\
    df_s_long[['id_game', 'matchweek', 'results']].apply(pd.to_numeric)


# %% get event per game with highest expected gain
# get best event by game
df_s_highest = df_s_long.assign(
        rho = lambda x: x['odds'] * x['probas'] - 1    
    ).sort_values( 
        by=['id_game', 'rho'], ascending=[True, False]
    ).groupby(
        'id_game'
    ).first(
        
    ).reset_index(
    
    )

# subset to dictionary 
# get subsets 
cols_probas = ['hatproba_home', 'hatproba_draw', 'hatproba_away']
cols_odds = ['maxo_h', 'maxo_d', 'maxo_a']
cols_results = ['result_H', 'result_D', 'result_A']

# get main dictionary
dict_games_one = dict()
for mtchwk in matchweek_test:
    # subset dataframe by matchweek
    df_aux = df_s_highest[df_s_highest['matchweek'] == mtchwk]
    
    # generate dictionary
    aux_key = 'matchweek_' + str(mtchwk)    
    dict_games_one[aux_key] = {
        'probas': df_aux['probas'].to_numpy(),
        'odds': df_aux['odds'].to_numpy(), 
        'results': df_aux['results'].to_numpy(),
        'results_categoric': df_aux['results_categoric'].to_numpy(),
        'id_game': df_aux['id_game'].to_numpy(),
        'num_events': 3 # 3 posible outcomes
        }
    dict_games_one[aux_key]['num_games'] = len(dict_games_one[aux_key]['probas'])


# %% generate optimizations
dict_stakes = dict()
for mtchwk, games in dict_games_one.items():
    print(f"Optimization for {mtchwk}")
    dict_stakes[mtchwk] = kelly_stake_one(games)
    


# %% generate dataframe with all the stakes
df_bets = pd.DataFrame(
    # columns=[
    #     'id_game', 'matchweek', 'results_categoric', 
    #     'bets', 'log_growth'         
    #   ]
)

for mtchwk, games in dict_games_one.items():
    # generate data
    games.update(dict_stakes[mtchwk])
    [games['matchweek']] = re.findall('\d+', mtchwk) # get matchweek number and unpack from list
    
    # append
    df_aux = pd.DataFrame(games)
    df_bets = pd.concat((df_bets, df_aux))

df_bets = df_bets[['id_game', 'matchweek', 'results_categoric', 'bets', 'log_growth']]
df_bets['matchweek'] = df_bets['matchweek'].apply(pd.to_numeric)

# %% merge bets and games dataframes
# join frames
df_main = (
    # left join
    pd.merge(
        left=df_s_long,
        right=df_bets,
        how='left',
        on=['id_game', 'matchweek', 'results_categoric']    
    )
    # select and order columns
    [[ 
      'id_game', 'matchweek', 
      # 'date',       'hometeam', 'awayteam', 'market_tracktake', 
      'results', 'results_categoric', 
      'odds', 'probas', 'bets', 'log_growth'      
      ]]
)

#%% write csv
# file_csv = PRODUCT_FOLDER + "ridge_one_kelly.csv"
# df_main.to_csv(file_csv, index=False)
















































