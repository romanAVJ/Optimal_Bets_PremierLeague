# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:45:32 2021

@author: Ryo

Simultanuous independent Markowitz Optimization
"""
import pandas as pd
import numpy as np
import os
import warnings
import re

import scipy.optimize as optimize
import pypfopt

from scipy import linalg
from pathlib import Path
from itertools import cycle, islice

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

def portfolio_returns(bets, returns):   
    # estimate mean
    portfolio_return = np.dot(bets, returns)
    
    return(portfolio_return)

def portfolio_volatility(bets, sigma):     
    # get volatitilty
    vol = np.dot(bets, np.matmul(sigma, bets)) # x'S x

    return(vol)

def portfolio_volatility_jac(bets, sigma):
    gradient = 2 * np.matmul(sigma, bets)
    return(gradient)

def sharpe_ratio(bets, returns, sigma):
    sharpe = portfolio_returns(bets, returns) / np.sqrt(portfolio_volatility(bets, sigma))
    return(sharpe)

# bets mean and covariance
def bets_mean(probas, odds):
    rho = odds * probas - 1 # decimal odds    
    return(rho)
    
def exclusive_bets_covariance(odds, probas):
    # get var-covar matrix of a multinomial distribution
        # diag(p) - p p.T
    sigma_multinomial = np.diag(probas) - np.matmul(probas.reshape((-1, 1)), probas.reshape((1, -1)))
    
    # Var(DX) = D Var(X) D.T
    diag_odds = np.diag(odds)
    sigma_exclusive_bets = np.matmul(np.matmul(diag_odds, sigma_multinomial), diag_odds)
    
    return(sigma_exclusive_bets)

def simultanuous_bets_covariance(*sigmas):
    sigma = linalg.block_diag(*sigmas)
    return(sigma)

def covariance_games(games):
    # split games
        # get index to split array
    games_index_split = np.cumsum(pd.Series(games['id_game']).value_counts().to_numpy())[:-1]    
    
        # get list of odds & probas per game
    games_odds = np.array_split(games['odds'], games_index_split)
    games_probas = np.array_split(games['probas'], games_index_split)    
    
    # get var covar per game
        # variance covariance per game
    list_sigmas = [exclusive_bets_covariance(o, p) for o, p in zip(games_odds, games_probas)]  
        # variance covariance for all games
    VAR_COVARm = simultanuous_bets_covariance(*list_sigmas)
    
    return(VAR_COVARm)

def restrictions_markowitz(mu):
    m_all = len(mu)
    # note: (y, kappa) <-> (x[:-1], x[-1])
    
    # constraints
    # sum(y) = kappa    
    eq_cons_kappa = {
        'type': 'eq',
        'fun': lambda y: np.sum(y[:-1]) - y[-1],
        'jac': lambda y: np.concatenate((np.ones(m_all), np.array([-1])))
    }
    # y'mu = 1
    eq_cons_budget = {
        'type': 'eq',
        'fun': lambda y: np.dot(y[:-1], mu) - 1,
        'jac': lambda y: np.concatenate((mu, np.array([0])))
    }
    
    # bounds
    lb = np.zeros(m_all + 1)
    ub = np.repeat(np.Inf, m_all + 1)
    bounds_mktz = optimize.Bounds(lb, ub)
    
    return((eq_cons_kappa, eq_cons_budget), bounds_mktz)

def initial_point_markowitz(mu):
    # get inverse of the mean
    inverse_mu = 1/mu
    
    # index with maximum value
    imax = inverse_mu.argmax()
    
    # generate initial feasable point
    x0 = np.zeros(len(mu) + 1)
    x0[imax] = inverse_mu.max()
    x0[-1] = inverse_mu.max()
    return(x0)

def initial_params_markowitz(mu, sigma):
    # restrictions
    constraints, bounds = restrictions_markowitz(mu)
    
    # initial feasable point
    bet0 = initial_point_markowitz(mu)
    
    # augment + 1 dimension the var-covar matrix for the optimization problem
    AUGM_SIGMAm = np.vstack((sigma, np.zeros(sigma.shape[0])))
    AUGM_SIGMAm = np.hstack((AUGM_SIGMAm, np.zeros(AUGM_SIGMAm.shape[0]).reshape(-1, 1)))

    # dict out
    dict_out = {
        'constraints': constraints,
        'bounds': bounds,
        'x0': bet0,
        'S': AUGM_SIGMAm        
    }    
    
    return(dict_out)

def tidy_rawoptimze_markowitz(opt, mu, xtol=1e-6):
    # get values 
    xopt_raw = opt['x'][:-1]
    kappa = opt['x'][-1]
    
    # get inputs
    xopt = xopt_raw / np.sum(xopt_raw)
    
    # flags
    flag_kappa = np.abs(np.sum(xopt_raw) - kappa)
    flag_mean = np.abs(np.dot(xopt_raw, mu) - 1)
    flag_xsum = np.abs(np.sum(xopt) - 1)
    
    if xtol < flag_kappa or xtol < flag_mean:
        warnings.warn(
            f"""
            |sum(y) - kappa| = {flag_kappa} and/or
            |y'mu - 1| = {flag_mean} and/or
            |sum(x) - 1| = {flag_xsum} 
            exceeds the tolerance of {xtol}            
            """
        ) 
        
    if not opt['success']:
        warnings.warn("Optimization wasn't succesful")
    
    return(xopt)

def markowitz_stake(games, ftol=1e-12):  
    # mean-variance porftolio parameters
    mu = bets_mean(probas=games['probas'], odds=games['odds'])
    SIGMAm = covariance_games(games)
    
    # initial values for optimization
    dict_initials = initial_params_markowitz(mu, SIGMAm)
    
    # optimize
    opt_raw = optimize.minimize(
        fun=portfolio_volatility,           # minimize transformed sharpe ratio
        x0=dict_initials['x0'],             # initial point
        args=(dict_initials['S']),          # arguments for the function and the gradient
        method='SLSQP',                      
        jac=portfolio_volatility_jac,       # gradient
        bounds=dict_initials['bounds'],
        constraints=dict_initials['constraints'],
        options={'ftol': ftol, 'disp': False}    
    )    
    
    # get and tidy optimal x
    xoptimal = tidy_rawoptimze_markowitz(opt_raw, mu)
    
    # dict out
    dict_out = {
        'bets': xoptimal,
        'sharpe_ratio': sharpe_ratio(xoptimal, mu, SIGMAm),
        'portfolio_return': portfolio_returns(xoptimal, mu),
        'portfolio_volatility': portfolio_volatility(xoptimal, SIGMAm)        
    }
    
    return(dict_out)

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



# %% subset dataframe rows by matchweek and save in a list
# get matchweeks to predict
matchweek_test = tuple(range(18, 39))

# subste dataframe to last season
df_s = df_work[(df_work['season'] == 20) & (df_work['matchweek'].isin(matchweek_test))].copy()
 
# get subsets 
cols_probas = ['hatproba_home', 'hatproba_draw', 'hatproba_away']
cols_odds = ['maxo_h', 'maxo_d', 'maxo_a']
cols_results = ['result_H', 'result_D', 'result_A']

# get main dictionary
dict_games = dict()
for mtchwk in matchweek_test:
    dict_games['matchweek_' + str(mtchwk)] = {
        'probas': df2array(df_s, cols_probas, 'matchweek', mtchwk),
        'odds': df2array(df_s, cols_odds, 'matchweek', mtchwk),
        'results': df2array(df_s, cols_results, 'matchweek', mtchwk),
        'id_game': ids_array(df_s, 3, 'id_game', 'matchweek', mtchwk),
        'num_events': 3, # 3 posible outcomes
        }
    dict_games['matchweek_' + str(mtchwk)]['num_games'] = np.sum(dict_games['matchweek_' + str(mtchwk)]['results'])


# %% optimize with pyportfolioopt



# init class
# ef = pypfopt.efficient_frontier.EfficientFrontier(mu, SIGMAm, verbose=True)

# # get best weights for sharpe ratio
# xopt_pypfopt = ef.max_sharpe(risk_free_rate=ftol)
# # xopt_mivol = ef.min_volatility()
# xopt_tidy = ef.clean_weights()

# look_sum = sum([v for k, v in xopt_tidy.items()])

# # print(xopt_tidy)

# # look performance
# ef.portfolio_performance(verbose=True)

# # plot

# #  look metrics in own functions
# xopt_tidy_mine = np.array([v for k, v in xopt_tidy.items()])

# final_portfolio_volatility_pyport = portfolio_volatility(xopt_tidy_mine, SIGMAm)
# final_portfolio_return_pyport = portfolio_returns(xopt_tidy_mine, mu)
# final_sharperatio_pyport = sharpe_ratio(xopt_tidy_mine, mu, SIGMAm)



# %% optimize with functions
dict_stakes = dict()
for mtchwk, games in dict_games.items():
    print(f"Optimization for {mtchwk}")
    dict_stakes[mtchwk] = markowitz_stake(games)


# %% generate dataframe with all the stakes
df_bets = pd.DataFrame(
    columns=[
        'id_game', 'matchweek', 'results', 'results_categoric', 
        'odds', 'probas', 'bets'       
     ]
)

for mtchwk, games in dict_games.items():
    # generate data
    games.update(dict_stakes[mtchwk])
    [games['matchweek']] = re.findall('\d+', mtchwk) # get matchweek number and unpack from list
    games['results_categoric'] = tuple(islice(cycle(['home', 'draw', 'away']), games['num_games']*games['num_events']))
    # delete unnecesary values
    games.pop('num_games')
    games.pop('num_events')
    
    # append
    df_aux = pd.DataFrame(games)
    df_bets = pd.concat((df_bets, df_aux))

df_bets[['id_game', 'matchweek', 'results']] = df_bets[['id_game', 'matchweek', 'results']].apply(pd.to_numeric)
# %% merge bets and games dataframes
# join frames
df_main = (
    # left join
    pd.merge(
        left=df_bets,
        right=df_s[
            ['id_game', 'matchweek', 'date', 
            'hometeam', 'awayteam', 'market_tracktake']
        ],
        how='inner',
        on=['id_game', 'matchweek']    
    )
    # select and order columns
    [[ 
      'id_game', 'matchweek', 'date', 
      'hometeam', 'awayteam', 'results', 'results_categoric', 
      'market_tracktake', 'odds', 'probas', 'bets',
      'sharpe_ratio', 'portfolio_return', 'portfolio_volatility'
      
      ]]
)

# %% write csv
file_csv = PRODUCT_FOLDER + "ridge_complete_markowitz.csv"
df_main.to_csv(file_csv, index=False)





































