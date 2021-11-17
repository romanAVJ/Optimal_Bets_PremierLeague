# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 17:21:42 2021

@author: Ryo

Main script for financial model
"""
import pandas as pd
import numpy as np
import warnings

import scipy.optimize as optimize
import scipy.stats as sstats
from scipy.sparse import csc_matrix
from scipy import linalg
from itertools import product

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# =============================================================================
#### functions
# =============================================================================
# library
# general portfolio
def get_meanvar_params(games):
    # portfolio expectation - no risk free asset
    mu = bets_mean(probas=games['probas'], odds=games['odds'])
    
    # variance-covariance matrix
    SIGMAm = covariance_games(games)
    
    # dict out
    dict_main = {
        'mu': mu,
        'SIGMAm': SIGMAm        
    }
    
    return(dict_main)

def get_loggrowth_params(games, is_bivariate=False, tol=1e-6):
    # unpack values
    m = games['num_events']
    r = games['num_games']
    
    # posibility matrix
    OMEGAm = posibility_matrix(m=m, n=r, is_bivariate=is_bivariate)
    
    # generate values of dictionary
    dict_main = {
        # trasnposed win matrix
        'Wt': np.matmul(np.transpose(OMEGAm), np.diag(games['odds'])),        
        # probabilities
        'proba_posibilities': independent_probabilities(
            Mposibilities=OMEGAm, 
            probas=games['probas'],
            is_bivariate=is_bivariate,
            tol=tol
        )        
    }    
    
    return(dict_main)

def get_portfolio_params(games, tol=1e-6):
    # look if is bivariate
    # look if is bivariate or multivariate
    if games['num_events'] > 1:
        is_bivariate=False
    else:
        is_bivariate=True
    
    # get params
    # kelly params
    dict_out = get_meanvar_params(games) 
    # markowitz params
    dict_out.update(get_loggrowth_params(games, is_bivariate=is_bivariate, tol=tol))  

    return(dict_out)

# kelly optimization
def posibility_matrix(m, n, is_bivariate=False):    
    # idea asked by autor in 
    # https://stackoverflow.com/questions/69531178/create-an-sparse-matrix-from-a-list-of-tuples-having-the-indexes-of-the-column-w
    if not is_bivariate:
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
    
        # generate sparse matrix of posibilities and cast to numpy
        array1d_sparse = np.ones(indptr[-1], dtype=int)
        A = csc_matrix((array1d_sparse, indices_event, indptr)).toarray()
        
    else:
        # generate all binomial posible combinations
        posibilities = list(product(range(2), repeat=n)) 
        
        # get posiblility matrix
        A = np.transpose(np.array(posibilities, dtype=np.int32))  
   
    return(A)

def set_probas(omega, probas):
    # compare if an ouctome happened or not and return probability
    p = np.where(omega > 0, probas, 1 - probas) 
    
    return(p) 

def independent_probabilities(Mposibilities, probas, is_bivariate=False, tol=1e-6):
    # get probaility matrix
    if not is_bivariate:
        # get probability matrix with 0's
        Pm = np.matmul(np.diag(probas), Mposibilities)
    
        # get proba matrix
        Pm[Pm == 0] = 1      
    else:
        # get probabilities for each element in omega
        Pm = np.apply_along_axis(
            func1d=set_probas,
            axis=0,                 # by row
            arr=Mposibilities,
            probas=probas           # *args
        )
    
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

def log_growth(bets, Wt, probs):   
    # get expected log return
    Gl = np.dot(probs, np.log(1 + np.matmul(Wt, bets) - np.sum(bets)))
    
    return(Gl)

def log_growth_grad(bets, Wt, probs): 
    # numerator
    xwin_return = Wt - 1
    xreturn = 1 + np.matmul(Wt, bets) - np.sum(bets)
    
    # divide each row by xreturn
    xwin_return_gradient = np.divide(xwin_return, xreturn[:, np.newaxis]) 
    
    # gradient
    Gl_grad = np.dot(np.transpose(xwin_return_gradient), probs)

    return(Gl_grad)

# aux functions for kelly stake
def neg_log_growth(bets, Wt, probs):
    return(- log_growth(bets, Wt, probs))

def neg_log_growth_grad(bets, Wt, probs):
    return(- log_growth_grad(bets, Wt, probs))

def restrictions_bounds_kelly(m_all, fractional=1):
    # constraint
    ineq_cons_budget = {
        'type': 'ineq',
        'fun': lambda x: fractional - np.sum(x),
        'jac': lambda x: - np.ones(x.shape)
    }
    
    constraints = (ineq_cons_budget,)
    
    # bounds
    lb = np.zeros(m_all)
    ub = np.repeat(fractional, m_all)
    bounds = optimize.Bounds(lb, ub)
    
    return(constraints, bounds)   

def get_initial_params_kelly(games, fractional=1, is_bivariate=False):
    # unpack values
    m = games['num_events']
    r = games['num_games']
    
    if not is_bivariate:
        m_all = m * r
    else:
        m_all = r      
        
    # restrictions
    constraints, bounds = restrictions_bounds_kelly(m_all, fractional=fractional)
    
    # initial feasble point
    x0 = np.zeros(m_all)
    
    # dict out
    dict_out = {
        'constraints': constraints,
        'bounds': bounds,
        'x0': x0   
    }  
    
    return(dict_out)

def kelly_stake(games, params, fractional=1, xmin=1e-4, ftol=1e-12):
    # look if is bivariate or multivariate
    if games['num_events'] > 1:
        is_bivariate=False
    else:
        is_bivariate=True
    
    #  maximize(f) <-> - minimize(-f)
    # initial values
    init_values = get_initial_params_kelly(
        games=games, 
        is_bivariate=is_bivariate, 
        fractional=fractional
    )
    
    # quadratic programming optimization (convex function)
    WTm = params['Wt'].copy()
    probas = params['proba_posibilities'].copy()
    
    opt = optimize.minimize(
        fun=neg_log_growth,             # maximize the negative log growth
        x0=init_values['x0'],           # initial point
        args=(WTm, probas),             # arguments for the function and the gradient
        method='SLSQP',                  
        jac=neg_log_growth_grad,        # gradient
        bounds=init_values['bounds'],
        constraints=init_values['constraints'],
        options={'ftol': ftol, 'disp': False}    
    )    
    
    if not opt['success']:
        warnings.warn("Optimization for Kelly wasn't succesful")     
    
    # tidy results
    dict_result = {
        'fmax': - opt['fun'],
        'x_opt': np.where(opt['x'] >= xmin, opt['x'], 0)        
    }
    
    return(dict_result)

# markowitz optimization
def portfolio_returns(bets, returns):   
    # estimate mean
    portfolio_return = np.dot(bets, returns)
    
    return(portfolio_return)

def portfolio_volatility(bets, sigma):     
    # get volatitilty defined as total variance
    vol = np.dot(bets, np.matmul(sigma, bets)) # x'S x

    # if the matrix isn't completelty semi positive definite:
    return(vol)

def portfolio_volatility_jac(bets, sigma):
    gradient = 2 * np.matmul(sigma, bets)
    return(gradient)

def sharpe_ratio(bets, returns, sigma):
    sharpe = portfolio_returns(bets, returns) / np.sqrt(portfolio_volatility(bets, sigma))
    return(sharpe)

def bets_mean(probas, odds):
    rho = odds * probas - 1 # decimal odds    
    return(rho)

def exclusive_bets_covariance(odds, probas):
    # get var-covar matrix of a multinomial distribution
    # i.e. Var(M) =  diag(p) - p p.T (outer product)
    sigma_multinomial = np.diag(probas) - np.outer(probas, probas)
    
    # Var(DM - 1) = D Var(M) D.T
    diag_odds = np.diag(odds)
    sigma_exclusive_bets = np.matmul(
        np.matmul(diag_odds, sigma_multinomial), diag_odds
    )        
    
    return(sigma_exclusive_bets)

def covariance_games(games):   
    # split games
    # get index to split array for all possible outcomes in a game
    games_index_split = np.cumsum(pd.Series(games['id_game']).value_counts().to_numpy())[:-1]    
    # get list of odds & probas per game
    games_odds = np.array_split(games['odds'], games_index_split)
    games_probas = np.array_split(games['probas'], games_index_split)    
    
    # get var covar matrix per game
    list_sigmas = [
        exclusive_bets_covariance(o, p) for o, p in zip(games_odds, games_probas)
    ]  
    
    # variance covariance for all games
    VAR_COVARm = linalg.block_diag(*list_sigmas)
    
    return(VAR_COVARm)

def restrictions_bounds_markowitz(mu):
    m_all = len(mu)
    # note: (y, kappa) <-> (x[:-1], x[-1])
    # constraints
    # 1. sum(y) = kappa    
    eq_cons_kappa = {
        'type': 'eq',
        'fun': lambda y: np.sum(y[:-1]) - y[-1],
        'jac': lambda y: np.concatenate((np.ones(m_all), np.array([-1])))
    }
    # 2. y'mu = 1
    eq_cons_budget = {
        'type': 'eq',
        'fun': lambda y: np.dot(y[:-1], mu) - 1,
        'jac': lambda y: np.concatenate((mu, np.array([0])))
    }
    constraints = (eq_cons_kappa, eq_cons_budget)
    
    # bounds
    lb = np.zeros(m_all + 1) # +1 for kappa
    ub = np.repeat(np.Inf, m_all + 1) # unbounded kappa values
    bounds = optimize.Bounds(lb, ub)        
    
    return(constraints, bounds)

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

def get_initial_params_mkwtz(mu, sigma):
    # restrictions
    constraints, bounds = restrictions_bounds_markowitz(mu)
    
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
    
def tidy_rawoptimze_markowitz(opt, mu, xmin, xtol=1e-6):
    # get values 
    xopt_raw = opt['x'][:-1]
    kappa = opt['x'][-1]
    
    # get inputs to original dimensions
    xopt = xopt_raw / np.sum(xopt_raw)
    xopt_tidy = np.where(xopt >= xmin, xopt, 0)
    
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
        warnings.warn("Optimization for markowitz wasn't succesful")
    
    return(xopt_tidy, xopt)

def markowitz_stake(games, params, xmin=1e-4, ftol=1e-12):  
    # unpack values
    mu_fixed = params['mu'].copy()
    SIGMAm_fixed = params['SIGMAm'].copy()
    
    # initial values for optimization
    dict_initials = get_initial_params_mkwtz(
        mu=mu_fixed, 
        sigma=SIGMAm_fixed
    )
    
    # optimize
    # idea given by Alonso Gonzalez Valdes in his Bachelor's thesis
    # which was taken from Reha Tutuncu in his book, page 157
    opt_raw = optimize.minimize(
        fun=portfolio_volatility,           # minimize transformed (convex) sharpe ratio
        x0=dict_initials['x0'],             # initial point
        args=(dict_initials['S']),          # arguments for the function and the gradient
        method='SLSQP',                      
        jac=portfolio_volatility_jac,       # gradient
        bounds=dict_initials['bounds'],
        constraints=dict_initials['constraints'],
        options={'ftol': ftol, 'disp': False}    
    )    
    
    # get and tidy optimal x (from dimension M + 1 to M)
    xoptimal_tidy, xoptimal_raw = tidy_rawoptimze_markowitz(
        opt=opt_raw, 
        mu=params['mu'],
        xmin=xmin
    )
    
    # dict out
    dict_out = {
        'fmax': sharpe_ratio(xoptimal_raw, mu_fixed, SIGMAm_fixed),
        'x_opt': xoptimal_tidy
    }
    
    return(dict_out)

def simulation_performance(x, games, params, num_sim, fractional=1):
    # unpack values
    xf = x * fractional
    mu_fixed = params['mu'].copy()
    SIGMAm_fixed = params['SIGMAm']
    
    # dict out
    win_loss = np.where(games['result'] == 1, xf * (games['odds'] - 1), - xf)
    
    dict_out = {              
        # portfolio metrics
        'matchweek': games['matchweek'],
        'wealth': np.sum(win_loss),
        'sharpe_ratio': sharpe_ratio(xf, mu_fixed, SIGMAm_fixed),
        'log_growth': log_growth(xf, params['Wt'], params['proba_posibilities']),
        'num_stakes': np.sum(xf > 0),
        'percent_wage': np.sum(xf),
        'total_hits': np.sum((games['result'] == 1) & (xf > 0)),
        'num_sim': num_sim
    }    
    
    return(dict_out)

def portfolio_performance(x, games, params, fractional=1, model_name='model'):
    # unpack values
    xf = x * fractional
    mu_fixed = params['mu'].copy()
    SIGMAm_fixed = params['SIGMAm']
    
    # dict out
    win_loss = np.where(games['result'] == 1, xf * (games['odds'] - 1), - xf)
    
    dict_out = {
        # about model
        'model': model_name,
        'type': np.where(games['num_events'] > 1, 'multiple', 'single'),
        'fractional': fractional,
        'matchweek': games['matchweek'],
        
        # about strategy
        'num_stakes': np.sum(xf > 0),
        'percent_wage': np.sum(xf),
        'max_stake2percent_wage': np.max(xf) / np.sum(xf),
        'charged_tt': np.dot(games['market_tracktake'], xf),
        
        # portfolio metrics
        'sharpe_ratio': sharpe_ratio(xf, mu_fixed, SIGMAm_fixed),
        'xreturn': portfolio_returns(xf, mu_fixed),
        'volatility': portfolio_volatility(xf, SIGMAm_fixed),
        'log_growth': log_growth(xf, params['Wt'], params['proba_posibilities']),
        
        # strategy performance
        'wealth': np.sum(win_loss),
        'total_hits': np.sum((games['result'] == 1) & (xf > 0)),
        'percent_hits': np.sum((games['result'] == 1) & (xf > 0)) / np.sum(xf > 0),
        'best_win': np.max(win_loss),
        'worst_loss': np.min(win_loss)
    }
    
    return(dict_out)

def portfolios_optimization(tournment, df,  stake_model='kelly', fractional=1, verbose=True):
    # init params
    COLS = ['id_game', 'result_categoric', 'bets']
    
    # optimize each matchweek
    list_metrics = list()
    df_bets = pd.DataFrame(columns=COLS)
    for mtchwk, games in tournment.items():
        # print
        if verbose:
            print(f"\nOptimization for {mtchwk}")
        
        # get parameters
        params_games = get_portfolio_params(games)
        
        # optimize
        if stake_model == 'kelly':
            # note: besides optimizing for sum(l) <= b, weren't taking this
            # by the fact it will not be completely comparable with markowitz
            # to revert set kelly_stake(fractional=fractional)
            bets = kelly_stake(games, params_games).get('x_opt')
        elif stake_model == 'markowitz':
            bets = markowitz_stake(games, params_games).get('x_opt')
        else:
            bets = None
            print(f"The stake model {stake_model} hasn't been implemented")
            
        games_aux = {k: v for k, v in games.items() if k in COLS} # subset dict
        games_aux['bets'] =  bets
        
        df_aux = pd.DataFrame(games_aux)
        df_bets = pd.concat((df_bets, df_aux))
        
        # get metrics & performance of portfolio
        # note: if kelly fractional != 1 and fractional bbetting is implemented
        # then the portfolio performance should be set fractional=1
        metrics = portfolio_performance(
            x=bets,
            games=games,
            params=params_games,
            fractional=fractional,
            model_name=stake_model           
        )
        list_metrics.append(metrics)
        
        # print metrics & performance for the given matchweek
        if verbose:
            print(
                f"""
                     num stakes: {metrics['num_stakes']} 
                     wage betted: {np.round(100 * metrics['percent_wage'])}%
                     percent hitted: {np.round(100 * metrics['percent_hits'])}%
                     final wealth: {np.round(100 * metrics['wealth'])}%                
                \n"""
            )

    # table of metrics
    df_metrics = (
        # instantiate metrics as a dataframe
        pd.DataFrame(list_metrics)
        
        # arrange ascendent by matchweek
        .sort_values('matchweek')
        
        # cummulative product
        .assign(
            cumm_wealth = lambda x: np.cumprod((1 + x['wealth'])),
            sharpe_ratio = lambda x: np.where(x['sharpe_ratio'] > 1e+3, np.nan, x['sharpe_ratio']),
            volatility = lambda x: np.where(x['volatility'] < 1e-6, 0, x['volatility'])                                      
       )        
    )    
    # sort table of metrics
    IMPO_COLS = ['model', 'type', 'matchweek', 'fractional', 'wealth', 'cumm_wealth']
    aux_index = df_metrics.columns.difference(IMPO_COLS, sort=False).tolist()
    df_metrics = df_metrics[IMPO_COLS + aux_index]    
    
    # table bets
    df_total_bets = (
        # left join
        pd.merge(
            left=df,
            right=df_bets,
            how='left',
            on=['id_game', 'result_categoric']
        )
        # select and order columns
        [[ 
          'id_game', 'season','matchweek', 
           'date', 'hometeam', 'awayteam', 'market_tracktake', 
          'result', 'result_categoric', 
          'odds', 'probas', 'bets'    
          ]]
        # modify 'nans' in bets 
        .assign(
            bets = lambda x: x['bets'].fillna(value=0),
            id_game = lambda x: pd.to_numeric( x['id_game'])
        )
    )

    # dict out
    dict_out = {
        'metrics': df_metrics,
        'bets': df_total_bets        
    }      
    
    return(dict_out)

def bets_simulation(tournment, n_sim=1e+3, kforce=1, fractional=1, xmin=1e-4, seed=8):
    # get values
    n_mtchw = len(tournment)
    
    # get auxiliary macthweek
    key_aux = next(iter(tournment))
    mtch_aux = tournment[key_aux]
    n_bets = mtch_aux['num_events'] * mtch_aux['num_games']
    
    # instantiate randomness
    rng = np.random.default_rng(seed)
    
    # simulate (dirichlet)
    rnd_dirichlet = rng.dirichlet(
        alpha=np.ones(n_bets) * kforce, # to change dirichlet distribution
        size=(n_sim, n_mtchw)    
    )
    rnd_dirichlet = rnd_dirichlet * fractional # to change bet proportion
    
    # get params
    dict_params = {k: get_portfolio_params(v) for k,v in tournment.items()}
    
    # simulate
    # for each total simulation
    list_sim = list()
    
    for j, s in enumerate(rnd_dirichlet):
        # for each simulation of matchweek of the total simulation
        list_metrics = list()
        
        for i, k in enumerate(tournment.keys()):
            # tidy bets
            s_mtchw = np.where(s[i] >= xmin, s[i], 0)
            
            # look performance
            perf = simulation_performance(
                x=s_mtchw, 
                games=tournment[k], 
                params=dict_params[k],
                num_sim=j
            )
            list_metrics.append(perf)
                        
            # if gambler got ruined then break
            if (perf['wealth'] == - perf['percent_wage']) and fractional == 1:
                break
               
        # generate metrics' dataframe of the whole simulation 
        df_metrics = (
            # instantiate metrics as a dataframe
            pd.DataFrame(list_metrics)
            
            # arrange ascendent by matchweek
            .sort_values('matchweek')
            
            # cummulative product
            .assign(
                cumm_wealth = lambda x: np.cumprod((1 + x['wealth'])),
                sharpe_ratio = lambda x: np.where(x['sharpe_ratio'] > 1e+3, np.nan, x['sharpe_ratio']),
                log_growth_r = lambda x: np.where(np.isnan(x['log_growth']), -100, x['log_growth']),                 
                log_growth = lambda x: np.where(x['log_growth_r'] == - np.inf, -100, x['log_growth_r']),      
                growth = lambda x: np.exp(x['log_growth'])                     
           )
            # sort columns
            [[
                "num_sim", 'matchweek', 'num_stakes', 'percent_wage',
                'total_hits', 'cumm_wealth', 'wealth', 'log_growth',
                'growth', 'sharpe_ratio'                
                ]]
            # reset index
            .reset_index(drop=True)
        )
        
        # append to simulation list
        list_sim.append(df_metrics)
    
    # concat
    df_main = pd.concat(list_sim)
    
    return(df_main)

def median_simulation_wealth(fractional, kforce, tournment, n_sim=1e+3, xmin=1e-4, seed=8):
    # generate simualation
    bets_sim_aux = bets_simulation(
        tournment=tournment,
        n_sim=n_sim, 
        kforce=kforce,
        fractional=fractional,
        xmin=xmin,
        seed=10
    )
    
    # get final wealth per simulation
    wn = bets_sim_aux.groupby('num_sim')['cumm_wealth'].apply(lambda x: x.tail(1))
    
    # median
    wn_median = np.median(wn)
    
    return(wn_median)

# colab
def busted_model(model):
    # unpack values
    df_stats_r = model['metrics'].copy()
    df_bets_r = model['bets'].copy()
    
    # find matchweek when the gambler got ruined
    if df_stats_r['cumm_wealth'].min() <= 0:
        index_ruin = np.argmax(df_stats_r['cumm_wealth'] <= 0)
        wn_last = df_stats_r.iloc[index_ruin]['matchweek']        
    else:
        wn_last = df_stats_r['matchweek'].max()
    
    # subset
    df_stats = df_stats_r.query("matchweek <= @wn_last")
    df_bets = df_bets_r.query("matchweek <= @wn_last")
        
    return(df_stats, df_bets)

def get_dummies_results(df, var='y', prefix='y'):
    # working data frame
    df = df.copy()
    
    # one hot encode results
    y = pd.get_dummies(df[var], prefix=prefix, drop_first=False, dtype=int)
    names_y = y.columns.values
    
    # append to dataframe
    df[names_y] = y
    
    # reorder columns
    columns_names = df.columns.values
    columns_names = np.concatenate([names_y, columns_names[:-3]])
    df = df[columns_names]
    
    # drop result
    df.drop(columns=var, inplace=True)
    
    return(df)

def get_df():
    
    # read dataframes
    df_prediction = pd.read_csv("Results\\statistical_estimates\\dropout_prediction.csv")
    df_odds = pd.read_csv("Data\\Main_DBB\\stake_odds.csv")
    
    # join frames
    df = (
        # left join
        pd.merge(
            left=df_prediction,
            right=df_odds,
            how='inner',
            on=['season', 'matchweek', 'hometeam', 'awayteam']
        )
        # get dummies for result
        .pipe(
            func=get_dummies_results,
            var='result',
            prefix='result'        
        )
        # generate key_index
        .assign(
            id_game = range(1, len(df_odds) + 1)
        )
        # rename columns
        .rename(
            columns = {
                'result_H': 'result_h',
                'result_D': 'result_d',
                'result_A': 'result_a',
                'hatproba_home': 'probas_h',
                'hatproba_draw': 'probas_d',
                'hatproba_away': 'probas_a',
                'maxo_h': 'odds_h',
                'maxo_d': 'odds_d',
                'maxo_a': 'odds_a'
            }            
        )     
        # select and arrange columns
        [[
          'id_game', 'season', 'matchweek', 
          'date','hometeam', 'awayteam', 'market_tracktake',
          'result_h', 'result_d', 'result_a',       
          'probas_h', 'probas_d', 'probas_a', 
          'odds_h', 'odds_d', 'odds_a'
        ]]
        # format wide 2 long
        .pipe(
            func=pd.wide_to_long,
            stubnames=['result', 'probas', 'odds'],
            i='id_game',
            j='result_categoric',
            sep='_',
            suffix='\\D+' # no numeric suffixes            
        )
        # unstack index
        .reset_index(
            level=['id_game', 'result_categoric']
        )             
        # re-select and order
        [[
          'id_game', 'season', 'matchweek', 
          'date','hometeam', 'awayteam', 'market_tracktake',
          'result', 'result_categoric', 'probas', 'odds'
        ]]
        .sort_values(
            by=['id_game', 'result_categoric'],
            axis='index',
            ascending=[True, False]
        )
    )    
    
    return(df)

def get_games_dict(df):
    # unpack values
    min_mw = df['matchweek'].min()
    max_mw = df['matchweek'].max()    
    
    # instatiate
    dicts = dict()
    range_matches = range(min_mw, max_mw + 1)
    
    for mtchwk in range_matches:
        # generate dictionary key
        key = 'matchweek' + str(mtchwk)
        
        # subset dataframe by macthweek
        df_aux = df.copy().query("matchweek == @mtchwk")

        # generate values of dictionary
        dicts[key] = {
            'id_game': df_aux['id_game'].to_numpy(),    
            'matchweek': mtchwk,    
            'odds': df_aux['odds'].to_numpy(),    
            'probas': df_aux['probas'].to_numpy(),    
            'result': df_aux['result'].to_numpy(),    
            'result_categoric': df_aux['result_categoric'].to_numpy(),
            'market_tracktake': df_aux['market_tracktake'].to_numpy(),
            'num_events': df_aux['id_game'].value_counts().to_numpy()[0],
            'num_games': len(pd.unique(df_aux['id_game']))
        }
        

    
    return(dicts)

def model_summary(model):    
    # subset bets & metrics ifgambler got ruined
    df_stats, df_bets = busted_model(model)
    
    # get individual bets performance
    all_bets = np.where(
        df_bets['result'] == 1, 
        df_bets['bets'] * (df_bets['odds'] - 1),
        - df_bets['bets']
    )
    all_bets = all_bets[all_bets != 0]
    
    # dict out
    model_name = df_stats.iloc[-1]['model']
    type_name = df_stats.iloc[-1]['type'].item()
    fractional_name = str(df_stats.iloc[-1]['fractional'])
    dict_out = {
        'model': '_'.join((model_name, type_name, fractional_name)),
        'final_wealth': df_stats['cumm_wealth'].iloc[-1],
        'total_bets': df_stats['num_stakes'].sum(),
        'mean_num_bets_per_matchweek': df_stats['num_stakes'].mean(),
        'mean_wage_bet_per_matchweek': df_stats['percent_wage'].mean(),
        'total_hits': df_stats['total_hits'].sum() / df_stats['num_stakes'].sum(),
        'mean_sharpe': df_stats['sharpe_ratio'].mean(),
        'mean_log_growth': df_stats['log_growth'].mean(),
        'mean_volatility': df_stats['volatility'].mean(),
        'pval_bets': sstats.ttest_1samp(all_bets, popmean=0, alternative='greater')[1],
        'pval_wealth': sstats.ttest_1samp(df_stats['wealth'], popmean=0, alternative='greater')[1]      
    }    
    
    return(dict_out)

def show_performance(model, model_summary, model_name='model', perform='all'):
    # subset bets & metrics if gambler got ruined
    df_stats, df_bets = busted_model(model)
    model_updated = dict(metrics=df_stats, bets=df_bets)    
    
    # inner plots
    def plt_cumm_wealth(model, model_summary, model_name):
        # unpack values
        df_stats = model['metrics'] 
        
        # get values
        wn = df_stats['cumm_wealth'].to_numpy()
        wn = np.concatenate((np.array([1]), wn))
        
        wi = df_stats['wealth'].to_numpy()
        wi = np.concatenate((np.array([0]), wi))
        
        x0 = np.arange(len(wn))
        
        mtchwlst = df_stats['matchweek'].to_list()
        mtchw = ['J' + str(mm) for mm in mtchwlst]
        mtchw.insert(0, 't0')
        
        # plot it
        fig, ax = plt.subplots()
        
        # plot wn
        ax.plot(x0, wn, color='black', alpha=0.9)
        
        # plot wi
        ax.bar(x0, wi, color='gray', alpha = 0.5)
        
        # plot text
        ss = [str(np.round(100 * w, decimals=1)) + '%' for w in wn]
        txt_ss = [plt.text(i, wn[i], ss[i], fontsize=15) for i in x0 if i % 3 == 0 or i == x0[-1]]
        adjust_text(
            txt_ss,
            only_move={'points':'y', 'text':'y'}, 
            autoalign='y', 
            force_points=0    
        )
        
        # show 0 threshold
        plt.axhline(y=1, linewidth=0.9, color='gray', linestyle='-.')
        plt.axhline(y=0, linewidth=0.5, color='black')
        
        # pretty graph
        plt.ylabel('$W_{n}$', fontsize=20)
        plt.xlabel('Jornada', fontsize=20)
        plt.title('Ganancias y Pérdidas:' + model_name, fontsize=30)
        plt.xticks(x0, rotation='vertical')
        ax.set_xticklabels(mtchw)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        
        plt.ylim(-1, 1.2)
        
        # reescale plot
        fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.show()
        return()
    
    def plt_bets(model, model_summary):
        # unpack values
        df_bets = model['bets']   
        
        # plot
        # get values
        all_bets = np.where(
            df_bets['result'] == 1, 
            df_bets['bets'] * (df_bets['odds'] - 1),
            - df_bets['bets']
        )
        all_bets = all_bets[all_bets != 0]
        
        # plot density 
        fig, ax = plt.subplots()
        
        # sns.kdeplot(all_bets, x="apuestas", fill=True, color='black')
        sns.histplot(all_bets, kde=True, color='black', fill=False)
        
        # show thresholds
        ax.axvline(x=0, linewidth=0.9, color='black', linestyle='-.')
        
        # annotate
        s = f"pval t-test: {np.round(model_summary['pval_bets'], 4)}\nmedia: {np.round(np.mean(all_bets), 4)}"
        at = AnchoredText(
            s,
            prop=dict(size=25),
            frameon=True,
            loc='upper right'
        )
        at.patch.set_boxstyle("round")
        ax.add_artist(at)
        
        #  preatty graph
        plt.ylabel('Densidad', fontsize=20)
        plt.xlabel('Apuestas', fontsize=20)
        plt.yticks(all_bets, "")
        plt.title('Distribución de las Apuestas Ejercidas', fontsize=30)
        
        # reescale plot
        fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.show()
        return()
    
    def plt_ind_wealth(model, model_summary):
        # unpack values
        df_stats = model['metrics']   
        
        # plot
        # get values
        wn = df_stats['wealth']
        # plot density 
        fig, ax = plt.subplots()
        
        # sns.kdeplot(wn, x="apuestas", fill=True, color='black')
        sns.histplot(wn, kde=True, color='black', fill=False)
        
        # show thresholds
        ax.axvline(x=0, linewidth=0.9, color='black', linestyle='-.')
        
        # annotate
        s = f"pval t-test: {np.round(model_summary['pval_wealth'], 4)}\nmedia: {np.round(np.mean(wn), 4)}"
        at = AnchoredText(
            s,
            prop=dict(size=25),
            frameon=True,
            loc='upper right'
        )
        at.patch.set_boxstyle("round")
        ax.add_artist(at)
        
        #  preatty graph
        plt.ylabel('Densidad', fontsize=20)
        plt.xlabel('Apuestas', fontsize=20)
        plt.yticks(wn, "")
        plt.title('Distribución de las Ganancias/Pérdidas', fontsize=30)
        
        # reescale plot
        fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.show()
        return()
    
    # plot them
    if perform == 'cumm_wealth' or perform == 'all':
        plt_cumm_wealth(model_updated, model_summary, model_name)
    if perform == 'bets' or perform == 'all':
        plt_bets(model_updated, model_summary)
    if perform == 'ind_wealth' or perform == 'all':
        plt_ind_wealth(model_updated, model_summary)
    
    return()

def fractional_models(tournment, df, type_model='kelly', fractions=[1, 0.75, 0.5, 0.25]):
    # init
    df_wealths = pd.DataFrame(columns=['fraction', 'matchweek', 'cumm_wealth'])
    
    # optimize for each fraction
    for f in fractions:
        # optimize
        aux_metrics = portfolios_optimization(
            tournment, 
            df, 
            type_model, 
            fractional=f, 
            verbose=False
        ).get('metrics')
        
        # get cummulative wealth and matchweeks 
        wn = aux_metrics['cumm_wealth'].to_numpy()
        wn = np.concatenate((np.array([1]), wn))
        
        mtchw = aux_metrics['matchweek'].to_list()
        # mtchw = ['J' + str(mm) for mm in mtchwlst]
        mtchw.insert(0, min(mtchw) - 1)
        mtchw = np.array(mtchw)
        
        # append to dataframe
        df_aux = pd.DataFrame(dict(fraction=f, matchweek=mtchw, cumm_wealth=wn))
        df_wealths = pd.concat((df_wealths, df_aux))
        
    df_wealths['matchweek'] = pd.to_numeric(df_wealths['matchweek'])
        
    # plot
    dfaux = df_wealths.copy()

    # get labels
    last_wn = dfaux.groupby('fraction').tail(1)
    dfaux.set_index('matchweek', inplace=True)
    
    # begin plot
    # plot lines
    fig, ax = plt.subplots()
    dfaux.groupby('fraction')['cumm_wealth'].plot(color='gray', ax=ax)
    
    # make space 4 labels
    left, right = plt.xlim()
    plt.xlim((left, right + 2))
    
    # labels
    last_game = last_wn.iloc[0, 1]
    ss = ['f' + str(round(100*f)) + '%' for f in fractions]
    n_plot = len(fractions)
    txt_ss = [plt.text(last_game + 2, last_wn.iloc[i, 2], ss[i], fontsize=15) for i in range(n_plot)]
    
    adjust_text(
        txt_ss,
        only_move={'points':'y', 'text':'y'}, 
        autoalign='y', 
        force_points=0    
    )
    
    # show 0 threshold
    plt.axhline(y=1, linewidth=0.9, color='gray', linestyle='-.')
    
    # pretty graph
    plt.ylabel('$W_{n}$', fontsize=20)
    plt.xlabel('Jornada', fontsize=20)
    plt.title('Estrategias Fraccionales: ' + type_model, fontsize=30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    
    # y limits
    left, right = plt.ylim()
    plt.ylim((0, right))
    
    # reescale plot
    fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    
    plt.show()
    
    return()

def plt_sim_vs_model(df_bets, strategy_metrics):
    # concat matchweek '0' with wealth = 1 to dataframe
    bets_sim_plot = (
        df_bets
        .groupby('num_sim')
        .apply(
            lambda x: x.append({'num_sim': x.name, 'cumm_wealth': 1, 'matchweek': x['matchweek'].min() - 1 }, ignore_index=True).astype({'num_sim': int})
        )
        .apply(lambda x: x.reset_index(drop=True))
        .sort_values(['num_sim', 'matchweek'])
        .reset_index(drop=True)
    )    
    
    # init plot
    fig, ax = plt.subplots()
    
    # plot lines
    bets_sim_plot.set_index('matchweek', inplace=True)
    bets_sim_plot.groupby('num_sim')['cumm_wealth'].plot(color='gray', alpha=0.4, ax=ax)
    
    # make space 4 labels
    left, right = plt.xlim()
    plt.xlim((left, right + 2))
        
    # show 0 threshold
    plt.axhline(y=1, linewidth=0.9, color='gray', linestyle='-.')
    
    # pretty graph
    plt.ylabel('$W_{n}$', fontsize=20)
    plt.xlabel('Jornada', fontsize=20)
    plt.title('Estrategias Simuladas contra Modelo', fontsize=30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    
    # y limits
    left, right = plt.ylim()
    plt.ylim((0, 3))
    
    # add main plot  
    # get values of my model
    wn = np.concatenate((np.array([1]), strategy_metrics['cumm_wealth']))
    mtchw = strategy_metrics['matchweek'].to_list()
    mtchw.insert(0, min(mtchw) - 1)
    mtchw = np.array(mtchw)
    
    ax.plot(mtchw, wn, color='black', linewidth=2)
    plt.text(mtchw.max() + 2e-1, wn[-1], s="Modelo", fontsize=20)
    
    # reescale plot
    fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.show()
    return()


def scalar_log_growth(l, p, o):
    gl = p * np.log(1 + l * o) + (1 - p) * np.log(1 - l)
    return(gl)

def scalar_log_growth_d(l, p, o):
    gl_d = p * (o / (1 + l * o)) - (1 - p) * (1/(1-l))
    return(gl_d)

def eval_bet(bet, result, odd):
    rho = np.where(result == 1, (odd - 1) * bet, - bet)
    return(rho)


# =============================================================================
# %%main
# =============================================================================
# %% 0 kelly examples
# eps = 1e-15
# # games
# p_aux = 1/2
# o_aux = 5

# game_aux = {
#     'id_game': 0,
#     'num_events': 1,
#     'num_games': 1,
#     'odds': np.array([o_aux]),
#     'probas': np.array([p_aux])
#     }

# # get params
# params_aux = get_portfolio_params(game_aux)

# # look kelly bell
# ll = np.linspace(0, 1, 100)
# gl_aux = [log_growth(np.array([bet]), Wt=params_aux['Wt'], probs=params_aux['proba_posibilities']) for bet in ll]
# # lc = 
# aux_opt = kelly_stake(
#     games=game_aux,
#     params=params_aux    
#     )


# # find root for G(l)
# l_opt = aux_opt['x_opt'].squeeze()
# sol = optimize.root_scalar(f=scalar_log_growth, x0=0.4, args=(p_aux, o_aux - 1), method='brentq', bracket=(0.1, 0.9))
# lc = sol.root

# # plot
# fig, ax = plt.subplots()
# ax.plot(ll, gl_aux, color='black')

# # annotate
# s = f"momios decimales: {o_aux} \nprobabilidad: {p_aux}"
# at = AnchoredText(
#     s,
#     prop=dict(size=25),
#     frameon=True,
#     loc='lower left'
# )
# at.patch.set_boxstyle("round")
# ax.add_artist(at)

# # show 0 threshold
# plt.axhline(y=0, linewidth=0.8, color='black')
# plt.axvline(x=aux_opt['x_opt'], color='gray', linestyle='-.')
# plt.axvline(x=lc,  color='gray', linestyle='-.')

# # pretty graph
# plt.ylabel('$G(l)$', fontsize=20)
# plt.xlabel('Fracción de Riqueza Apostada $l$', fontsize=20)
# plt.title('Log-Crecimiento', fontsize=30)

# ax.set_xticks([0, 0.075, l_opt, 0.5, 0.675, lc, 0.9, 1])
# ax.set_xticklabels(["0%", '$l_{*}^{-}$', '$l_{*}$',  "50%", '$l_{*}^{+}$', '$l_{c}$', '$l_{c}^{+}$', "100%"])
# ax.tick_params(axis='both', which='major', labelsize=50)

# # reescale plot
# fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
# ax.tick_params(axis='both', which='major', labelsize=20)
# plt.show()


# # %% 0.1 simulated growth before l_opt in l_opt post l_opt in lc and post lc
# N = 1000
# rng = np.random.default_rng(42)
# results_sim = rng.binomial(n=1, p=p_aux, size=N)

# # generate dataframe
# df_sim_kelly = (
#     # generate dataframe
#     pd.DataFrame({
#         'trail': range(1,N+1),
#         'result': results_sim,
#         'l_opt_minus': eval_bet(0.075, results_sim, o_aux),    
#         'l_opt': eval_bet(l_opt, results_sim, o_aux),    
#         'l_opt_plus': eval_bet(0.675, results_sim, o_aux),    
#         'l_c': eval_bet(lc, results_sim, o_aux),    
#         'l_c_plus': eval_bet(0.9, results_sim, o_aux)    
#     })
#     # pivot longer
#     .melt(
#         id_vars='trail',
#         value_vars=['l_opt_minus', 'l_opt', 'l_opt_plus', 'l_c', 'l_c_plus'],
#         var_name='strategy',
#         value_name='wealth'
#     )
#     # generate wealth
#     .assign(
#         cumm_wealth = lambda x: x.groupby('strategy').\
#             transform(lambda x: np.cumprod(x+1))['wealth']
#     )
#     # concat matchweek '0' with wealth = 1 to dataframe
#     .groupby('strategy')
#     .apply(
#         lambda x: x.append({'strategy': x.name, 'cumm_wealth': 1, 'trail': 0}, ignore_index=True)
#     )
#     .apply(lambda x: x.reset_index(drop=True))
#     .sort_values(['strategy', 'trail'])
#     .reset_index(drop=True)
# )

# # init plot
# fig, ax = plt.subplots()

# # plot lines
# df_sim_kelly.set_index('trail', inplace=True)
# df_sim_kelly.groupby('strategy')['cumm_wealth'].plot(color='gray', ax=ax)

# # make space 4 labels
# MOVEMENT = 100
# left, right = plt.xlim()
# plt.xlim((left, right + MOVEMENT))
    
# # labels
# last_sim = df_sim_kelly.groupby('strategy').tail(1)
# ss = ['$l_{c}$', '$l_{c}^{+}$', '$l_{*}$', '$l_{*}^{-}$', '$l_{*}^{+}$']
# n_plot = len(ss)
# txt_ss = [plt.text(N + MOVEMENT, last_sim.iloc[i, 2], ss[i], fontsize=21) for i in range(n_plot)]

# adjust_text(
#     txt_ss,
#     only_move={'points':'y', 'text':'y'}, 
#     # autoalign='xy',
#     expand_objects=(20, 40),
#     force_points=1e+2,
#     force_objects=1e+2,
#     force_text=1e+1,
#     lim=100
# )

# # show 0 threshold
# plt.axhline(y=1, linewidth=0.9, color='gray', linestyle='-.')

# # pretty graph
# plt.ylabel('$W_{n}$', fontsize=20)
# plt.xlabel('Simulación', fontsize=20)
# plt.title('Riqueza por Estrategia', fontsize=30)
# ax.tick_params(axis='both', which='major', labelsize=50)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
# plt.yscale('log')

# # reescale plot
# fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])
# ax.tick_params(axis='both', which='major', labelsize=20)    

# plt.show()



# %% 1. get database

# main
df_all = get_df()
df_work = df_all.query("season == 20 & matchweek >= 18")

#%% get dicts
# best possible outcome
df_work_best = (
    # get expected value for each event
    df_work.assign(
        rho = lambda x: (x['odds'] * x['probas'] - 1)  
    )
    # sort games in ascending order with respected the expected gain
    .sort_values( 
        by=['id_game', 'rho'], ascending=[True, False]
    )
    # get the best event by game
    .groupby(
        'id_game'
    )
    .first(
        
    )
    # reset index
    .reset_index(
    
    )
)

#  2 generate dictionaries of games and events
# get dictionaries
dict_games = get_games_dict(df_work)
dict_best_games = get_games_dict(df_work_best)

#%%  3. Optimization 
# incomplete kelly
# kelly_incomplete = portfolios_optimization(
#     tournment=dict_best_games,
#     df=df_work,
#     stake_model='kelly',
#     fractional=1
# )

# kelly_incomplete_50 = portfolios_optimization(
#     tournment=dict_best_games,
#     df=df_work,
#     stake_model='kelly',
#     fractional=0.5
# )

# complete kelly
# kelly_general = portfolios_optimization(
#     tournment=dict_games,
#     df=df_work,
#     stake_model='kelly',
#     fractional=1
# )

# incomplete markowitz
# mktz_incomplete = portfolios_optimization(
#     tournment=dict_best_games,
#     df=df_work,
#     stake_model='markowitz',
#     fractional=1
# )

# mktz_incomplete_50 = portfolios_optimization(
#     tournment=dict_best_games,
#     df=df_work,
#     stake_model='markowitz',
#     fractional=0.5
# )

# complete markowitz
mktz_general = portfolios_optimization(
    tournment=dict_games,
    df=df_work,
    stake_model='markowitz',
    fractional=1
)


# %% 4 Evaluate models
eval_kelly_general = model_summary(kelly_general)
eval_kelly_incomplete = model_summary(kelly_incomplete)
eval_mktz_incomplete = model_summary(mktz_incomplete)


# %% 5 Look performance
show_performance(
    model=kelly_incomplete, 
    model_summary=eval_kelly_incomplete,
    model_name='Kelly Restringido'
    # perform='ind_wealth'
)
show_performance(
    model=mktz_incomplete, 
    model_summary=eval_mktz_incomplete,
    model_name='Markowitz Restringido'
    # perform='ind_wealth'
)


# %% 6 fractional 
lista = [1, 0.75, 0.5, 0.25, 0.10]
fractional_models(
    tournment=dict_best_games, 
    df=df_work,
    type_model='kelly',
    fractions=lista
)
fractional_models(
    tournment=dict_best_games, 
    df=df_work,
    type_model='markowitz',
    fractions=lista
)

# %% 7 comparar modelos
list_models_metrics = [
    eval_kelly_incomplete, eval_mktz_incomplete   
]

df_models_metrics = (
    # instantiate
    pd.DataFrame(
    data=list_models_metrics,
    columns=list(list_models_metrics[0])    
    )
    # arrange
    .sort_values('final_wealth', ascending=False)
)


# %% 8 simulate bets
# incomplete
bets_sim = bets_simulation(
    tournment=dict_games,
    n_sim=100, 
    kforce=1,
    fractional=1,
    seed=10
)



# %% 9 grid of simulated strategies
N = 10
fract = np.linspace(0.01, 1, N)
k_alpha = np.power(10, np.linspace(-3, 1.5, N))

x, y = np.meshgrid(fract, k_alpha)
Mmedian = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Mmedian[i, j] = median_simulation_wealth(
            fractional=x[i, j],
            kforce=y[i, j],
            tournment=dict_games,
            n_sim=20,
            seed=10            
            )



# %% 9.1 plot 
# plot it
fig, ax = plt.subplots()
cf = ax.contourf(x, y, Mmedian, cmap='gray')

# add color bar
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Riqueza Final $W_{n}$", size=20)
cbar.ax.tick_params(labelsize=20)

# pretty graph
ax.set_yscale('log')
plt.ylabel('Diversificación ($\\alpha$)', fontsize=20)
plt.xlabel('Fracción de Riqueza Apostada', fontsize=20)
plt.title('Estrategias a priori', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)

# reescale plot
fig.tight_layout(rect=[0, 0.03, 1*2, 0.95*2])

plt.show()

# %% 10 regresion analisis
bets_sim_smmry =(
    bets_sim
    .groupby('num_sim', as_index=False)
    .agg({'cumm_wealth': lambda x: np.log(x.tail(1)), 'log_growth': 'mean'})
)

bets_sim_smmry[['cumm_wealth', 'log_growth']] = StandardScaler().\
    fit_transform(bets_sim_smmry[['cumm_wealth', 'log_growth']])

# plot scatter & lm fit
sns.lmplot(x='log_growth', y='cumm_wealth', data=bets_sim_smmry)


# %% 10.1 lm analisis
X = sm.add_constant(bets_sim_smmry['log_growth'])
Y = bets_sim_smmry['cumm_wealth']
lm = sm.OLS(Y, X)
lm = lm.fit()
print(lm.summary())












