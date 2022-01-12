# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:43:40 2021

@author: Ryo

Auxiliar script: look if kelly criterion for a binary outcome re-expressed in 
terms of two probabilities, can be equivalent than the original Kelly Criterion
"""
import numpy as np

import scipy.optimize as optimize
from matplotlib import pyplot as plt


# =============================================================================
# %% functions
# =============================================================================
def log_growth(bets, game):
    log_return = np.log(1 + game['odds'] * bets - np.sum(bets))    
    g_bet = np.dot(game['probas'], log_return)  
    
    return(g_bet)

def log_growth_der(bets, game):
   
    # calculate gradient
    expected_cost = game['probas'] / (1 + game['odds'] * bets - np.sum(bets)) # element wise division
    expected_win_return = (game['odds'] - 1) * expected_cost
    bets_cost = np.sum(expected_cost) - expected_cost
    
    g_bet_der = expected_win_return - bets_cost
    return(g_bet_der)
    
    
    

# =============================================================================
# %% init params
# =============================================================================
aux_game = {
    'events': ['e1', 'e2'],
    'odds': np.array([7/2, 7/6]),
    'probas': np.array([1/2, 1/2])
}
aux_game2 = {
    'events': ['e1', 'e2'],
    'odds': np.array([5, 5/4]),
    'probas': np.array([1/4, 3/4])
}
aux_game3 = {
    'events': ['e1', 'e2'],
    'odds': np.array([7/2, 7/4]),
    'probas': np.array([1/3, 2/3])
}

N = 100
linspace = np.linspace(0, 1, N)
x, y = np.meshgrid(linspace, linspace)

print(
      f"""Los expected returns por outcome están dados por\n: 
          {aux_game['odds' ]* aux_game['probas']}
      """
)

# %% example
g_l = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        bet_ij = np.array([x[i, j], y[i, j]])
        g_l[i, j] = log_growth(bet_ij, aux_game3)


# %% plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
cf = ax.contourf(x, y, np.where(g_l <= 0, 0, g_l), cmap='gray')


fig.colorbar(cf, ax=ax)
plt.show()

#  plot 2
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, np.where(g_l <= 0, 0, g_l), 30, cmap = 'binary')
# ax.set_zlimit(0, 0.5)
# ax.plot_wireframe(x, y, g_l,  color='black')

# ax.view_init(azim=270)


# %% get max
index_max = np.unravel_index(g_l.argmax(), g_l.shape)
x_max = x[index_max]
y_max = y[index_max]

print(f'Las apuestas óptimas son: \nb1 = {x_max} y b2 = {y_max}')



#%% understand
# get maximum
gmax = g_l.max()

# look how many are at max
bool_index_max_gl = g_l >= gmax
index_max_gl = np.nonzero(bool_index_max_gl)

g_sub_max = g_l[g_l >= gmax]



# %% original kelly criterion
bet_opt1 = (aux_game['probas'][0] * aux_game['odds'][0] - 1) / (aux_game['odds'][0] - 1)
bet_opt2 = 0
bet_kelly = np.array([bet_opt1, bet_opt2])

g_kelly = log_growth(bet_kelly, aux_game)


print(f"Criterio de Kelly: {g_kelly} \n Grid Search: {gmax}")



# %% get kelly criterion via optimization
# init value 
l0 = np.zeros(2)

# constraints
ineq_cons_budget = {
    'type': 'ineq',
    'fun': lambda x: 1 - np.sum(x),
    'jac': lambda x: - np.ones(x.shape)
    }

# bounds 
bounds_gl = optimize.Bounds(np.zeros(2), np.ones(2))

# optimize
def neg_log_growth(bets, game):
    return(- log_growth(bets, game))

def neg_log_growth_der(bets, game):
    return(- log_growth_der(bets, game))

res = optimize.minimize(
    fun=neg_log_growth,
    x0=l0,
    args=aux_game,
    method='SLSQP', #sequential least squares quadratic programming
    jac=neg_log_growth_der,
    bounds=bounds_gl,
    constraints=ineq_cons_budget,
    options={'ftol': np.finfo(float).eps*2, 'disp': True}    
)



# %% compare
print(res)
print(res['x'])







