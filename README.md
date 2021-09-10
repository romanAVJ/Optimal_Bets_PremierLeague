# Optimal Bet for English Premier League: B.S. Thesis, ITAM

Optimal betting algorithm to set the optimal stake amount to maximize profit
subject to minimum risk.

The algorithm is divided in two parts:

1. With historical data from teams and books, get the personal probabilities
of each match of being Home win, Away win or Draw using Neural Networks.

2. With own odds, compare with books and use Kelly Criterion to optimize the stake amount.

**Bachelors' thesis to obtain the title of Applied Mathematician.**

# Building Status :smile:

Developing part 1. :soon:

- [x] Web scrap data
- [X] Tidy data
- [X] Final data set
- [X] EDA
- [ ] Model NN
- [ ] Kelly Model

# Tech Framework :snake:
For data web scrap, Neural Networks I used `python`. For data viz, data wrangling, I used `R`.

## Python packages
Se usaron los Paquetes `AIOHTTP`, `json`, `understat`, `tensorflow`, `beautifulsoup4`, `sklearn`, `numpy`, `pandas`


Para instalar los paquetes usar los siguientes comandos en terminal de python

```
# get wrapper of Understat
pip install understat

# install AUOHTTP to retrive data in http
pip install aiohttp[speedups]

# [optional] If using spyder4 or Jupyter Notebooks, use this library
# to use aiohttp
pip install nest-asyncio
```

## R libraries


# Credits

## Data :shipit:

The data used was obtained from:
- [Understat](!https://understat.com/)
- [SoFIFA](!https://sofifa.com/teams)
- [Football-Data.co.uk](!https://www.football-data.co.uk/)

## Acknowledge :bowtie:
I personally thank _Dr. Manuel Lecuanda_ for guide me in my thesis oddysey.
As well, special thanks to
- Dr. Edgar Possani
- Dr. Edgar Román
- Sofía de la Mora
- Fernando Stein
- Jerónimo Pineda
- Rodrigo Hernández
- M.S. Robin Jakobsson
- Luz María Rodriguez Becerril
- Sebastián Calderón
- Dr. Alfredo Garbuño
- Jesús David Martinez
- Emiliano Zambrano
- Pablo Landeros
- Bernardo
- Anahí Moctezuma
- Hennessy Nicole Zaragoza
- Irma Itzel Gaviño

And everyone who touch my life in college.
