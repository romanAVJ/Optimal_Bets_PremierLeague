# Optimal Bets for English Premier League: B.S. in Applied Mathematics Thesis, ITAM 2022

Betting algorithm to find the optimal stake amount to maximize utility subject to the risk appetite of the player -logarithmic or quadratic- given the data.

The algorithm is divided in two parts:
1. **Retraive info**. With historical data from teams and books, estimate the probabilities of each match outcome with deep learning.

2. **Modern Portfolio Theory**. With the estimated odds and the risk profile of the bettor, use Modern Portfolio Theory -Kelly Criterion or Sharpe Ratio- to optimize the stake amount.

**Bachelor thesis to obtain a Applied Mathematician degree at ITAM (Intituto Tecnológico Autónomo de México).**



# Building Status :hammer:

Finished. :white_check_mark:

- [x] Web scrap data
- [X] Tidy data
- [X] Final data set
- [X] EDA
- [X] Model NN
- [X] Stake Models

# Tech Framework :high_brightness:
For data web scrap, Neural Networks and optimization problems: `python`. For data viz, data wrangling: `R`.

## Python packages :snake:

Use python 3.7.7 <=

Used packages: `AIOHTTP`, `json`, `understat`, `tensorflow`, `beautifulsoup4`, `sklearn`, `numpy`, `pandas`.

The majority of the packages are in Google Colab, therefore you shouldn't worry about their respective versions. However, for the web scrap process, several packages should be installed. Use the following console commands in a python console environment.

```
# get wrapper for Understats
pip install understat

# install AUOHTTP to retrive data in http
pip install aiohttp[speedups]

# [optional] If using spyder4 or Jupyter Notebooks, use this library
# to use aiohttp
pip install nest-asyncio

# install webscrap main package
pip install beautifulsoup4

# install common packages
pip install numpy
pip install pandas
pip install re
```

## R libraries :bar_chart:
Use R version 4.0.0. Go to `session_info_R.txt` in the `Results` folder for the versions of each package used. The next command is required to install the necessary packages to replicate the insights and databases.

```
# install multiple packages
instal.packages(c("understatr", "tidyverse", "tidyr", "lubridate","gtools") )
```

# How to replicate it :crystal_ball:
Although the algorithm is exemplified for a logarithmic or quadratic bettor in the English Premier League, the present model can be generalized for any finite multiple and simultaneous bet. For further detail, read the thesis of this algorithm.

## 1. Download Data
All scripts used in this section are located in the folder `Source/web_scrapping_data/`.

i. First, you should run `webscrap_sofifa.py` or `webscrap_sofifaAll.py`. The difference between the scripts is that the latter downloads from season 2013/2014 up to 2020/2021 from the EA Sports FIFA' in a single command. The only issue doing this that SoFIFA sometimes cancels the connection and all the progress is lost. That's why `webscrap_sofifa.py` exists, because this script downloads one by one (from 13 to 21).

ii. Next, you should run `webdownload_footballuk.py`.

iii. Finally, run `webscrap_understat.R`

**Note:** For both scripts you have to change the `os.chdir()`/`setwd()` (in python/ in R)  of your working folder to save the  web scrapped databases in a folders named `Data/SoFIFA/`, `Data/football_uk/`  or `Data/Understat/`, respectively.

## 2. Generate main database
All scripts used in this section are located in the folder `Source/wrangling`.

i. Run `tidy_footballuk.R` `tidy_sofifa.R` and `tidy_undertats`.

ii. At last, run `generate_finaldbb.R`

**Note:** Set the working directory `setwd()` in the general project folder -the root folder of this repo- in order to the scripts work flawless.


## 3. Exploratory Data Analysis
All scripts used in this section are located in the folder `Source/eda/`.

In order to explore the three main databases -SoFIFA, Undertats, Football Data UK- you must run all the scripts in the latter folder.

It is important to previously run part 1 and part 2, also that the working directory is in the root of the project folder, and finally that the databases written are named exactly as they were named in the originals scripts.

## 4. Get signal from noise with deep learning
All scripts used in this section are located in the folder `Source/statistical_learning_model/`.

The notebook to be consulted is `deeplearning_epl_models.ipynb`, where all the functions and results are made. This notebook was made in Google Colab, then in order to replicate the results you _should_ use Google Colab to run this notebook.

The rest of the scripts are auxiliary scripts that helped to build the latter code. They aren't required for it to work.

## 5. Build Optimal Portfolios
All scripts used in this section are located in the folder `Source/financial_model/`.

For last, in order to create the optimum portfolios for a quadratic bettor and logarithmic bettor you must consult `Financial_Models.ipynb`. Like the Deep Learning Model, in order to replicate the results use Google Colab.


# Credits:tm:
## Data :shipit:

The data used was obtained from:
- [Understat](!https://understat.com/)
- [SoFIFA](!https://sofifa.com/teams)
- [Football-Data.co.uk](!https://www.football-data.co.uk/)

## Acknowledge :bowtie:
I personally thank _Dr. Manuel Lecuanda_ for guide me in my thesis odyssey. As well to
- Dr. Edgar Possani
- Dr. Edgar Román
- Sofía de la Mora
- Hennessy Nicole Zaragoza
- Fernando Stein
- Jerónimo Pineda
- Santiago Muriel
- Rodrigo Hernández
- Anahí Moctezuma
- M.S. Robin Jakobsson
- Luz María Rodriguez Becerril
- Sebastián Calderón
- Dr. Alfredo Garbuno
- Dr. Jorge de la Vega
- Dr. Fernando Pérez
- Jesús David Martinez
- Emiliano Zambrano
- Pablo Landeros
- Bernardo Madrigal
- Irma Itzel Gaviño
- Dr. Thomas kirschenmann
