# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:12:54 2021

@author: Ryo
Download and tidy https://.football-data.co.uk/england
"""
# %% modules
import pandas as pd
import time
import re

import numpy as np
from pathlib import Path
# FINAL VARS
def data_urls():
    URLS = {
        'season_20': "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        'season_19': "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
        'season_18': "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
        'season_17': "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
        'season_16': "https://www.football-data.co.uk/mmz4281/1617/E0.csv",
        'season_15': "https://www.football-data.co.uk/mmz4281/1516/E0.csv",
        'season_14': "https://www.football-data.co.uk/mmz4281/1415/E0.csv",
        'season_13': "https://www.football-data.co.uk/mmz4281/1314/E0.csv"
        }
    return(URLS)

def set_variables(yy = 20):
    variables = [
        # key results
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        
        # match statistics
        "Referee", "HS", "AS", "HST",
        "AST", "HF", "AF", "HC",
        "AC", "HY", "AY", "HR",
        "AR",
        
        # betting data
        ## before match
        "B365H", "B365D", "B365A", "BWH",
        "BWD", "BWA", "IWH", "IWD",
        "IWA", "PSH", "PSD", "PSA",
        "WHH", "WHD", "WHA", "VCH", 
        "VCD", "VCA"
        ]
    
    if yy >= 19:
        changed_vars = [
            # key results
            "Time",
            
            # betting data
            ## last odds before match
            "B365CH", "B365CD", "B365CA", "BWCH",
            "BWCD", "BWCA", "IWCH", "IWCD", 
            "IWCA", "PSCH", "PSCD", "PSCA", 
            "WHCH", "WHCD", "WHCA", "VCCH",
            "VCCD", "VCCA"
            ]
    else:
        changed_vars = [
            # betting data
            ## last odds before match
            "PSCH", "PSCD", "PSCA"
            ]
    
    # append column names
    variables.extend(changed_vars)
    
    return(variables)

def fetch_data(url, yy=20):    
    # get columns headers
    columns_names = set_variables(yy)
    
    # get data from football co uk
    df = pd.read_csv(url, usecols=columns_names)
    
    # set which season it is
    df["season"] = str(yy)
    
    return(df)
    
def write_df(df, season, path):    
    # write 
    name_file = path + "football_uk_" + str(season) + ".csv"
    df.to_csv(name_file, index=False)
    
def get_yy(key):
    # get digits of regex
    key_digits_list = re.findall('\d+', key)
    
    # get elements and parse as integer
    key_digits = int(key_digits_list[0])
    
    return(key_digits)


def get_data(urls_dict, time_sleep, path):
    # process for each fifa
    for k in urls_dict.keys():
        # get yy from key
        yy = get_yy(k)
        url = urls_dict[k]
          
        # get data
        print("Obteniendo información para la temporada: " + str(yy) + "\n\t...")
        df_season = fetch_data(url, yy=yy)
        
        # lower case
        df_season.columns = df_season.columns.str.lower()
        
        # save data
        write_df(df_season, yy, path)
        
        print("Saved data for season: " + str(yy) + "\n\n")
        time.sleep(np.sqrt(time_sleep) * np.random.lognormal())
    
    # end process
    print("Finished Fetching all. Bye bye")
    return()
        
def menu(path):
    # get initial data
    TIME_SLEEP = 3
    dict_urls = data_urls()
    
    
    # show menu
    print("Menu para descargar tablas de momios de Football-data co uk\n")
    print("""Presione: \n\t1 <--- Descargar una temporada particular 
          \n\t2 <--- Descargar todo desde el 2013 a la fecha
          \n\tOtro <--- Terminar
          """
          )
    
    option = int(input("Opción: \t --> "))
    
    if option == 1:
        # get specific subset of one 
        season_yy = str(input("Temporada a bajar (13-20): \t --> "))
        
        # subset only for season yy
        dict_season = {k:v for k, v in dict_urls.items() if get_yy(k) == int(season_yy)}
        
        # process of fetching data
        get_data(dict_season, 0, path)
    
    elif option == 2:
        print("Bajanado toda la información ... \n\n")
        # fetch all data
        get_data(dict_urls, TIME_SLEEP, path)
        
    else: 
        print("See you!")
        
    return()
    
        
# =============================================================================
# %% main
# =============================================================================
# where to save data
PATH = "..\\..\\Data\\football_uk\\"

# create folder
Path(PATH).mkdir(parents=True, exist_ok=True)

# menu
menu(PATH)
