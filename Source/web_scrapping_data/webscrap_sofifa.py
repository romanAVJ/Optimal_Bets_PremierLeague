# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:11:54 2021

@author: Ryo

Web scrap SOFIFA, by specific FIFA
"""
# %% modules
import pandas as pd
import numpy as np
import requests
import re
import time
import os 
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# =============================================================================
# %% FUNCTIONS
# =============================================================================
def get_html(url, parser='html.parser'):
    # fetch data from url
    page = requests.get(url)
    
    # set it to a BeautifulSoup object
    soup = BeautifulSoup(page.content, parser)
    
    return(soup)

def get_fifa(title):
    # brute force
    fifa = title[6:13].replace(" ", "")
    
    return(fifa)

def get_date(title):
    # brute force
    date_raw = title[14:].replace("SoFIFA", "").strip()
    
    # date format
    date = datetime.strptime(date_raw, '%b %d, %Y').date()
    
    return(date)

def print_elements_tag(results):
    for res in results:
        print(res.prettify())
    return()

def print_elements_string(results):
    for res in results:
        print(res.get_text(), sep="\n")
    return()

def _tidy_string(tag):
     # get text, eliminate \t,\n, etc
    s = tag.get_text().strip()
    
    # coerce to string
    s = str(s)
    
    # to lower case and replace " " with "_"
    s = s.replace(" ", "_").lower()
    
    return(s)

def get_elements_string(results):
    # fetch elements from results and tidy them
    lis = [_tidy_string(res) for res in results]
    
    return(lis)

def get_name_team(team):
    # tag tag
    name_tag = team.find(class_ = "bp3-text-overflow-ellipsis")
    name = name_tag.get_text().strip().replace(" ", "_").lower()
    
    return(name)

def tidy_date(date_raw):
    # date format
    return(datetime.strptime(date_raw, '%b %d, %Y').date())

def get_elements_date(results):
    # fetch elements and get date format
    lis = [tidy_date(res.get_text()) for res in results]
    
    return(lis)

def get_href(results):
    # fetch href
    lis = [res.get('href', default='no_url') for res in results]
    
    return(lis)

def _appendSoFIFA(x):
    return('https://sofifa.com' + x)

def get_urls(results, tage_type='a'):
    fifa_tags = results.find_all(tage_type)
    
    fifa_names = get_elements_string(fifa_tags)
    
    fifa_urls = get_href(fifa_tags)
    fifa_urls = map(_appendSoFIFA, fifa_urls)
    
    dict_urls = dict(zip(fifa_names, fifa_urls))
    
    return(dict_urls)
    
def get_urls_week(results, tage_type='a'):
    fifa_tags = results.find_all(tage_type)
    
    fifa_week = get_elements_date(fifa_tags)
    
    fifa_urls = get_href(fifa_tags)
    fifa_urls = map(_appendSoFIFA, fifa_urls)
    
    dict_urls = dict(zip(fifa_week, fifa_urls))
    
    return(dict_urls)

def get_table(soup):
    return(soup.find(class_ = "table table-hover persist-area"))

def get_tableheader(table):
    # get headers <thead>
    table_headers_tag = table.find("thead")
    
    # get tags with <a></a>
    results = table_headers_tag.find_all("a")
    
    # get header
    header = get_elements_string(results)
    
    return(header)

def get_tablebody(table):
    # find tags with <tr>
    return(table.find("tbody"))

def get_teams_tags(table_body, which='all'):
    if which == 'all':
        results = table_body.find_all('tr')
    else:
        results = table_body.find("tr")
    return(results)    

def get_team_stats(team, header): 
    # get tag <td>
    stats_tag = team.find_all("td", attrs={"data-col": True})
    
    # get stats
    stats = get_elements_string(stats_tag)
    
    # merge to dict
    team_stats = dict(zip(header, stats))
    
    return(team_stats)

def getall_team_stats(teams, header):
    dict_teams = {}
    i = 1
    
    for team in teams:
        name_team = get_name_team(team)
        
        team_stats = get_team_stats(team, header)
        
        dict_teams[name_team] = team_stats
        # print("Fetching stats of: ", name_team, ". \n\t Team: ", i, sep='')
        
        i += 1
        
    return(dict_teams)

def stats2df(team_stats, date, fifa):
    # get dataframe
    df_teams_raw = pd.DataFrame(team_stats).transpose()
    
    # pop index to column
    df_teams_raw.reset_index(level=0, inplace=True)
    
    # rename column of teams
    df_teams_raw.rename(columns={'index': 'name_team'}, inplace=True)
    
    # set date and fifa
    df_teams_raw['date'] = date
    df_teams_raw['fifa'] = fifa
    
    # set final dataframe
    df_teams = df_teams_raw.copy()
    
    return(df_teams)
    
def matchweek(soup, title):
    # get week date and fifa
    date_fifa = get_date(title)
    which_fifa = get_fifa(title)
    
    # get table and its elements
    table = get_table(soup)
    table_headers = get_tableheader(table)
    table_body = get_tablebody(table)
    
    # get teams stats of actual matchweek
    team_tags = get_teams_tags(table_body, which='all')
    team_stats = getall_team_stats(team_tags, table_headers)
    
    #get dataframe
    df_teams = stats2df(team_stats, date=date_fifa, fifa=which_fifa)
    
    return(df_teams)

def get_fifa_urls(soup):
    # get menus
    urls_tags = soup.find_all('div', attrs={'class': 'bp3-menu'})
    
    # by inspection, are the tags in position 2
    urls_fifas = urls_tags[1]
    dict_urls = get_urls(urls_fifas)
    
    return(dict_urls)

def get_fifa_weeks(soup):
    # get menus
    urls_tags = soup.find_all('div', attrs={'class': 'bp3-menu'})
    
    # by inspection, are the tags in position 3
    urls_week = urls_tags[2]
    dict_urls = get_urls_week(urls_week)
    
    return(dict_urls)

def is_atleast(key, mini=14):
    # get digits of regex
    key_digits_list = re.findall('\d+', key)
    
    # get elements and parse as integer
    key_digits = int(key_digits_list[0])
    
    # compare
    flag = key_digits >= mini
    return(flag)

def data_sofifa(urls_fifa, time_sleep=5):
    # save each fifa data in a list
    fifa_dflist = []
    
    # process for each fifa
    for fifa, fifa_url in urls_fifa.items():
        print("Fetching data from:", fifa)
    
        # fetch dummy fifa url data
        fifa_soup = get_html(fifa_url, parser='html.parser')
        
        # get all weeks' url
        urls_week = get_fifa_weeks(fifa_soup)
        
        # save data
        df_fifa = pd.DataFrame()
        
        # process for each week
        for week, week_url in urls_week.items():
            # get soup and initial values
            week_soup = get_html(week_url, parser='html.parser')
            
            week_title = week_soup.title.string
            date_fifa = get_date(week_title)
            
            print("\t\tMatchweek:", date_fifa)
            
            # fetch stats from actual matchweek
            df_matchweek = matchweek(week_soup, week_title)
            df_fifa = df_fifa.append(df_matchweek)
            
            # sleep fetching data
            time.sleep(time_sleep)
            
        # end of fetching data in actual fifa
        fifa_dflist.append(df_fifa)
        
        print("Finished\n")
        time.sleep(time_sleep + 3)
    
    # return dataframes
    return(fifa_dflist)

def data_fifa(url_fifa, time_sleep):
    # fetch dummy fifa url data
    fifa_soup = get_html(url_fifa, parser='html.parser')
    
    # get all weeks' url
    urls_week = get_fifa_weeks(fifa_soup)
    
    # save data
    df_fifa = pd.DataFrame()
    
    # process for each week
    for week, week_url in urls_week.items():
        # get soup and initial values
        week_soup = get_html(week_url, parser='html.parser')
        
        week_title = week_soup.title.string
        date_fifa = get_date(week_title)
        
        print("\t\tMatchweek:", date_fifa)
        
        # fetch stats from actual matchweek
        df_matchweek = matchweek(week_soup, week_title)
        df_fifa = df_fifa.append(df_matchweek)
        
        # sleep fetching data
        time.sleep(time_sleep + np.random.lognormal(sigma=2))
    
    # return dataframes
    print("Finished\n")
    return(df_fifa)

def tidy_dataframe(df):
    # parse values
    int_cols = ['ova', 'att', 'mid', 'def', 'dp', 'ip']
    float_cols = ['saa', 'taa']
    
    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)
    
    # get value in millions euros
    df['transfer_budget'] = (df['transfer_budget']
        .apply(lambda x: x[1:-1])
        .astype(float)
    )
    
    # reset index
    df.reset_index(drop=True, inplace=True)
    
    return(df)
    
def joindata(df_list):
    # append all dataframes
    df_r = pd.concat(df_list, axis='index', ignore_index=True)
    
    # order by fifa and date
    df_r.sort_values(by = ['fifa', 'date'], ascending=False, inplace=True)
    
    # reset index
    df_r.reset_index(drop=True, inplace=True)
    
    return(df_r)
   


# =============================================================================
# %% MAIN
# =============================================================================
# set working dir (bad practice but util)
os.chdir('C:\\Users\\Ryo\\Documents\\Estudios\\ITAM\\Tesis\\sportsAnalytics\\tesis\\Repository')

# INIT
FIFA_URLS = {
  'fifa_14': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=140052&set=true",
  'fifa_15': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=150059&set=true",
  'fifa_16': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=160058&set=true",
  'fifa_17': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=170099&set=true",
  'fifa_18': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=180084&set=true",
  'fifa_19': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=190075&set=true",
  'fifa_20': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=200061&set=true",
  'fifa_21': "https://sofifa.com/teams?type=all&lg%5B0%5D=13&showCol%5B0%5D=ti&showCol%5B1%5D=oa&showCol%5B2%5D=at&showCol%5B3%5D=md&showCol%5B4%5D=df&showCol%5B5%5D=tb&showCol%5B6%5D=bs&showCol%5B7%5D=bd&showCol%5B8%5D=bp&showCol%5B9%5D=bps&showCol%5B10%5D=cc&showCol%5B11%5D=cp&showCol%5B12%5D=cs&showCol%5B13%5D=cps&showCol%5B14%5D=da&showCol%5B15%5D=dm&showCol%5B16%5D=dw&showCol%5B17%5D=dd&showCol%5B18%5D=dp&showCol%5B19%5D=ip&showCol%5B20%5D=ps&showCol%5B21%5D=sa&showCol%5B22%5D=ta&r=210028&set=true"
}

which_fifa = input("FIFA a bajar: \n\t -->\t")
url_fifa = FIFA_URLS.get(which_fifa)

if (url_fifa == None):
    raise('No existe el FIFA solicitado. Intenta de nuevo.')

product_folder = "Data\\SoFIFA\\"
product_file_csv = product_folder + which_fifa + "SoFIFA.csv"

# fetch fifa
df_fifa_r = data_fifa(url_fifa, 2)
df_fifa = tidy_dataframe(df_fifa_r)

# write csv
Path(product_folder).mkdir(parents=True, exist_ok=True)
df_fifa.to_csv(product_file_csv, index=False)

print('All Finished and ok. Bye.')




















