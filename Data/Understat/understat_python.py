# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 07:46:38 2021

@author: Roman
"""
#  %% import modules
import asyncio
import json
import pandas as pd 

import aiohttp
import nest_asyncio


from understat import Understat

# to use asyncio in spdyer
nest_asyncio.apply()
# %% league final tables
async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        table = await understat.get_league_table("EPL", "2019")
        return(table)

loop = asyncio.get_event_loop()
aux_list = loop.run_until_complete(main())

# build dataframe
df_season2019 = pd.DataFrame(aux_list)
# %% league match results
# get each match 
# data in nested dictionaries

async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        fixtures = await understat.get_league_results(
            "epl",
            2018,
            {
                "a": {"id": "89",
                    "title": "Manchester United",
                    "short_title": "MUN"},
                "datetime": "2018-09-15 16:30:00"
            }
        )
        print(json.dumps(fixtures, indent=2)) # pretty printing

# retraive data
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

# build dataframe
# df_matches2019 = pd.DataFrame(aux_list)

#%% league general stats per month/year/etc
async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        stats = await understat.get_stats({"league": "EPL", "year": "2019"})
        print(json.dumps(stats, indent=2))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())


# %% team results by 
async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        results = await understat.get_team_results(
            "Manchester United",
            2018,
            side="h"
        )
        print(json.dumps(results, indent=2))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())


# %% team high stats
async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        team_stats = await understat.get_team_stats("Manchester United", 2018)
        print(json.dumps(team_stats, indent=2))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())



#%% teams general stats 
async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        teams = await understat.get_teams(
            "epl",
            2018,
        )
        return(teams)

loop = asyncio.get_event_loop()
list_teams2019 = loop.run_until_complete(main())



