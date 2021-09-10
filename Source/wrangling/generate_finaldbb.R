#####################################################
# Final data set

# 24/06/2021
# Roman Alberto Velez Jimenez
#####################################################
library(tidyverse)



# functions ---------------------------------------------------------------
# utils #
normalize_backwards <- function(x){
  z <- (max(x) - x) / (max(x) - min(x))
  return(z)
}

normalize <- function(x){
  z <- (x - min(x)) / (max(x) - min(x))
  return(z)
}

normalize_manual <- function(x, maxi, mini){
  z <- (x - mini)/(maxi-mini)
  return(z)
}

scale_wall <- function(x, wall=100){
  x <- if_else(x <= wall, x, wall)
  z <- (x - mean(x))/sd(x)
  return(z)
}


# data frames #
tidy_understats <- function(df, df_teamnames){
  # set names
  df_ans <- df %>% 
    left_join(
      df_teamnames %>% select(team_understats, official_name),
      by = c("team_name" = "team_understats")
    ) %>%
    select(
      season, matchweek, official_name,
      position_table, total_pts, npxGD_ma:promoted_team
    ) %>% 
    rename(
      team = official_name
    ) %>% 
    # lag observations
    arrange(
      season, matchweek
    ) %>% 
    group_by(
      team
    ) %>% 
    mutate(
      across(position_table:npxGD_var, lag)
    ) %>% 
    ungroup(
      
    ) %>% 
    arrange(
      season, matchweek, position_table
    )
  
  return(df_ans)
}

tidy_football <- function(df, df_teamnames){
  df_ans <- df %>% 
    left_join(
      df_teamnames %>% select(team_football, official_name),
      by = c("hometeam" = "team_football")
    ) %>% 
    left_join(
      df_teamnames %>% select(team_football, official_name),
      by = c("awayteam" = "team_football"),
      suffix = c(".home", ".away")
    ) %>% 
    select(
      season, date, matchweek,
      official_name.home, official_name.away, result,
      starts_with("maxo_"), market_tracktake, psch:psca,
      starts_with("diff")
    ) %>% 
    rename(
      hometeam = official_name.home,
      awayteam = official_name.away,
    )
  
  return(df_ans)
}

tidy_sofifa <- function(df, df_teamnames){
  df_ans <- df %>% 
    left_join(
      df_teamnames %>% select(team_sofifa, official_name),
      by = c("name_team" = "team_sofifa")
    ) %>% 
    select(
      year_date, week_date, official_name, ova:taa
    ) %>%
    rename(
      team = official_name
    ) %>% 
    # lag observations
    arrange(
      year_date, week_date
    ) %>% 
    group_by(
      team
    ) %>% 
    mutate(
      across(ova:taa, lag)
    ) %>% 
    ungroup(
      
    )
  
  return(df_ans)
}

# final df's
final_df1 <- function(df_understats, df_football, df_sofifa){
  # modify data to get vars in low values
  df_understats <- df_understats %>% 
    group_by(
      matchweek
    ) %>% 
    mutate(
      position_table = normalize_backwards(position_table),
      total_pts = scale(total_pts)
    ) %>% 
    ungroup(
      
    )
  
  df_sofifa <- df_sofifa %>% 
    group_by(
      week_date
    ) %>% 
    mutate(
      across(ova:def, scale),
      transfer_budget = scale_wall(transfer_budget, wall=100),
      across(dp:ip, normalize),
      across(saa:taa, scale)
    ) %>% 
    ungroup(
      
    ) %>% 
    arrange(
      year_date, week_date, ova
    )
  
  # merge final dff
  df_main <- df_football %>% 
    inner_join(
      df_understats,
      by = c("season", "matchweek", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_understats ,
      by = c("season", "matchweek", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    mutate(
      year_date = lubridate::isoyear(date) - 2000,
      week_date = lubridate::isoweek(date) 
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    arrange(
      season, matchweek, date
    ) %>% 
    # get rid of the first matchweek
    filter(
      matchweek != 1
    ) %>% 
    mutate(
      # modify odds to probabilities
      proba_h = 1/psch,
      proba_d = 1/pscd,
      proba_a = 1/psca
    )
    
  
  return(df_main)
  
}

final_df2 <- function(df_understats, df_football, df_sofifa){
  # merge final dff
  df_main <- df_football %>% 
    inner_join(
      df_understats,
      by = c("season", "matchweek", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_understats ,
      by = c("season", "matchweek", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    mutate(
      year_date = lubridate::isoyear(date) - 2000,
      week_date = lubridate::isoweek(date) 
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    arrange(
      season, matchweek, date
    ) %>% 
    # get rid of the first matchweek
    filter(
      matchweek != 1
    ) %>% 
    mutate(
      # modify odds to probabilities
      proba_h = 1/psch,
      proba_d = 1/pscd,
      proba_a = 1/psca
    )
  
  return(df_main)
}

final_df3 <- function(df_understats, df_football, df_sofifa){
  # eps
  eps <- 2*.Machine$double.eps 
  
  # merge final dff
  df_main <- df_football %>% 
    inner_join(
      df_understats,
      by = c("season", "matchweek", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_understats ,
      by = c("season", "matchweek", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    mutate(
      year_date = lubridate::isoyear(date) - 2000,
      week_date = lubridate::isoweek(date) 
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "hometeam" = "team")
    ) %>% 
    inner_join(
      df_sofifa,
      by = c("year_date", "week_date", "awayteam" = "team"),
      suffix = c("_home", "_away")
    ) %>% 
    arrange(
      season, matchweek, date
    ) %>% 
    # get rid of the first matchweek
    filter(
      matchweek != 1
    ) %>% 
    # interactions between variables
    # all with respect the home team
    # try to make that if positive then is better for home
    # and viceversa for away
    mutate(
      # dummies
      big_six_ad = big_six_home - big_six_away,
      promoted_team_ad = promoted_team_away - promoted_team_home,
      
      # stats
      position_table_ad = position_table_away - position_table_home, # is backward for interpretation
      total_pts_ad = total_pts_home - total_pts_away,
      npxGD_ma_ad = npxGD_ma_home - npxGD_ma_away,
      npxGD_var_rd = 1 - (npxGD_var_away + eps)/(npxGD_var_home + eps), # avoid cero div
      ova_rd = 1 - ova_away/ova_home,
      att_rd = 1 - att_away/att_home,
      def_rd = 1 - def_away/def_home,
      transfer_budget_rd =  1 - transfer_budget_away/transfer_budget_home,
      ip_ad = ip_home - ip_away,
      saa_ad = saa_home - saa_away,
      # modify odds to probabilities
      proba_h = 1/psch,
      proba_d = 1/pscd,
      proba_a = 1/psca
    )
  
  return(df_main)
}

# main --------------------------------------------------------------------
#### read data ####
# read
df_understats <- read_csv("Data/Understat/final_dbb/understat_finaldbb.csv")
df_football <- read_csv("Data/football_uk/final_dbb/footballuk_finaldbb.csv")
df_sofifa <- read_csv("Data/SoFIFA/final_dbb/sofifa_finaldbb.csv")

# get same names for teams
df_teamnames <- tibble(
  team_understats = df_understats$team_name %>% unique() %>% sort(),
  team_football = df_football$hometeam %>% unique()  %>% sort(),
  team_sofifa = df_sofifa$name_team %>% unique()  %>% sort()
) %>% 
  mutate(
    official_name = team_football %>% str_to_lower() %>% str_replace(pattern = " ", replacement = "_"),
    official_name = if_else(official_name == "qpr", "queens_rangers", official_name)
  ) 


#### tidy dataframes ####
# work dfs
df_understats <- tidy_understats(df_understats, df_teamnames)
df_football <- tidy_football(df_football, df_teamnames)
df_sofifa <- tidy_sofifa(df_sofifa, df_teamnames)

# final dfs
df_main1 <- final_df1(df_understats, df_football, df_sofifa)
df_main2 <- final_df2(df_understats, df_football, df_sofifa)
df_main3 <- final_df3(df_understats, df_football, df_sofifa)

# subset dfs
## neural net variables
df_nn_vars1 <- df_main1 %>% 
  select(
    # identificators
    result, season, matchweek, 
    date, hometeam, awayteam, 
    
    # covariables
    ## understats
    position_table_home, total_pts_home:promoted_team_away,
    ## sofifa
    ### home
    ova_home:transfer_budget_home, ip_home, saa_home,
    ### away
    ova_away:transfer_budget_away, ip_away, saa_away,
    
    ## football uf
    starts_with("proba")
  ) 

df_nn_vars2 <- df_main2 %>% 
  select(
    # identificators
    result, season, matchweek, 
    date, hometeam, awayteam, 
    
    # covariables
    ## understats
    position_table_home, total_pts_home:promoted_team_away,
    ## sofifa
    ### home
    ova_home:transfer_budget_home, ip_home, saa_home,
    ### away
    ova_away:transfer_budget_away, ip_away, saa_away,
    
    ## football uf
    starts_with("proba")
  ) 

df_nn_vars3 <- df_main3 %>% 
  select(
    # identificators
    result, season, matchweek, 
    date, hometeam, awayteam, 
    
    # covariables
    ends_with("_ad"), ends_with("_ad"),
    
    # odds
    starts_with("proba")
  ) 
  

## table for the kelly stake model
df_stake_vars <- df_main1 %>% 
  select(
    # identificators
    result, season, matchweek, 
    date, hometeam, awayteam, 
    
    # odds
    starts_with("maxo_"), market_tracktake, starts_with("diff_")
  )


#### write csv ####
write_csv(df_nn_vars1, file = "Data/Main_DBB/model_myscale.csv")
write_csv(df_nn_vars2, file = "Data/Main_DBB/model_original.csv")
write_csv(df_nn_vars3, file = "Data/Main_DBB/model_interactions.csv")
write_csv(df_stake_vars, file = "Data/Main_DBB/stake_odds.csv")




















