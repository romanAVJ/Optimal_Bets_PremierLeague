#####################################################
# Understats, data wrange to create final dataframe

# 06/07/2021
# Roman Alberto Velez Jimenez
#####################################################

library(tidyverse)
library(tidyr)

# functions ---------------------------------------------------------------
read_file_x <- function(file_name, dir_name="Data/"){
  # append rute 
  filename <- paste0(dir_name, file_name)
  
  # read
  df_r <- read_csv(filename)
  
  # return
  return(df_r)
}

read_dfs <- function(file_names, dir_name){
  # get all 
  df_ans <- do.call("rbind", lapply(file_names, read_file_x, dir_name=dir_name))
  
  # get matchweek
  df_ans <- df_ans %>% 
    group_by(team_id, year) %>% 
    mutate(
      matchweek = row_number()
    ) %>% 
    ungroup()
  
  
  return(df_ans)
}

get_tablepositions <- function(df){
  df_ans <- df %>%
    group_by(
      # position table for each season
      year, team_name
    ) %>%
    transmute(
      # select important vars
      year, matchweek, team_name, 
      
      # select important vars to compute aggregates
      scored, missed, pts,
      
      # running statistics by season 
      total_pts = cumsum(pts),
      diff_goals = cumsum(scored - missed),
      total_goals = cumsum(scored)
    ) %>% 
    ungroup(
      
    ) %>% 
    arrange(
      year, matchweek, 
      # premier league orders by: points, goal difference, total goals
      desc(total_pts), desc(diff_goals), desc(total_goals)
    ) %>% 
    group_by(
      year, matchweek
    ) %>% 
    mutate(
      position_table = dense_rank(
        desc(interaction(total_pts, diff_goals, total_goals, lex.order = TRUE))
      )
    )
  
  return(df_ans)
}

# ponderated mean and variance
exp_decay <- function(t, xi){
  return(exp(-xi*t))
}

weighted_mean <- function(x, t, decay=1e-3){
  # get length
  size_n <- length(t)
  
  # get decay weights
  w <- exp_decay(t, xi=decay)
  sumw <- cumsum(w)
  
  # get mean
  meanw_vec <- NULL
  for(k in 1:size_n){
    # get normalized factors
    lambdas <- w[1:k] / sumw[k]
    xk <- x[1:k]
    
    # compute mean
    meanw <- sum(lambdas*xk)
    meanw_vec <- c(meanw_vec, meanw)
  }
  
  
  return(meanw_vec)
}

weighted_var <- function(x, t, decay=1e-3){
  # get length
  size_n <- length(t)
  
  # get decay weights
  w <- exp_decay(t, xi=decay)
  sumw <- cumsum(w)
  
  
  # get cumm var
  varw_vec <- NULL
  for(k in 1:size_n){
    # get normalized factors
    lambdas <- w[1:k] / sumw[k]
    
    # get mean
    xk <- x[1:k]
    meanw <- sum(lambdas*xk)
    
    # compute var
    unbias_constant <- 1/(1-sum(lambdas^(2)))
    sqrd_diff <- (xk-meanw)^(2)
    varw <- unbias_constant * sum(lambdas*sqrd_diff)
    
    varw_vec <- c(varw_vec, varw)
  }
  
  return(varw_vec)
}

get_ponderatedvars <- function(df, decay=0.1, units_time="weeks"){
  # generate base dataframe
  df_ans <- df_r %>% 
    group_by(
      team_name
    ) %>% 
    mutate(
      # get t - t0
      last_game = max(date),
      diff_time = difftime(last_game, date, units = units_time) %>% as.numeric(),
      
      # compute mean and var
      dinamic_mean = weighted_mean(x = npxGD, t=diff_time, decay=decay),
      dinamic_var = weighted_var(x = npxGD, t=diff_time, decay=decay)
    ) %>% 
    ungroup(
      
    ) %>% 
    arrange(
      team_name, year
    ) %>% 
    select(
      date, year, matchweek, team_name, dinamic_mean, dinamic_var
    )
  
  return(df_ans)
  
}

get_multiplemeans <- function(df, units_time="weeks"){
  # generate base dataframe
  df_ans <- df %>% 
    group_by(
      team_name
    ) %>% 
    mutate(
      # get t - t0
      last_game = max(date),
      diff_time = difftime(last_game, date, units = units_time) %>% as.numeric(),
      
      # compute mean and var
      wmean_inf = weighted_mean(x = npxGD, t=diff_time, decay=2.11),
      wmean_05 = weighted_mean(x = npxGD, t=diff_time, decay=0.5),
      wmean_01 = weighted_mean(x = npxGD, t=diff_time, decay=0.1),
      wmean_005 = weighted_mean(x = npxGD, t=diff_time, decay=0.05),
      wmean_0 = weighted_mean(x = npxGD, t=diff_time, decay=1e-10)
    ) %>% 
    ungroup(
      
    ) %>% 
    arrange(
      team_name, year
    ) %>% 
    select(
      date, team_name, npxGD, wmean_inf:wmean_0
    ) %>% 
    pivot_longer(
      cols = wmean_inf:wmean_0,
      names_to = "xi",
      va|lues_to = "mean_value",
    ) 
  
  return(df_ans)
}

# get new teams at premier league
get_newteams <- function(df){
  # get unique names per season
  list_names <- df %>% 
    split(.$year) %>% 
    map(~ unique(.$team_name))
  
  n_len <- length(list_names)
  name_seasons <- names(list_names)
  df_ans <- tibble(
    season = 2014,
    team1 = "Leicester",
    team2 = "Burnley",
    team3 = "Queens Park Rangers"
  )
  
  # get teams that weren't previous season
  for(i in 2:n_len){
    indi <- list_names[[i]] %in% list_names[[i-1]]
    teams <- list_names[[i]][!indi]
    df_ans <- rbind(df_ans, c(name_seasons[i], teams))
  }
  
  return(df_ans)
  
  
}

# final dataframe
get_tidy <- function(positions, goals, new_teams){
  df_ans <- positions %>% 
    inner_join(
      goals,
      by = c("year", "matchweek", "team_name")
    ) %>% 
    select(
      # in common
      year, matchweek, date, team_name,
      
      # table
      position_table, total_pts,
      
      # npxGD
      date, dinamic_mean, dinamic_var
      
    ) %>% 
    group_by(
      team_name
    ) %>% 
    mutate(
      dinamic_var = if_else(is.na(dinamic_var), 0, dinamic_var)
    ) %>% 
    rename(
      npxGD_ma = dinamic_mean,
      npxGD_var = dinamic_var
    ) %>% 
    ungroup(
      
    ) %>% 
    mutate(
      season = year - 2000,
      big_six = if_else(
        team_name %in% c("Arsenal", "Chelsea", "Liverpool",
                                         "Manchester City", "Manchester United", "Tottenham"),
        true = 1,
        false = 0
      )
    )
  # look if in that season were new teams
  new_teams.longer <- new_teams %>% 
    pivot_longer(
      cols = team1:team3,
      names_to = "type",
      values_to = "team_name"
    ) %>% 
    select(-type) %>% 
    mutate(
      promoted_team = 1,
      season = as.double(season) - 2000
    )
  
  df_ans <- df_ans %>% 
    left_join(
      new_teams.longer,
      by = c("season", "team_name")
    ) %>% 
    mutate(
      promoted_team = if_else(is.na(promoted_team), 0, 1)
    )
  
  
  
  return(df_ans)
  
}

# main --------------------------------------------------------------------
#### read data ####
DATA_UNDERSTATS <- "Data/Understat/"
files_names <- list.files(DATA_UNDERSTATS, pattern = ".csv$")

df_r <- read_dfs(files_names, DATA_UNDERSTATS)


#### generate tables ####
# positions
df_points <- get_tablepositions(df_r)

# new teams in EPL by season
df_newteams <- get_newteams(df_r)

# pondarated vars
aux_mainxi <- get_ponderatedvars(df_r, decay=0.1, units_time = "week")

df_means <- get_multiplemeans(df_r, units_time = "weeks")

# final dataframe
df_final <- get_tidy(df_points, aux_mainxi, df_newteams)

#### write tables ####
write_csv(df_r, file = "Data/Understat/final_dbb/understats_raw.csv")
write_csv(df_means, file = "Data/Understat/final_dbb/means_xi.csv")
write_csv(df_final, file = "Data/Understat/final_dbb/understat_finaldbb.csv")
