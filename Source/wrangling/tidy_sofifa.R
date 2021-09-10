#####################################################
# SOFIFA,  data wrange to create final dataframe

# 07/07/2021
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
  
  # transform data
  df_ans <- df_ans %>% 
    mutate(
      year_week = strftime(date, format = "%Y-W%V")
    )
  
  
  return(df_ans)
}

generate_teams_dates <- function(df){
  fifa_number <- df[["fifas_number"]][1]
  
  cat("\n FIFA", fifa_number)
  
  
  # 1. Generate grid with teams and weeks
  df_grid <- expand.grid(
    name_team = unique(df[["name_team"]]),
    week_date = 1:53
  ) %>%
    # 2. create year
    mutate(
      year_date = ifelse(week_date >= 32, fifa_number - 1, fifa_number)
    )
  
  # 3. full outer join
  df_ans <- df %>% 
    full_join(
      df_grid,
      by = c("name_team","week_date", "year_date")
    ) %>% 
    arrange(
      year_date, week_date
    )
  
  return(df_ans)
}

get_allmacthweeks <- function(df){
  # get unique observations by week
  df_uniqueweeks <- df %>% 
    mutate(
      year_date = lubridate::isoyear(date) - 2000,
      week_date = lubridate::isoweek(date),
      fifas_number = str_extract(fifa, "[[:digit:]]+") %>% as.integer()
    ) %>% 
    filter(
      # there exists same weeks for distinct fifas
      ((fifas_number - year_date == 1) & (week_date >= 32)) | 
        ((fifas_number - year_date == 0) & (week_date < 32))
    ) %>% 
    arrange(
      # if there are one or more obs in a week, get last obs  
      desc(fifa), desc(year_week), desc(date)
    ) %>% 
    distinct(
      # get unique obs
      name_team, fifa, year_week, .keep_all = TRUE
    )
  
  # split and apply function for each fifa to get all weeks of fifa #
  df_ans <- df_uniqueweeks %>%
    group_by(fifa) %>%
    # apply function by groups and append results in one dataframe
    group_modify(~ generate_teams_dates(df = .x)) %>% 
    ungroup() %>% 
    group_by(name_team) %>% 
    # fill missing values with previous obs by team
    fill(names(df) ,.direction = "downup") %>% 
    ungroup() %>%
    mutate(
      # mark if the register existed or not
      was_na = if_else(is.na(ova), 1, 0),
      season = str_extract(fifa, "[[:digit:]]+") %>% as.integer() - 1
    ) %>% 
    select(
      season, fifa, date, year_date, 
      week_date, name_team, ova, att, 
      mid, def, transfer_budget,
      dp, ip, saa, 
      taa, was_na
    )
  
  return(df_ans)
}

# main --------------------------------------------------------------------
#### read data ####
DATA_SOFIFA <- "Data/SoFIFA/"
files_names <- list.files(DATA_SOFIFA, pattern = ".csv$")

# merge all files
df_r <- read_dfs(files_names, DATA_SOFIFA)


#### generate tables ####
df_final <- get_allmacthweeks(df_r)

#### write csv ####
write_csv(df_r, "Data/SoFIFA/final_dbb/sofifa_raw.csv")
write_csv(df_final, "Data/SoFIFA/final_dbb/sofifa_finaldbb.csv")


