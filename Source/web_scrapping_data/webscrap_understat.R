#######################################
#   Script to download data from
#   from understat with the understatr
#   package


# @Roman AVJ      9 March 2021
#######################################
library(tidyverse)
library(understatr)

#### functions ####
fetch_write_season <- function(season=2020, path){
  # get data
  df <- get_league_teams_stats('EPL', season)
  
  # write data
  file_name <- paste0(path, 'understat_',season, '.csv')
  write_csv(df, file = file_name)
  
  return(df)
  
}



#### main ####
# seasons
seasons <- 2016:2020
path <- "Data/Understat/"

# read data and write it
df_all <- data.frame(NULL)

for (s in seasons){
  cat("Fetching data from", s)
  df_all <- rbind(df_all, fetch_write_season(s, path))
  
  sleeptime <- rlnorm(n = 1, sdlog = 2) + 3
  cat("\t... sleeping")
  Sys.sleep(sleeptime)
}


# write it all
file_name <- paste0(path, 'understat_all.csv')
write_csv(df_all, file = file_name)
























