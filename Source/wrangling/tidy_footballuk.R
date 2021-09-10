#####################################################
# Football UK, data wrange to create final dataframe

# 06/07/2021
# Roman Alberto Velez Jimenez
#####################################################


# functions ---------------------------------------------------------------
read_file_x <- function(file_name, dir_name="Data/"){
  # append rute 
  filename <- paste0(dir_name, file_name)
  
  # read
  df_ans <- read_csv(filename)
  
  # get only same headers
  COL_NAMES <- c(
    "date", "hometeam", "awayteam", "fthg",
    "ftag", "ftr", "referee", "b365h",
    "b365d", "b365a", "bwh", "bwd", "bwa",
    "iwh", "iwd", "iwa", "psh", "psd", "psa",
    "whh", "whd", "wha", "vch", "vcd", "vca",
    "psch", "pscd", "psca", "season"
  )
  
  # select only these columns
  df_ans <- df_ans %>% select(one_of(COL_NAMES))
  
  # return
  return(df_ans)
}

read_dfs <- function(file_names, dir_name){
  df_ans <- do.call("rbind", lapply(file_names, read_file_x, dir_name=dir_name))
  
  return(df_ans)
}

# get odds
getodds_indiv <- function(df, type_odd="h"){
  # cols
  cols_odds <- c(
    "b365", "bw", "iw", "ps",
    "wh", "vc"
  )
  
  # concat type of odd
  cols_odds <- str_c(cols_odds, type_odd)
  
  
  # look spread by match and season
  df_ans <- df %>% 
    select(
      # unselect closing odds
      -psch, -pscd, -psca
    ) %>% 
    mutate(
      match_name = paste(hometeam, awayteam, sep = "-")
    ) %>% 
    # get all odds and id's by type of odd
    select(
      match_name, season, ends_with(type_odd)
    ) %>% 
    pivot_longer(
      cols = all_of(cols_odds),
      names_to = "books",
      values_to = "odds"
    ) %>% 
    group_by(
      match_name, season
    ) %>% 
    summarise(
      max_odds = max(odds, na.rm = TRUE),
      spread_odds = max(odds, na.rm = TRUE) - min(odds, na.rm = TRUE),
      mean_odds = mean(odds, na.rm = TRUE),
      stdev = sd(odds, na.rm = TRUE)
    )
  
  return(df_ans)
}

get_odds <- function(df){
  # get home, draw, away odds
  df_odds.home <- getodds_indiv(df_r, type_odd = "h")
  df_odds.draw <- getodds_indiv(df_r, type_odd = "d")
  df_odds.away <- getodds_indiv(df_r, type_odd = "a")
  
  # merge
  df_ans <- df_odds.home %>% 
    left_join(
      df_odds.draw,
      by = c("match_name", "season"),
      suffix = c(".home", ".draw")
    ) %>% 
    left_join(
      df_odds.away,
      by = c("match_name", "season")
    ) %>%
    rename(
      max_odds.away = max_odds,
      spread_odds.away = spread_odds,
      mean_odds.away = mean_odds,
      stdev.away = stdev
    ) %>% 
    ungroup(
      
    ) %>% 
    group_by(
      match_name, season
    ) %>% 
    mutate(
      best_spread = max(spread_odds.home, spread_odds.draw, spread_odds.away),
      market_tracktake =  (1/max_odds.home + 1/max_odds.draw + 1/max_odds.away) - 1
    )
  
}

# get difference between pinnacle odds
get_odds.pinnacle <- function(df){
  # get base dataframe
  df_change_odds <- df %>% 
    mutate(
      match_name = paste(hometeam, awayteam, sep = "-"),
      
      season = factor(season, levels = 13:20, ordered = TRUE),
      
      apriori_tt = 1/psh + 1/psd + 1/psa - 1,
      posterior_tt = 1/psch + 1/pscd + 1/psca - 1,
      
      psh = psh * (apriori_tt + 1),
      psd = psd * (apriori_tt + 1),
      psa = psa * (apriori_tt + 1),
      
      psch = psch * (posterior_tt + 1),
      pscd = pscd * (posterior_tt + 1),
      psca = psca * (posterior_tt + 1),
      
      diff_home = psch/psh - 1,
      diff_draw = pscd/psd - 1,
      diff_away = psca/psa - 1
    )
  
  # get first, last and difference dataframes in long format
  df_update_odds.first_odds <- df_change_odds %>% 
    select(
      hometeam, awayteam, season, 
      psh, psd, psa
    ) %>% 
    pivot_longer(
      cols = psh:psa,
      names_to = "type",
      values_to = "first_odd"
    ) %>% 
    mutate(
      type = case_when(
        type == "psh" ~ "home",
        type == "psd" ~ "draw",
        type == "psa" ~ "away",
        TRUE ~ "error"
      )
    )
  
  df_update_odds.last_odds <- df_change_odds %>% 
    select(
      hometeam, awayteam, season, 
      psch, pscd, psca
    ) %>% 
    pivot_longer(
      cols = psch:psca,
      names_to = "type",
      values_to = "last_odd"
    ) %>% 
    mutate(
      type = case_when(
        type == "psch" ~ "home",
        type == "pscd" ~ "draw",
        type == "psca" ~ "away",
        TRUE ~ "error"
      )
    )
  
  df_update_odds.diffs <- df_change_odds %>% 
    select(
      hometeam, awayteam, season, 
      diff_home, diff_draw, diff_away
    ) %>% 
    pivot_longer(
      cols = diff_home:diff_away,
      names_to = "type",
      values_to = "diff"
    ) %>% 
    mutate(
      type = case_when(
        type == "diff_home" ~ "home",
        type == "diff_draw" ~ "draw",
        type == "diff_away" ~ "away",
        TRUE ~ "error"
      )
    )
  
  # merge long dataframes
  df_ans <- df_update_odds.first_odds %>% 
    left_join(
      df_update_odds.last_odds,
      by = c("hometeam", "awayteam", "season", "type")
    ) %>% 
    left_join(
      df_update_odds.diffs,
      by = c("hometeam", "awayteam", "season", "type")
    )
  
  return(df_ans)
}

# variance
get_varodds <- function(df, num_quantiles = 10){
  # normalize odds of pinnacle sports
  df_change_odds <- df_r %>% 
    mutate(
      match_name = paste(hometeam, awayteam, sep = "-"),
      
      season = factor(season, levels = 13:20, ordered = TRUE),
      
      apriori_tt = 1/psh + 1/psd + 1/psa - 1,
      posterior_tt = 1/psch + 1/pscd + 1/psca - 1,
      
      psh = psh * (apriori_tt + 1),
      psd = psd * (apriori_tt + 1),
      psa = psa * (apriori_tt + 1),
      
      psch = psch * (posterior_tt + 1),
      pscd = pscd * (posterior_tt + 1),
      psca = psca * (posterior_tt + 1),
      
      diff_home = psch/psh - 1,
      diff_draw = pscd/psd - 1,
      diff_away = psca/psa - 1
    )
  
  # discretize by deciles
  df_change_odds <- df_change_odds %>% 
    mutate(
      # get var
      psh_dscrt = gtools::quantcut(psh, q = num_quantiles),
      psd_dscrt = gtools::quantcut(psd, q = num_quantiles),
      psa_dscrt = gtools::quantcut(psa, q = num_quantiles),
      
      # transform in rank
      psh_dscrt_level = as.numeric(psh_dscrt),
      psd_dscrt_level = as.numeric(psd_dscrt),
      psa_dscrt_level = as.numeric(psa_dscrt)
    )
  
  # long format for home, draw and away variance
  df_vardiff.home <- df_change_odds %>% 
    group_by(psh_dscrt_level, psh_dscrt) %>% 
    summarise(
      home = var(diff_home, na.rm = TRUE)
    )
  
  df_vardiff.draw <- df_change_odds %>% 
    group_by(psd_dscrt_level, psd_dscrt) %>% 
    summarise(
      draw = var(diff_draw, na.rm = TRUE)
    )
  
  df_vardiff.away <- df_change_odds %>% 
    group_by(psa_dscrt_level, psa_dscrt) %>% 
    summarise(
      away = var(diff_away, na.rm = TRUE)
    )
  
  # merge
  df_ans <- df_vardiff.home %>% 
    left_join(
      df_vardiff.draw,
      by = c("psh_dscrt_level" = "psd_dscrt_level")
    ) %>% 
    left_join(
      df_vardiff.away,
      by = c("psh_dscrt_level" = "psa_dscrt_level")
    ) %>% 
    select(
      psh_dscrt_level, home,
      draw, away
    ) %>% 
    pivot_longer(
      cols = home:away,
      names_to = "type",
      values_to = "variance"
    )
  
  return(df_ans)
}

# track take of books
get_tracktake <- function(df){
  df_ans <- df %>% 
    mutate(
      # get name match
      match_name = paste(hometeam, awayteam, sep = "-"),
      
      # track take
      tt_bet365 = 1/b365h + 1/b365d + 1/b365a - 1,
      tt_bwin = 1/bwh + 1/bwd + 1/bwa - 1,
      tt_interwetten = 1/iwh + 1/iwd + 1/iwa - 1,
      tt_pinnacle = 1/psh + 1/psd + 1/psa - 1,
      tt_williamhill = 1/whh + 1/whd + 1/wha - 1,
      tt_vcbet = 1/vch + 1/vcd + 1/vca - 1
    ) %>% 
    select(
      date, season, match_name, starts_with("tt")
    ) %>% 
    pivot_longer(
      cols = tt_bet365:tt_vcbet,
      names_to = "books",
      values_to = "track_take"
    ) %>% 
    mutate(
      books = str_sub(books, start = 4L, end = -1L)
    ) 
  
  return(df_ans)
}

# final
get_tidy <- function(df_base, df_odd){
  df_ans <- df_base %>% 
    mutate(
      # change format date
      date = lubridate::parse_date_time(date, orders = c("%d/%m/%y", "%d/%m/%Y")) %>% 
        lubridate::as_date(),
      
      # type of trake takes
      apriori_tt = 1/psh + 1/psd + 1/psa - 1,
      posterior_tt = 1/psch + 1/pscd + 1/psca - 1,
      
      # normalize odds
      psh = psh * (apriori_tt + 1),
      psd = psd * (apriori_tt + 1),
      psa = psa * (apriori_tt + 1),
      
      psch = psch * (posterior_tt + 1),
      pscd = pscd * (posterior_tt + 1),
      psca = psca * (posterior_tt + 1),
      
      diff_home = psch/psh - 1,
      diff_draw = pscd/psd - 1,
      diff_away = psca/psa - 1
    ) %>% 
    select(
      # get specific variables to avoid overlaps in join
      date, hometeam, awayteam, ftr, season,
      posterior_tt, psch:psca, diff_home:diff_away
    ) %>% 
    inner_join(
      df_odd %>% 
        # separate match_name to return home and away team values
        separate(
          match_name, 
          into = c("hometeam", "awayteam"), 
          sep = "-"
        ) %>% 
        # get specific vars of the odds
        select(
          hometeam, awayteam, season,
          starts_with("max_odds"), market_tracktake
        ),
      by = c("hometeam", "awayteam", "season")
    ) %>% 
    # generate matchweek
    group_by(
      season
    ) %>% 
    mutate(
      matchweek = ((row_number() -1) %/% 10) + 1
    ) %>% 
    ungroup(
      
    ) %>% 
    select(
      season, date, matchweek,
      hometeam, awayteam,  ftr, 
      starts_with("max_odds"), market_tracktake, psch:psca,
      starts_with("diff")
    ) %>% 
    rename(
      result = ftr,
      # evitate name with points
      maxo_h = max_odds.home,
      maxo_d = max_odds.draw,
      maxo_a = max_odds.away
    )
  
  return(df_ans)
}

# main ---------------------------------------------------------------------
#### read data ####
DATA_FBLL_UK <- "Data/football_uk/"
files_names <- list.files(DATA_FBLL_UK, pattern = ".csv$")

df_r <- read_dfs(files_names, DATA_FBLL_UK)

#### generate tables ####
# odds max - spread - market track take
df_odds <- get_odds(df_r)

# change in odds of pinnacle sports
df_updateodds.pinnacle <- get_odds.pinnacle(df_r)

# variance in odds
df_varodds <- get_varodds(df_updateodds.pinnacle)

# get all the trake takes of the books, long format
df_tt <- get_tracktake(df_r)

# final database
df_final <- get_tidy(df_r, df_odds)

#### write tables ####
write_csv(df_odds, file = "Data/football_uk/final_dbb/stats_odds.csv")
write_csv(df_updateodds.pinnacle, file = "Data/football_uk/final_dbb/odds_pinnacle.csv")
write_csv(df_varodds, file = "Data/football_uk/final_dbb/variance_odds_deciles.csv")
write_csv(df_tt, file = "Data/football_uk/final_dbb/track_take_books.csv")
write_csv(df_final, file = "Data/football_uk/final_dbb/footballuk_finaldbb.csv")
write_csv(df_r, file = "Data/football_uk/final_dbb/footballuk_raw.csv")
