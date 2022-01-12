#####################################################
# Understats, data wrange to create final dataframe

# 06/07/2021
# Roman Alberto Velez Jimenez
#####################################################
library(tidyverse)
library(tidyr)
library(ggrepel)


# functions ---------------------------------------------------------------
rho_bet <-  function(decimal_odd, result, bet){
  if_else(
    result == 1,
    bet * (decimal_odd - 1),
    - bet,
  )
}





# init params -------------------------------------------------------------
FILE_BETS <- "Results/bets/ridge_complete_markowitz.csv"

FRACTIONAL_KELLY <- 1


# main --------------------------------------------------------------------
# read
df_r <- read_csv(FILE_BETS)

# wrangle
df_main_bets <- df_r %>% 
  mutate(
    fractional_bets = bets * FRACTIONAL_KELLY,
    fractional_return = rho_bet(odds, results, fractional_bets),
    theoric_return = portfolio_return * FRACTIONAL_KELLY
  ) 

# growth
ts_wealth <- df_main_bets %>% 
  group_by(
    matchweek
  ) %>% 
  summarise(
    wealth_return = 1 + sum(fractional_return),
    theoric_return = theoric_return[1]
  ) %>% 
  ungroup(
    
  ) %>% 
  mutate(
    final_wealth = cumprod(wealth_return),
    theoric_final_wealth = cumprod(theoric_return)
  ) %>% 
  add_row(
    matchweek = 17, wealth_return = 1, final_wealth = 1
  ) %>% 
  arrange(
    matchweek
  )

#### plot ####
# matchweeks
mtchwk_vec <- ts_wealth$matchweek %>% as.character()
mtchwk_vec[1] <- 't0'

ts_wealth %>% 
  ggplot(aes(x = matchweek)) +
  geom_col(aes(y = wealth_return - 1), fill = 'gray70') +
  geom_line(aes(y = final_wealth), color = 'black', size = 1) +
  geom_point(aes(y = final_wealth), color = 'black', size = 4) +
  geom_text_repel(
    aes(y = final_wealth + 5e-2, label = str_glue('{round(final_wealth * 100, 2)} %')),
    size = 4,
    color = 'gray30',
    direction = "y",
    nudge_y = 1e-3
  ) +
  xlab("Jornada") + ylab("Riqueza") +
  ggtitle("Riqueza Acumulada por Apuestas") +
  scale_x_continuous(labels = mtchwk_vec, breaks = ts_wealth$matchweek, minor_breaks = NULL) +
  scale_y_continuous(labels = scales::percent, limits = c(-1, 1.05)) +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 20),
    axis.text.y = element_text(size = 20),
    axis.title.x = element_text(size = 25),
    axis.title.y = element_text(size = 25),
    title = element_text(size = 25),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 30),
    legend.position = "bottom",
    strip.text.x = element_text(size = 20)
  ) 

# save
dir.create("Figures/Financial_Model/")
ggsave(
  filename = str_glue("complete_markowitz{FRACTIONAL_KELLY}.png"), 
  device = png(), 
  path = "Figures/Financial_Model/", 
  width = 25, 
  height = 15,
  scale = 0.5
  )




































