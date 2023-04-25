library(tidyverse)
library(ggplot2)

# data loading
df <- read_csv('perf-all-models.csv')

theme_set(theme_bw())

df %>% 
  filter(r2 >= -2.0) %>%
  ggplot(aes(corrcoef, r2)) +
  geom_point(aes(color=activities)) +
  geom_smooth(method='lm', formula= y~x) +
  #geom_vline(xintercept = 30) +
  facet_wrap(~radius)

evals <- read_csv('../data/eval/results.csv')

evals %>%
  mutate(radius = as_factor(radius)) %>%
  ggplot(aes(inferred, corrcoef, fill=radius)) +
  scale_color_brewer(palette = "Dark2") +
  geom_point(color='black', shape=21, size=2, alpha=0.8) +
  facet_grid(activities ~ radius)
