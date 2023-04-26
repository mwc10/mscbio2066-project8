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

evals <- read_csv('../data/eval/alltypes-results.csv')
  

evals %>%
  filter(k != 1) %>%
  mutate(radius = as_factor(radius), k = as_factor(k)) %>%
  ggplot(aes(inferred, corrcoef, fill=k)) +
  scale_color_brewer(palette = "Dark2") +
  geom_point(color='black', shape=21, size=2, alpha=0.6) +
  facet_grid(activities ~ radius + use_chirality)

evals %>%
  filter(radius == 3) %>%
  mutate(min_cmpds = replace_na(min_cmpds, 10)) %>%
  ggplot(aes(metric_val, corrcoef, color=min_cmpds)) +
  geom_point() +
  facet_wrap(~activities)

topk <- read_csv('../data/eval/k-results.csv')

topk %>%
  filter(radius == 3) %>%
  ggplot(aes(inferred, corrcoef, fill=k)) +
  scale_color_brewer(palette = "Dark2") +
  geom_point(color='black', shape=21, size=2, alpha=0.8) +
  facet_grid(activities ~ k)
