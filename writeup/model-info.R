library(tidyverse)
library(ggplot2)

# Final Graphs.v2.final.final
all <- read_csv('results-everything.csv')
å
# Find Highest Performing Fingerprint radius and dataset on entire validation set
all %>%
  filter(use_features == FALSE & use_chirality == FALSE) %>%
  mutate(k = as_factor(k), radius = as_factor(radius)) %>%
  ggplot(aes(inferred, corrcoef, fill=k)) +
  geom_point(color='black', shape=21, size=2, alpha=0.75) +
  facet_grid(activities ~ radius) +
  scale_y_continuous(name='Spearman Correlation') +
  scale_x_continuous(name='Number of Inferred Kinases', guide = guide_axis(angle = 90))

# Focus on R={1, 2} and ACT={KdKi, KdKiEC50IC50} with different features
all %>%
  filter(radius %in% c(1, 2), activities %in% c('KD, KI', 'KD, KI, EC50, IC50')) %>%
  mutate(k = as_factor(k), radius = as_factor(radius)) %>%
  ggplot(aes(inferred, corrcoef, fill=k)) +
  geom_point(color='black', shape=21, size=2, alpha=0.75) +
  facet_grid(activities ~ radius+use_chirality) +
  scale_y_continuous(name='Spearman Correlation') +
  scale_x_continuous(name='Number of Inferred Kinases', guide = guide_axis(angle = 90))

# Table of model results by validation/test set?
# Ensemble matching
# Powerset of Ensembles Results?
