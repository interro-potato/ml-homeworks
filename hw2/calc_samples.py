# Used in question 2. to calculate the values of y_3-related probabilities

import numpy as np  
from scipy.stats import norm

y_3_observations = {
  "Positive": [1.2, 0.8, 0.5, 0.9, 0.8],
  "Negative": [1, 0.9, 1.2, 0.8]
}

y_3_values = [0.5, 0.8, 0.9, 1, 1.2]

def calc_normal(observations, label):
  mean = np.mean(observations)
  std = np.std(observations)
  for value in y_3_values:
    print(f'P(y_3 = {value} | class = {label}) = {norm.pdf(value, mean, std)}')

for label, observations in y_3_observations.items():
  calc_normal(observations, label)
