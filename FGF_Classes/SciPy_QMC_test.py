import scipy
import numpy as np
import csv

nval = 10000
numTrials = 20000

# instantiate a multivariate normal distribution with SciPy
dist = scipy.stats.qmc.MultiVariateNormalQMC(
	mean=np.zeros(nval)
)

# generate an array of numTrials samples
sample = np.array(
	dist.random(numTrials)
)

with open('../Data/python_random_number_data_n_10000_trials_20000.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(sample)
