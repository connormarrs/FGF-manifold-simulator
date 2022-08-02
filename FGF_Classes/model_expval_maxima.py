import sys
import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc

sys.path.append('../FGF_Classes')

import DFGF_S1

# now we compute the growth of the expected maxima as a function of n.
# set parameters for the simulation
dirichlet = True
compute = True
n_start = 1000
n_stop = 10000
n_step = 10

s_start = 0.0
s_stop = 0.5
s_step = 0.01

numTrials = 100

nvals_linspace = np.arange(start = n_start, stop = n_stop+n_step, step = n_step)
svals_linspace = np.arange(start = s_start, stop = s_stop+n_step, step = s_step)

sample = None
eigenValues = None
eigenVectors = None
maxima = np.empty((linspace.shape[0], 2))

for n_index in range(linspace.shape[0]):
	dfgf = DFGF_S1.DFGF_S1(s,linspace[n_index],numTrials,dirichlet,compute)
	dfgf.runTrials()
	dfgf.computeMaximaVector()
	dfgf.computeMeanOfMaxima()
	maxima[n_index] = np.array([linspace[n_index], dfgf.getMeanOfMaxima()])

np.savetxt('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', maxima, delimiter=",")
