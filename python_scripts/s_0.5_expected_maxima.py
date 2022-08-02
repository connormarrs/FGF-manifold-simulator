import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc
import csv
import DFGF_S1
import DFGF

os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())

# now we compute the growth of the expected maxima as a function of n.
# set parameters for the simulation
dirichlet = True
compute = True
n_start = 5
n_stop = 1000
n_step = 5

s = 0.5

numTrials = 10

linspace = np.arange(start = n_start, stop = n_stop+n_step, step = n_step)
sample = None
eigenValues = None
eigenVectors = None
maxima = np.empty((linspace.shape[0], 2))



for n_index in range(linspace.shape[0]):
	if n_index == 0:
		with open('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', 'w', newline = '') as csvFile:
			writer = csv.writer(csvFile)
			dfgf = DFGF_S1.DFGF_S1(s,linspace[n_index],numTrials,dirichlet,compute)
			dfgf.runTrials()
			dfgf.computeMaximaVector()
			dfgf.computeMeanOfMaxima()
			print('writing', linspace[n_index])
			row = [linspace[n_index], dfgf.getMeanOfMaxima()]
			writer.writerow(row)
	elif n_index>0:
		with open('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', 'a', newline = '') as csvFile:
			writer = csv.writer(csvFile)
			dfgf = DFGF_S1.DFGF_S1(s,linspace[n_index],numTrials,dirichlet,compute)
			dfgf.runTrials()
			dfgf.computeMaximaVector()
			dfgf.computeMeanOfMaxima()
			print('writing', linspace[n_index])
			row = [linspace[n_index], dfgf.getMeanOfMaxima()]
			writer.writerow(row)

#np.savetxt('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', maxima, delimiter=",")
