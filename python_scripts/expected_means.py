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
n_start = 500
n_stop = 2000
n_step = 10

s_start = 0.000000
s_stop = 0.500000
s_step = .005

numTrials = 2000

linspace = np.arange(start = n_start, stop = n_stop+n_step, step = n_step)
linspace_s = np.arange(start = s_start, stop = s_stop+s_step, step = s_step)
sample = None
eigenValues = None
eigenVectors = None
maxima = np.empty((linspace.shape[0], 2))

sample = None
eigenValues = None
EigenVectors = None



for n_index in range(linspace.shape[0]):
	for s_index in range(linspace_s.shape[0]):
		if s_index == 0:
			with open('output/expected_maxima_s_'+str(s_start)+'-'+str(s_stop)+'_n_'+str(linspace[n_index])+'_numTrials_'+str(numTrials)+'.csv', 'w', newline = '') as csvFile:
				writer = csv.writer(csvFile)
				dfgf = DFGF_S1.DFGF_S1(linspace_s[s_index],linspace[n_index],numTrials,dirichlet,True)
				dfgf.runTrials()
				dfgf.computeMaximaVector()
				dfgf.computeMeanOfMaxima()
				print('writing n=',linspace[n_index],', s= ', linspace_s[s_index])
				row = [linspace_s[s_index], dfgf.getMeanOfMaxima()]
				writer.writerow(row)
				sample = dfgf.getSample()
				eigenValues = dfgf.getEigenValues()
				eigenVectors = dfgf.getEigenVectors()
		elif s_index>0:
			with open('output/expected_maxima_s_'+str(s_start)+'-'+str(s_stop)+'_n_'+str(linspace[n_index])+'_numTrials_'+str(numTrials)+'.csv', 'a', newline = '') as csvFile:
				writer = csv.writer(csvFile)
				dfgf = DFGF_S1.DFGF_S1(linspace_s[s_index],linspace[n_index],numTrials,dirichlet,False)
				dfgf.setParams(sample, eigenValues, eigenVectors)
				dfgf.runTrials()
				dfgf.computeMaximaVector()
				dfgf.computeMeanOfMaxima()
				print('writing n=',linspace[n_index],', s= ', linspace_s[s_index])
				row = [linspace_s[s_index], dfgf.getMeanOfMaxima()]
				writer.writerow(row)

#np.savetxt('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', maxima, delimiter=",")
