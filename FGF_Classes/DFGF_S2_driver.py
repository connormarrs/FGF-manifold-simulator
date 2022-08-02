import sys
import numpy as np
import math
import multiprocessing as mp
import os
#from scipy.stats import qmc

import DFGF_S2
import Laplace_S2

n_start = 10              #number of points to start at
n_stop = 100              #maximum number of points to be computed
n_step = 10               #increment size between number of points

s = 0.5                   #DFGF parameter

boundingParam = 0.1       #how far the eigenvalues should stay bounded away from 0
bandwidthParam = 1        #constant for determining the size of bandwidth parameters for laplacian

numTrials = 100           #how many trials should be run for each number of points

laplace = Laplace_S2.Laplace_S2(n_stop, n_step, boundingParam, bandwidthParam)
laplace.computeEigenSystems()

linspace = np.arange(start = n_start, stop = n_stop + n_step, step = n_step)
sample = None
eigenValues = laplace.getEigenValues()
eigenVectors = laplace.getEigenVectors()
#maxima = np.empty((linspace.shape[0], 2))


for n_index in range(linspace.shape[0]):
    dfgf = DFGF_S2.DFGF_S2(s, linspace[n_index], numTrials, eigenValues[linspace[n_index]], eigenVectors[linspace[n_index]], laplace.grid[:linspace[n_index]])
    dfgf.runTrials()
    dfgf.computeMaxima()
    dfgf.computeEmpMean()
    #maxima[n_index] = np.array([linspace[n_index], dfgf.getMeanofMaxima()])

#np.savetxt('../output/expected_maxima_s_'+str(s)+'_n_'+str(n_start)+'-'+str(n_stop)+'_numTrials_'+str(numTrials)+'.csv', maxima, delimiter=",")

print(min(laplace.getSpectralGaps()))
print(laplace.getComputeCounts())