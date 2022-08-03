import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc
import DFGF_S1
import DFGF

# driver file to compute the approximate pdf of the maxima of the DFGF on S1

# first set n values and s values and number of trials for the simulation

# also specify the dirichlet case

s = 0.5 
n = 20000
numTrials = 40000
dirichlet = True
compute = True

# create an instance of DFGF_S1 with the given parameters
distributionDFGF = DFGF_S1.DFGF_S1(s,n,numTrials,dirichlet,compute)

# run the calculations to get a vector of maxima
distributionDFGF.runTrials()
distributionDFGF.computeMaximaVector()

# write maximaVector to csv
maximaVector = distributionDFGF.getMaximaVector()
maximaVector = maximaVector.reshape(maximaVector.shape[0], 1)
np.savetxt('output/maxima_distributions_s_'+str(s)+'_n_'+str(n)+'_numTrials_'+str(numTrials)+'.csv', maximaVector, delimiter=",")


