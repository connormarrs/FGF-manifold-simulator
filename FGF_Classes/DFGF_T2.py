import DFGF
import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc

# sets the number of threads available to python based on system specs
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())


class DFGF_T2(DFGF.DFGF):
	def __init__(self, s, n, numTrials, isDirichlet, compute):
				#set parameters
		self.s = s
		self.n = n
		self.numTrials = numTrials
		self.isDirichlet = isDirichlet
		self.sample = np.zeros((self.numTrials, self.n,self.n), dtype = float)
		self.eigenValues = np.zeros((self.n,self.n), dtype = float)
		self.denominators = np.zeros((self.n,self.n), dtype = float)
		self.eigenVectors = np.zeros((self.n,self.n,self.n), dtype = float)
		self.coefficients = np.zeros((self.n,self.n,self.n), dtype = float)
		self.trialData = np.zeros((self.numTrials, self.n, self.n), dtype = float)

		if compute:
			self.computeSample()
			self.computeEigenValues()
			# self.computeEigenVectors()
			# self.computeCoefficients()

	def computeSample(self):
		dist = qmc.MultivariateNormalQMC(
			mean = np.zeros(self.n-1)
			)
		for i in range(self.numTrials):
			self.sample[i] = np.array(
				dist.random(self.n-1)
				)

	def computeEigenValues(self):
		normalizer = np.power(self.n, 2) / (2* np.power(np.pi, 2))
		arg1 = (2*np.pi)/(self.n) * np.arange(1, self.n)
		arg2 = (2*np.pi)/(self.n) * np.arange(1, self.n)
		self.eigenValues = normalizer* (2-np.subtract.outer(np.cos(arg1),np.cos(arg2)))
		self.denominators = np.power(self.eigenValues, self.s)

	def computeEigenVector(self, k1, k2):
		tempEigenVectorSines = np.arange(1,math.floor((self.n-1)/2)+1)
		tempEigenVectorCosines = np.arange(1, math.ceil((self.n-1)/2)+1)

	def computeEigenVectors(self):



 		
dfgf = DFGF_T2(1,100,20,True,True)

