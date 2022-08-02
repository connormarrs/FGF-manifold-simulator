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
			# self.computeEigenValues()
			# self.computeEigenVectors()
			# self.computeCoefficients()

	def computeSample(self):
		dist = qmc.MultivariateNormalQMC(
			mean = np.zeros(self.n)
			)
		for i in range(self.numTrials):
			print(self.sample[i])
			self.sample[i] = np.array(
				dist.random(self.n)
				)

	def computeEigenValues(self):
		

dfgf = DFGF_T2(1,100,20,True,True)