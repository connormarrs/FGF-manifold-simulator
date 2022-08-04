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
		self.sample = np.zeros((self.numTrials, self.n-1,self.n-1), dtype = float)
		self.eigenValues = np.zeros((self.n-1,self.n-1), dtype = float)
		self.denominators = np.zeros((self.n-1,self.n-1), dtype = float)
		self.eigenVectors = np.zeros((self.n,self.n,self.n-1,self.n-1), dtype = float)
		self.coefficients = np.zeros((self.n,self.n,self.n-1,self.n-1), dtype = float)
		self.trialData = np.zeros((self.numTrials, self.n, self.n), dtype = float)

		if compute:
			self.computeSample()
			self.computeEigenValues()
			self.computeEigenVectors()
			self.computeCoefficients()

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

	def computeEigenVector(self, ks):
		k1 = ks[0]
		k2 = ks[0]
		if self.isDirichlet==False:
			tempEigenVectorSinesp = np.arange(1,math.floor((self.n-1)/2)+1)
			tempEigenVectorCosinesp= np.arange(1, math.ceil((self.n-1)/2)+1)
			tempVectorq = np.arange(1,self.n)
			sines = np.sin(2*np.pi*(1/self.n)*np.add.outer(k1*tempEigenVectorSinesp, k2*tempVectorq))
			cosines =  np.sin(2*np.pi*(1/self.n)*np.add.outer(k1*tempEigenVectorCosinesp, k2*tempVectorq))
			self.eigenVectorQueue.put([[k1,k2],np.concatenate((cosines, sines), axis = 0)])
		elif self.isDirichlet == True:
			tempEigenVectorSinesp = np.arange(1,math.floor((self.n-1)/2)+1)
			tempEigenVectorCosinesp= np.arange(1, math.ceil((self.n-1)/2)+1)
			tempVectorq = np.arange(1,self.n)
			sines = np.sin(2*np.pi*(1/self.n)*np.add.outer(k1*tempEigenVectorSinesp, k2*tempVectorq))
			cosines = np.zeros((math.ceil((self.n-1)/2), self.n-1))
			self.eigenVectorQueue.put([repr(ks),np.concatenate((cosines, sines), axis = 0)])
		
	def computeEigenVectors(self):
		kRange = [*range(self.n)]
		kInputs = [(k1, k2) for k1 in kRange for k2 in kRange]
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		pool.map(self.computeEigenVector, kInputs)
		for k in range(len(kInputs)):
			temp = self.eigenVectorQueue.get()
			self.eigenVectorDict[temp[0]] = temp[1]
		pool.close()

		for pair in kInputs:
			self.eigenVectors[pair[0],pair[1]] = self.eigenVectorDict[repr(pair)]

	def computeCoefficientsHelper(self,ks):
		k1 = ks[0]
		k2 = ks[1]
		self.coefficientsQueue.put([repr(ks), np.multiply(self.eigenVectors[k1,k2], self.denominators)])
	def computeCoefficients(self):
		kRange = [*range(self.n)]
		kInputs = [(k1, k2) for k1 in kRange for k2 in kRange]
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		pool.map(self.computeCoefficientsHelper, kInputs)
		for k in range(len(kInputs)):
			temp = self.coefficientsQueue.get()
			self.coefficientsDict[temp[0]] = temp[1]
		pool.close()
		for pair in kInputs:
			self.coefficients[pair[0],pair[1]] = self.coefficientsDict[repr(pair)]
		print(self.coefficients)

	def runTrialsHelper(self, ks):
		k1 = ks[0]
		k2 = ks[1]

	def runTrials(self):
		self.coefficients.reshape(self.n,self.n,(self.n-1)**2)
		#print(self.coefficients)
	def computeMaxima(self):
		pass
	def computeMeanOfMaxima(self):
		pass

dfgf = DFGF_T2(1,5,5,True,True)

dfgf.runTrials()