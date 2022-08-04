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
		self.denominators = np.power(self.eigenValues, -self.s)

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

	def evaluate(self, trialNum):
		self.trialDataQueue.put([trialNum,np.dot(self.coefficients, self.sample[trialNum])])
	def runTrials(self):
		self.coefficients = self.coefficients.reshape(self.n,self.n,(self.n-1)**2)
		self.sample = self.sample.reshape(self.numTrials, (self.n-1)**2)
		# python multiprocessing
		# instantiate threadpoo
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		# evaluate trials
		pool.map(self.evaluate, [*range(self.numTrials)])
		# gets the trial data from the associated multiprocessing queue and places them into the associated dictionary
		# this ensures data is entered in the correct order and protects against racing
		for trial in range(self.numTrials):
			temp = self.trialDataQueue.get()
			self.trialDataDict[temp[0]] = temp[1]
		for i in range(self.numTrials):
			self.trialData[i] = self.trialDataDict[i]

		pool.close()
		pool.join()

	def computeMaxima(self):
		temp = 0.5*(self.trialData[:,:,math.ceil(self.n/2)]+self.trialData[:,:,math.floor(self.n/2)]).reshape(self.numTrials,self.n,1)
		maximaCandidates = self.trialData[:,:,0:math.floor(self.n/2)]
		maximaCandidates = np.append(maximaCandidates, temp, axis = 2)
		print(maximaCandidates.shape)
		print(maximaCandidates[:,math.ceil(self.n/2),:].shape, maximaCandidates[:,math.floor(self.n/2),:].shape)
		temp = 0.5*(maximaCandidates[:,math.ceil(self.n/2),:]+maximaCandidates[:,math.floor(self.n/2),:]).reshape(self.numTrials,1,maximaCandidates.shape[2])
		maximaCandidates = maximaCandidates[:,0:math.floor(self.n/2),:]
		maximaCandidates = np.append(maximaCandidates, temp, axis = 1)
		maximaCandidates = maximaCandidates.reshape(self.numTrials, (maximaCandidates.shape[1]*maximaCandidates.shape[2]))
		self.maximaVector = np.amax(maximaCandidates, axis=1)

	def computeMeanOfMaxima(self):
		self.meanOfMaxima = np.mean(self.maximaVector)
