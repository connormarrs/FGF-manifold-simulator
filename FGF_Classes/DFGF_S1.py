import DFGF
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import math
import threading
import multiprocessing as mp

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, sRange, n, numTrials, isDirchlet, compute):
		#set parameters
		self.sRange = sRange
		self.n=n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet

		if compute:
			self.computeSample()
			self.computeEigenValues()
			self.computeEigenVectors()
			self.computeCoefficients(self.s)


	def computeSample(self):
		#could use an nxnxnumTrials matrix and matrix multiplication but this could take up huge amounts of memory...right?
		self.sample = self.rng.standard_normal((self.numTrials, self.n))

	def computeEigenValues(self):
		tempVector = np.arange(1, math.ceil(self.n/2)+1)
		self.eigenValues = self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n))
		tempVector = np.arange(1, math.floor(self.n/2)+1)
		self.eigenValues = np.append(self.eigenValues, self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n)))

	def computeEigenVector(self, k):
		if self.isDirchlet == False:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			tempEigenVectorCosines = np.arange(1, math.ceil(self.n/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			cosines = np.cosine(2*np.pi*k/self.n * tempEigenVectorCosines)
			return np.append(consines,sines)
		elif self.isDirchlet == True:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			cosines = np.zeros((math.ceil(self.n/2)))
			return np.append(cosines, sines).reshape(1,self.n)
	def computeEigenVectors(self):
		self.eigenVectors = self.computeEigenVector(0)
		for k in np.arange(1,self.n):
			self.eigenVectors = np.insert(self.eigenVectors,self.eigenVectors.shape[0], self.computeEigenVector(k),axis  = 0)

	def computeCoefficients(self,s):
		denominators = np.power(self.eigenValues, -s)
		self.coefficients = np.multiply(self.eigenVectors , denominators)

	def evaluate(self,trialNum):
		return np.dot(self.coefficients, self.sample[trialNum])
	def computeTrial(self, trialNum):
		self.trialData[trialNum] = self.evaluate(trialNum)

	def runTrials(self):
		threads = []
		# num_workers = mp.cpu_count()  

		# pool = mp.Pool(num_workers)
		# tasks = np.arange(self.numTrials)
		# for task in tasks:
		#     pool.apply_async(self.computeTrial, args = (task,))

		# pool.close()
		# pool.join()

		#try with propermultiprocessing
		for trialNum in np.arange(self.numTrials):
			thread = threading.Thread(target = self.computeTrial, args =(trialNum,))
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join()
dfgf = DFGF_S1(.05, 1000, 5000, True, True)

dfgf.runTrials()
print(len(dfgf.getTrialData()))