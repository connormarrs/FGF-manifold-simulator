import DFGF
# import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# import tensorflow.experimental.numpy as tnp
import math
import csv
import threading

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, sRange, n, numTrials, isDirchlet, compute, useThreads):
		#set parameters
		self.sRange = sRange
		self.n=n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet
		self.useThreads = useThreads

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
			return np.append(cosines,sines)
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
		if self.useThreads:
			# python threading
			threads = []
			for trialNum in np.arange(self.numTrials):
				thread = threading.Thread(target = self.computeTrial, args =(trialNum,))
				thread.start()
				threads.append(thread)
			for thread in threads:
				thread.join()
		else:
			# no threading
			for trialNum in np.arange(self.numTrials):
				self.computeTrial(trialNum)

# Instantiate a DFGF to test its
num_trials = 20000
n_val = 10000
dfgf = DFGF_S1(.5, n_val, num_trials, True, True, False)

dfgf.runTrials()

print("Completed running trials")

# write some shitty code to compute the mean
emp_mean_of_max = np.mean(
	np.array(
		[np.max(dfgf.sample[j]) for j in range(num_trials)]
	)
)

# make a numTrials x 2 array to hold max values
max_n_array = []
for j in range(num_trials):
	max_n_array.append(
		[np.max(dfgf.sample[j]), emp_mean_of_max]
	)

# Write its sample random number data to a csv
with open('../Data/MaxDist/python_random_number_data_n10000_trials20000.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerows(max_n_array)

print("Completed")

# Write its trial data to a csv
# TODO