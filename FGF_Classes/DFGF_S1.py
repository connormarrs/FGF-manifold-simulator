import DFGF
import pandas as pd
# import tensorflow.experimental.numpy as tnp
import numpy as tnp
import math
import csv
import threading

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, sRange, n, numTrials, isDirchlet, compute, useThreads, useSample):
		#set parameters
		self.sRange = sRange
		self.n=n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet
		self.useThreads = useThreads
		self.read_sample = useSample

		if compute:
			if self.read_sample:
				self.readSample()
			else:
				self.computeSample()
			self.computeEigenValues()
			self.computeEigenVectors()
			self.computeCoefficients(self.s)


	def computeSample(self):
		#could use an nxnxnumTrials matrix and matrix multiplication but this could take up huge amounts of memory...right?
		self.sample = self.rng.standard_normal((self.numTrials, self.n))

	def readSample(self):
		Jupyter_Random_Data_File = r"/Users/connormarrs/Math/FGF-manifold-simulator/Data/python_random_number_data_n100_trials100.csv"

		colnames=[f'{i}' for i in range(100)]
		self.sample = tnp.array(pd.read_csv(Jupyter_Random_Data_File, names=colnames))

	def computeEigenValues(self):
		tempVector = tnp.arange(1, math.ceil(self.n/2)+1)
		self.eigenValues = self.n**2/(2*tnp.pi**2)*(1-tnp.cos(2*tnp.pi*(tempVector)/self.n))
		tempVector = tnp.arange(1, math.floor(self.n/2)+1)
		self.eigenValues = tnp.append(self.eigenValues, self.n**2/(2*tnp.pi**2)*(1-tnp.cos(2*tnp.pi*(tempVector)/self.n)))

	def computeEigenVector(self, k):
		if self.isDirchlet == False:
			tempEigenVectorSines = tnp.arange(1,math.floor(self.n/2)+1)
			tempEigenVectorCosines = tnp.arange(1, math.ceil(self.n/2)+1)
			sines = (tnp.sin(2*tnp.pi*k/self.n * tempEigenVectorSines))
			cosines = tnp.cosine(2*tnp.pi*k/self.n * tempEigenVectorCosines)
			return tnp.append(cosines,sines)
		elif self.isDirchlet == True:
			tempEigenVectorSines = tnp.arange(1,math.floor(self.n/2)+1)
			sines = (tnp.sin(2*tnp.pi*k/self.n * tempEigenVectorSines))
			cosines = tnp.zeros((math.ceil(self.n/2)))
			return tnp.append(cosines, sines).reshape(1,self.n)
	def computeEigenVectors(self):
		self.eigenVectors = self.computeEigenVector(0)
		for k in tnp.arange(1,self.n):
			self.eigenVectors = tnp.insert(self.eigenVectors,self.eigenVectors.shape[0], self.computeEigenVector(k),axis  = 0)

	def computeCoefficients(self,s):
		denominators = tnp.power(self.eigenValues, -s)
		self.coefficients = tnp.multiply(self.eigenVectors , denominators)

	def evaluate(self,trialNum):
		return tnp.dot(self.coefficients, self.sample[trialNum])
	def computeTrial(self, trialNum):
		self.trialData[trialNum] = self.evaluate(trialNum)

	def runTrials(self):
		if self.useThreads:
			# python threading
			threads = []
			for trialNum in tnp.arange(self.numTrials):
				thread = threading.Thread(target = self.computeTrial, args =(trialNum,))
				thread.start()
				threads.append(thread)
			for thread in threads:
				thread.join()
		else:
			# no threading
			for trialNum in tnp.arange(self.numTrials):
				self.computeTrial(trialNum)

# Instantiate a DFGF to test its
num_trials = 200
n_val = 100
dfgf = DFGF_S1(.5, n_val, num_trials, True, True, False, True)

print(dfgf.getSample())

print("Completed running trials")

# # write some shitty code to compute the mean
# emp_mean_of_max = tnp.mean(
# 	tnp.array(
# 		[tnp.max(dfgf.sample[j]) for j in range(num_trials)]
# 	)
# )

# # make a numTrials x 2 array to hold max values
# max_n_array = []
# for j in range(num_trials):
# 	max_n_array.append(
# 		[tnp.max(dfgf.sample[j]), emp_mean_of_max]
# 	)

# # Write its sample random number data to a csv
# with open('../Data/MaxDist/python_random_number_data_n10000_trials20000.csv', 'w') as file:
# 	writer = csv.writer(file)
# 	writer.writerows(max_n_array)

# print("Completed")

# Write its trial data to a csv
# TODO