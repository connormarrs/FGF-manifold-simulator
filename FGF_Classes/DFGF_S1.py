# import DFGF
# import matplotlib.pyplot as plt
# import numpy as np
# # import tensorflow as tf
# # import tensorflow.experimental.numpy as tnp
# import math
# import threading
# import multiprocessing as mp

# class DFGF_S1(DFGF.DFGF):
	
# 	def __init__(self, sRange, n, numTrials, isDirchlet, compute):
# 		#set parameters
# 		self.sRange = sRange
# 		self.n=n
# 		self.numTrials = numTrials
# 		self.isDirchlet = isDirchlet

# 		if compute:
# 			self.computeSample()
# 			print("sample computed")
# 			self.computeEigenValues()
# 			print("eigenvalues computed")
# 			self.computeEigenVectors()
# 			print("eigenvectors computed")
# 			self.computeCoefficients(self.s)
# 			print("coefficients computed")


# 	def computeSample(self):
# 		#could use an nxnxnumTrials matrix and matrix multiplication but this could take up huge amounts of memory...right?
# 		self.sample = self.rng.standard_normal((self.numTrials, self.n))

# 	def computeEigenValues(self):
# 		tempVector = np.arange(1, math.ceil(self.n/2)+1)
# 		self.eigenValues = self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n))
# 		tempVector = np.arange(1, math.floor(self.n/2)+1)
# 		self.eigenValues = np.append(self.eigenValues, self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n)))

# 	def computeEigenVector(self, k):
# 		if self.isDirchlet == False:
# 			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
# 			tempEigenVectorCosines = np.arange(1, math.ceil(self.n/2)+1)
# 			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
# 			cosines = np.cosine(2*np.pi*k/self.n * tempEigenVectorCosines)
# 			return np.append(consines,sines)
# 		elif self.isDirchlet == True:
# 			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
# 			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
# 			cosines = np.zeros((math.ceil(self.n/2)))
# 			return np.append(cosines, sines).reshape(1,self.n)
# 	def computeEigenVectors(self):
# 		self.eigenVectors = self.computeEigenVector(0)
# 		for k in np.arange(1,self.n):
# 			self.eigenVectors = np.insert(self.eigenVectors,self.eigenVectors.shape[0], self.computeEigenVector(k),axis  = 0)

# 	def computeCoefficients(self,s):
# 		denominators = np.power(self.eigenValues, -s)
# 		self.coefficients = np.multiply(self.eigenVectors , denominators)

# 	def evaluate(self,trialNum):
# 		return np.dot(self.coefficients, self.sample[trialNum])

# 	def computeTrial(self, trialNum):
# 		trialDataQueue.put([trialNum, self.evalueate(trialNum)])

# 	def runTrials(self):
# 		# python multiprocessing
# 		print("computing trials")
# 		num_workers = mp.cpu_count()
# 		tasks = [*range(self.numTrials)] 
# 		pool = mp.Pool(num_workers)
# 		for task in tasks:
# 			pool.apply_async(self.computeTrial, args = (int(task), self.trialData))


# 		for trial in range(self.numTrials):
# 			temp = self.trialDataQueue.get()
# 			self.trialData[temp[0]] = temp[1]
# 		pool.close()
# 		pool.join()
# 		# print(num_workers)
# 		# pool = mp.Pool(num_workers)
# 		# pool.map(self.computeTrial, [1,2,3,4])
# 		# pool.close()
# 		# pool.join()

# 		# python threading
# 		# threads = []
# 		# for trialNum in np.arange(self.numTrials):
# 		# 	thread = threading.Thread(target = self.computeTrial, args =(trialNum,))
# 		# 	thread.start()
# 		# 	threads.append(thread)
# 		# for thread in threads:
# 		# 	thread.join()

# 		# # no threading
# 		# for trialNum in np.arange(self.numTrials):
# 		# 	self.computeTrial(trialNum)



# dfgf = DFGF_S1(.05, 5000, 1000, True, True)

# dfgf.runTrials()
# print(len(dfgf.getTrialData()))



import DFGF
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# import tensorflow.experimental.numpy as tnp
import math
import threading
import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, s, n, numTrials, isDirchlet, compute):
		#set parameters
		self.s = s
		self.n=n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet

		if compute:
			self.computeSample()
			print("sample computed")
			self.computeEigenValues()
			print("eigenvalues computed")
			self.computeEigenVectors()
			print("eigenvectors computed")
			self.computeCoefficients()
			print("coefficients computed")


	def computeSample(self):
		#could use an nxnxnumTrials matrix and matrix multiplication but this could take up huge amounts of memory...right?
		self.sample = self.rng.standard_normal((self.numTrials, self.n))

	def computeEigenValues(self):
		tempVector = np.arange(1, math.ceil(self.n/2)+1)
		self.eigenValues = self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n))
		tempVector = np.arange(1, math.floor(self.n/2)+1)
		self.eigenValues = np.append(self.eigenValues, self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n)))
		self.denominators = np.power(self.eigenValues, -self.s)

	def computeEigenVector(self, k):
		if self.isDirchlet == False:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			tempEigenVectorCosines = np.arange(1, math.ceil(self.n/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			cosines = np.cosine(2*np.pi*k/self.n * tempEigenVectorCosines)
			#return np.append(consines,sines).reshape(1,self.n)
			#self.eigenVectorDict[k] = np.append(consines,sines).reshape(1,self.n)
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n)])
		elif self.isDirchlet == True:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			cosines = np.zeros((math.ceil(self.n/2)))
			#return np.append(cosines, sines).reshape(1,self.n)
			#self.eigenVectorDict[k] = np.append(cosines, sines).reshape(1,self.n)
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n)])
	def computeEigenVectors(self):
		# self.eigenVectors = self.computeEigenVector(0)
		# for k in np.arange(1,self.n):
		# 	self.eigenVectors = np.insert(self.eigenVectors,self.eigenVectors.shape[0], self.computeEigenVector(k),axis  = 0)
		# threads = []
		# for k in np.arange(self.n):
		# 	thread = threading.Thread(target = self.computeEigenVector, args =(k,))
		# 	thread.start()
		# 	threads.append(thread)
		# for thread in threads:
		# 	thread.join()
		# self.eigenVectors = self.eigenVectorDict[0]

		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		pool.map(self.computeEigenVector, [*range(self.n)])


		for vector in range(self.n):
			print('getting eigenVector: ', vector)
			temp = self.eigenVectorQueue.get()
			print(temp[1].shape)
			self.eigenVectorDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		self.eigenVectors = self.eigenVectorDict[0].reshape(1,self.n)
		for k in np.arange(1,self.n):
			self.eigenVectors = np.insert(self.eigenVectors,self.eigenVectors.shape[0], self.eigenVectorDict[k],axis  = 0)

	# def computeCoefficients(self):
	# 	denominators = np.power(self.eigenValues, -self.s)
	# 	self.coefficients = np.multiply(self.eigenVectors , denominators)
	def computeCoefficientsHelper(self, i):
		#self.coefficientsDict[i] = np.multiply(self.eigenVectors[i], self.denominators)
		self.coefficientsQueue.put([i,np.multiply(self.eigenVectors[i], self.denominators)])
	def computeCoefficients(self):
		# threads = []
		# for i in range(self.eigenVectors.shape[0]):
		# 	thread = threading.Thread(target = self.computeCoefficientsHelper, args =(i,))
		# 	thread.start()
		# 	threads.append(thread)
		# for thread in threads:
		# 	thread.join()
		# self.coefficients = self.coefficientsDict[0].reshape(1,self.n)
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		pool.map(self.computeCoefficientsHelper, [*range(self.n)])

		for vector in range(self.n):
			print('getting coefficient: ', vector)
			temp = self.coefficientsQueue.get()
			print(temp[1].shape)
			self.coefficientsDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		self.coefficients = self.coefficientsDict[0].reshape(1,self.n)
		for k in np.arange(1,self.n):
			self.coefficients = np.insert(self.coefficients,self.coefficients.shape[0], self.coefficientsDict[k],axis  = 0)
	def evaluate(self,trialNum):
		return np.dot(self.coefficients, self.sample[trialNum])

	def computeTrial(self, trialNum):
		print("computing trial: ", trialNum)
		self.trialDataQueue.put([trialNum, self.evaluate(trialNum)])

	def runTrials(self):
		# python multiprocessing
		print("computing trials")
	
		# tasks = [*range(self.numTrials)] 
		# pool = mp.Pool(num_workers)
		# for task in tasks:
		# 	pool.apply_async(self.computeTrial, args = (int(task), self.trialData))

		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		pool.map(self.computeTrial, [*range(self.numTrials)])

		for trial in range(self.numTrials):
			print('getting trial: ', trial)
			temp = self.trialDataQueue.get()
			self.trialData[temp[0]] = temp[1]

		pool.close()
		pool.join()

		# print(num_workers)
		# pool = mp.Pool(num_workers)
		# pool.map(self.computeTrial, [*range(self.numTrials)])
		# pool.close()
		# pool.join()

		# python threading
		# threads = []
		# for trialNum in np.arange(self.numTrials):
		# 	thread = threading.Thread(target = self.computeTrial, args =(trialNum,))
		# 	thread.start()
		# 	threads.append(thread)
		# for thread in threads:
		# 	thread.join()

		# # no threading
		# for trialNum in np.arange(self.numTrials):
		# 	self.computeTrial(trialNum)



dfgf = DFGF_S1(.05, 5000, 1000, True, True)

dfgf.runTrials()
print("length is ", len(dfgf.getTrialData()))
print(dfgf.coefficients.shape)
