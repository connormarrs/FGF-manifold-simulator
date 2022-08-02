import DFGF
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> 520c17a8040026844910a9aaee8b96176415d090
import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc

# sets the number of threads available to python based on system specs
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, s, n, numTrials, isDirchlet, compute):
		#set parameters
		self.s = s
		self.n = n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet

		if compute:
			self.computeSample()
			self.computeEigenValues()
			self.computeEigenVectors()
			self.computeCoefficients()

	# computes the samples of random vectors and stores them as a numpy array in samples
	def computeSample(self):
<<<<<<< HEAD
		# could use an nxnxnumTrials matrix and matrix multiplication but this could take up huge amounts of memory...right?
		self.sample = self.rng.standard_normal((self.numTrials, self.n))
		# replace with scipy rng^^

	# uses numpy operations to compute the vector of eigenvalues
	# the eigenvalues can be passed to other instances of DFGF_S1 with the same n value
=======
		dist = qmc.MultiVariateNormalQMC(
			mean=np.zeros(self.n)
		)

		# generate an array of numTrials samples
		self.sample = np.array(
			dist.random(self.numTrials)
		)

	# compute the eigenvalues and stores them for future use
>>>>>>> 520c17a8040026844910a9aaee8b96176415d090
	def computeEigenValues(self):
		tempVector = np.arange(1, math.ceil(self.n/2)+1)
		# first calculate the eigenvalues associated with cosine eigenvectors
		self.eigenValues = self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n))
		# create temp vector to help with computation of the sine associated eigenvalues
		tempVector = np.arange(1, math.floor(self.n/2)+1)

		# append the calculated eigenvalues to the eigenvalue vector
		self.eigenValues = np.append(self.eigenValues, self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n)))
		# compute the denominators for the terms in the sum for calculating the value of the DFGF on S1
		self.denominators = np.power(self.eigenValues, -self.s)

	# computes the eigenvector for DFGF_S1
	# the eigenvectors can be passed to other instances of DFGF_S1 with the same n value
	def computeEigenVector(self, k):
		if self.isDirchlet == False:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			tempEigenVectorCosines = np.arange(1, math.ceil(self.n/2)+1)
			sines = (np.sin((2*np.pi*k/self.n) * tempEigenVectorSines))
			cosines = np.cosine((2*np.pi*k/self.n) * tempEigenVectorCosines)

			# places the reseult of the caluclation into the multiprocessing queue
			# this allows multiple threads to share data
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n)])

		# evaluates the dirichlet case
		elif self.isDirchlet == True:
			tempEigenVectorSines = np.arange(1,math.floor(self.n/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			# returns zeros instead of the cosine eigenfunctions
			cosines = np.zeros((math.ceil(self.n/2)))
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n)])

	# computes the eigenvectors for DFGF_S1 using the computeEigenVector helper function
	# creates threads to calculate each eigenvector in then compiles them into one 2d numpy array
	# note that the rows of eigenVectors are of the form phi_m(k) for fixed k and varying m
	# this will make dot products by the sample easier later
	def computeEigenVectors(self):
		# instantiate a threadpool to work on calculating the eigenvectors
		# threadpool is slower for small n values
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		# evaluates computeEigenVector for each k value from 0 to n-1
		pool.map(self.computeEigenVector, [*range(self.n)])

		# gets the values from the multiprocessing queue and places them into the associated dictionary
		# this ensures data is entered in the correct order and protects against racing
		for vector in range(self.n):
			print('getting eigenVector: ', vector)
			temp = self.eigenVectorQueue.get()
			print(temp[1].shape)
			self.eigenVectorDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		# takes eigenvectors from the eigenVectorDict and places them into a 2d numpy array
		self.eigenVectors = self.eigenVectorDict[0].reshape(1,self.n)
		for k in np.arange(1,self.n):
			self.eigenVectors = np.insert(self.eigenVectors,self.eigenVectors.shape[0], self.eigenVectorDict[k],axis  = 0)

	# helper function for computing coefficient matrix
	# puts coefficient vectors into the associated multiprocessing queue
	def computeCoefficientsHelper(self, i):
		# coefficients are computed by elementwise multiplication of an eigenvector row with the denominators
		self.coefficientsQueue.put([i,np.multiply(self.eigenVectors[i], self.denominators)])

	# computes the coefficients using the coefficients helper function and threadpools
	def computeCoefficients(self):
		# instantiate threadpool to compute coefficients
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)

		# evaluate coefficient vectors using multiprocessing and places them into the associated multiprocessing queue
		pool.map(self.computeCoefficientsHelper, [*range(self.n)])

		# gets the values from the multiprocessing queue and places them into the associated dictionary
		# this ensures data is entered in the correct order and protects against racing
		for vector in range(self.n):
			print('getting coefficient: ', vector)
			temp = self.coefficientsQueue.get()
			print(temp[1].shape)
			self.coefficientsDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		# takes coefficient vectors from the dictionary and puts them into a 2d numpy array
		self.coefficients = self.coefficientsDict[0].reshape(1,self.n)
		for k in np.arange(1,self.n):
			self.coefficients = np.insert(self.coefficients,self.coefficients.shape[0], self.coefficientsDict[k],axis  = 0)

	# evaluate a trial of DFGF_S1
	def evaluate(self,trialNum):
		return np.dot(self.coefficients, self.sample[trialNum])

<<<<<<< HEAD
	# helper function to put results of a trial evualtion into the associated multiprocessing queue
=======
>>>>>>> 520c17a8040026844910a9aaee8b96176415d090
	def computeTrial(self, trialNum):
		print("computing trial: ", trialNum)
		self.trialDataQueue.put([trialNum, self.evaluate(trialNum)])

	# function to evaluate all trials using multiprocessing
	def runTrials(self):
<<<<<<< HEAD

=======
>>>>>>> 520c17a8040026844910a9aaee8b96176415d090
		# python multiprocessing
		print("computing trials")
		# instantiate threadpool
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		# evaluate trials
		pool.map(self.computeTrial, [*range(self.numTrials)])

		# gets the trial data from the associated multiprocessing queue and places them into the associated dictionary
		# this ensures data is entered in the correct order and protects against racing
		for trial in range(self.numTrials):
			print('getting trial: ', trial)
			temp = self.trialDataQueue.get()
			self.trialDataDict[temp[0]] = temp[1]
		for i in range(self.numTrials):
			self.trialData[i] = self.trialDataDict[i]
		pool.close()
		pool.join()
<<<<<<< HEAD
	# computes the maxima of each trial and places it into a numpy vector
	def computeMaximaVector(self):
		print(maximaVector.shape)
		self.maximaVector = np.mean(self.trialData, axis = 1)
		
	# computes the mean of the maxima vector
	def computeMeanOfMaxima(self):
		self.meanOfMaxima = np.mean(self.maximaVector)
=======
>>>>>>> 520c17a8040026844910a9aaee8b96176415d090
