import DFGF
import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc

# sets the number of threads available to python based on system specs
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())

class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, s, n, numTrials, isDirichlet, compute):
		#set parameters
		self.s = s
		self.n = n
		self.numTrials = numTrials
		self.isDirichlet = isDirichlet

		self.eigenValues = np.zeros((self.n,1))
		self.denominators = np.zeros((self.n,1))
		self.eigenVectors = np.zeros((self.n,self.n))
		self.coefficients = np.zeros((self.n,self.n))
		self.trialData = np.zeros((self.numTrials, self.n), dtype = float)

		if compute:
			self.computeSample()
			self.computeEigenValues()
			self.computeEigenVectors()
			self.computeCoefficients()

	# computes the samples of random vectors and stores them as a numpy array in samples
	def computeSample(self):
	# uses numpy operations to compute the vector of eigenvalues
	# the eigenvalues can be passed to other instances of DFGF_S1 with the same n value
		dist = qmc.MultivariateNormalQMC(
			mean=np.zeros(self.n-1)
		)

		# generate an array of numTrials samples
		self.sample = np.array(
			dist.random(self.numTrials)
		)

	# compute the eigenvalues and stores them for future use
	def computeEigenValues(self):
		tempVector = np.arange(1, math.ceil((self.n-1)/2)+1)
		# first calculate the eigenvalues associated with cosine eigenvectors
		self.eigenValues = self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n))
		# create temp vector to help with computation of the sine associated eigenvalues
		tempVector = np.arange(1, math.floor((self.n-1)/2)+1)

		# append the calculated eigenvalues to the eigenvalue vector
		self.eigenValues = np.append(self.eigenValues, self.n**2/(2*np.pi**2)*(1-np.cos(2*np.pi*(tempVector)/self.n)))
		# compute the denominators for the terms in the sum for calculating the value of the DFGF on S1
		self.denominators = np.power(self.eigenValues, -self.s)

	# computes the eigenvector for DFGF_S1
	# the eigenvectors can be passed to other instances of DFGF_S1 with the same n value
	def computeEigenVector(self, k):
		if self.isDirichlet == False:
			tempEigenVectorSines = np.arange(1,math.floor((self.n-1)/2)+1)
			tempEigenVectorCosines = np.arange(1, math.ceil((self.n-1)/2)+1)
			sines = (np.sin((2*np.pi*k/self.n) * tempEigenVectorSines))
			cosines = np.cosine((2*np.pi*k/self.n) * tempEigenVectorCosines)

			# places the reseult of the caluclation into the multiprocessing queue
			# this allows multiple threads to share data
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n-1)])

		# evaluates the dirichlet case
		elif self.isDirichlet == True:
			tempEigenVectorSines = np.arange(1,math.floor((self.n-1)/2)+1)
			sines = (np.sin(2*np.pi*k/self.n * tempEigenVectorSines))
			# returns zeros instead of the cosine eigenfunctions
			cosines = np.zeros((math.ceil((self.n-1)/2)))
			self.eigenVectorQueue.put([k,np.append(cosines, sines).reshape(1,self.n-1)])

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
			temp = self.eigenVectorQueue.get()
			self.eigenVectorDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		# takes eigenvectors from the eigenVectorDict and places them into a 2d numpy array
		self.eigenVectors = self.eigenVectorDict[0].reshape(1,self.n-1)
		for k in np.arange(1,self.n):
			self.eigenVectors = np.append(self.eigenVectors, self.eigenVectorDict[k],axis  = 0)

	# helper function for computing coefficient matrix
	# puts coefficient vectors into the associated multiprocessing queue
	def computeCoefficientsHelper(self, i):
		# coefficients are computed by elementwise multiplication of an eigenvector row with the denominators
		self.coefficientsQueue.put([i,np.multiply(self.eigenVectors[i], self.denominators).reshape(1,self.n-1)])

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
			temp = self.coefficientsQueue.get()
			self.coefficientsDict[temp[0]] = temp[1]
		pool.close()
		pool.join()

		# takes coefficient vectors from the dictionary and puts them into a 2d numpy array
		self.coefficients = self.coefficientsDict[0].reshape(1,self.n-1)
		for k in np.arange(1,self.n):
			self.coefficients = np.append(self.coefficients, self.coefficientsDict[k],axis  = 0)

	# evaluate a trial of DFGF_S1
	def evaluate(self,trialNum):
		return np.dot(self.coefficients, self.sample[trialNum])

	# helper function to put results of a trial evualtion into the associated multiprocessing queue
	def computeTrial(self, trialNum):
		self.trialDataQueue.put([trialNum, self.evaluate(trialNum)])

	# function to evaluate all trials using multiprocessing
	def runTrials(self):
		# python multiprocessing
		# instantiate threadpoo
		num_workers = mp.cpu_count()
		pool  = mp.Pool(num_workers)
		# evaluate trials
		pool.map(self.computeTrial, [*range(self.numTrials)])

		# gets the trial data from the associated multiprocessing queue and places them into the associated dictionary
		# this ensures data is entered in the correct order and protects against racing
		for trial in range(self.numTrials):
			temp = self.trialDataQueue.get()
			self.trialDataDict[temp[0]] = temp[1]
		for i in range(self.numTrials):
			self.trialData[i] = self.trialDataDict[i]
		pool.close()
		pool.join()
	# computes the maxima of each trial and places it into a numpy vector
	def computeMaximaVector(self):
		temp = 0.5*(self.trialData[:,math.ceil(self.n/2)]+self.trialData[:,math.floor(self.n/2)]).reshape(self.numTrials, 1)
		maximaCandidates = self.trialData[:,0:math.floor(self.n/2)]
		maximaCandidates = np.append(maximaCandidates, temp, axis = 1)
		self.maximaVector = np.amax(maximaCandidates, axis = 1)


	def computeMaxOverS1(self):
		self.maximaVector = np.amax(self.trialData, axis = 1)
	# computes the mean of the maxima vector
	def computeMeanOfMaxima(self):
		self.meanOfMaxima = np.mean(self.maximaVector)

	def setParams(self, sample, eigenValues, eigenVectors):
		self.setSample(sample)
		self.setEigenValues(eigenValues)
		self.setEigenVectors(eigenVectors)
		self.denominators = np.power(self.eigenValues, -self.s)
		self.computeCoefficients()
