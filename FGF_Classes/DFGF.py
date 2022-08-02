import numpy as np
<<<<<<< HEAD
import math
import multiprocessing as mp
=======
>>>>>>> 6ab0b1c058834bda88eb4e361321efb9f94346f3

class DFGF:
	#parameters for the simulation
	s = 0.0
	n = 0
	numTrials = 0
<<<<<<< HEAD
	isDirichlet  = True
	trialDataQueue = mp.Queue()
	eigenVectorQueue = mp.Queue()
	coefficientsQueue = mp.Queue()
	eigenVectorDict = {}
=======
	isDirichlet = None
	useThreads = None
>>>>>>> 6ab0b1c058834bda88eb4e361321efb9f94346f3
	def __init__(self):
		pass
	#eigenvalues and eigenvectors for the discrete fractional gaussian field
	#their specific sizes will change depening on the degree of the approximation 
	#and the dimension that is being approximated.
	eigenValues = np.array((n,1))
	eigenVectors = np.array((n,n))
	denominators = np.array((n,1))

	#coefficients are calculated using the eigenvectors and eigenvalues
	coefficients = np.array((n,n))
	coefficientsDict = {}

	#ndarray of iid standard normals
	sample = None

	#data resulting from DFGF calculations
	trialData = {}
	maximaVector = None
	meanOfMaxima = 0

	########################################
	#           Getters & Setters          #
	########################################

	def getTrialData(self):
		return self.trialData
	def getMaximaVector(self):
		return self.maximaVector
	def getMeanOfMaxima(self):
		return self.meanOfMaxima
	def getCoefficients(self):
		return self.coefficients
	def getEigenValues(self):
		return self.eigenValues
	def getEigenVectors(self):
		return self.eigenVectors
	def getSample(self):
		return self.sample
	def setSample(self, sample):
		self.sample = sample
	def setCoefficients(self, coeffs):
		self.ceofficients = coeffs
	def setEigenValues(self, vals):
		self.eigenValues = vals
	def setEigenVectors(self, vecs):
		self.eigenVectors = vecs

	def reuseArrays(self, eigenValues, eigenVectors, sample):
		self.setEigenValues(eigenValues)
		self.setEigenVectors(eigenVectors)
		self.setSample(sample)
