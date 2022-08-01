import matplotlib.pyplot as plt
import numpy as np
import math
import multiprocessing as mp

class DFGF:
	#parameters for the simulation
	s = 0.0
	n = 0
	numTrials = 0
	isDirichlet  = True
	trialDataQueue = mp.Queue()
	eigenVectorQueue = mp.Queue()
	coefficientsQueue = mp.Queue()
	eigenVectorDict = {}
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
	rng = np.random.default_rng(1020304050)
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
	def setSample(self, sample):
		self.sample = sample
	def setCoefficients(self, coeffs):
		self.ceofficients = coeffs
	def setEigenValues(self, vals):
		self.eigenValues = vals

	def reuseArrays(self, eigenValues, eigenVectors, sample):
		self. eigenValues = setEigenValues
		self.eigenVectors = getEigenVectors
		self.sample = sample
