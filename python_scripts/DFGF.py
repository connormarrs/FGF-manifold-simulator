import numpy as np
import multiprocessing as mp
class DFGF:
	#parameters for the simulation
	s = 0.0
	n = 0
	numTrials = 0
	isDirichlet  = True

	#ndarray of iid standard normals
	sample = None

	#eigenvalues and eigenvectors for the discrete fractional gaussian field
	#their specific sizes will change depening on the degree of the approximation 
	#and the dimension that is being approximated.
	eigenValues = None
	denominators = None

	eigenVectors = None
	eigenVectorQueue = mp.Queue()
	eigenVectorDict = {}

	#coefficients are calculated using the eigenvectors and eigenvalues
	coefficients = None
	coefficientsQueue = mp.Queue()
	coefficientsDict = {}

	#data resulting from DFGF calculations
	trialData = None
	trialDataQueue = mp.Queue()
	trialDataDict = {}
	maximaVector = np.zeros((numTrials))
	meanOfMaxima = 0.0

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
