import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import math

class DFGF:
	#parameters for the simulation
	sRange = []
	n = 0
	numTrials = 0
	isDirichlet  = True

	#eigenvalues and eigenvectors for the discrete fractional gaussian field
	#their specific sizes will change depening on the degree of the approximation 
	#and the dimension that is being approximated.
	eigenValues = None
	eigenVectors = None

	#coefficients are calculated using the eigenvectors and eigenvalues
	coefficients = None

	#ndarray of iid standard normals
	rng = np.random.default_rng(1020304050)
	sample = None

	#data resulting from DFGF calculations
	trialData = None
	maximaVector = None
	meanOfMaxima = 0





	###############################
	#           Getters           #
	###############################

	def getTrialData(self):
		return self.trialData

	def getMaximaVector(self):
		return self.maximaVector

	def getMeanOfMaxima(self):
		return self.meanOfMaxima