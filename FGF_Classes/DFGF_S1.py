import DFGF
class DFGF_S1(DFGF.DFGF):
	
	def __init__(self, sRange, n, numTrials, isDirchlet):
		#set parameters
		self.sRange = sRange
		self.n=n
		self.numTrials = numTrials
		self.isDirchlet = isDirchlet
		self.eigenVals = np.arange(1,math.ceil(n/2))
		self.computeSample()
		self.computeEigenValues()
		self.computeEigenVectors()
		self.computeCoefficients()


	def computeSample(self):
		self.sample = self.rng.standard_normal((n,numTrials))

	def computeEigenValues(self):
		self.eigenVals = self.n**2/(2*np.pi**2)*(1-cos(2*np.pi*np.cos(self.eigenVals)))

	def computeEigenVectorComponent(self, m):
		if m <= 
