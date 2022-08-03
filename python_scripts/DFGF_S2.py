import numpy as np
import random as rand
import multiprocessing as mp
import DFGF
#from Laplace_S2 import LaplaceS2

class DFGF_S2(DFGF.DFGF):
    eigenVals = []
    eigenVects = []
    grid = []
    coefficients = {}
    gaussianVector = []
    trialData = {}
    npTrialData = []
    maxima = {}
    meanOfMaxima = 0.0
    
    def __init__(self, numPoints, s, numTrials, eigenVals, eigenVects, grid):
        self.n = numPoints
        self.s = s
        self.numTrials = numTrials
        self.eigenVals = eigenVals
        self.eigenVects = eigenVects
        #convert cartesian grid into spherical coordinates:
        for i in range(len(grid)):
            self.grid.append([np.arctan(grid[i][1] / grid[i][0]) * np.sign(x), np.arccos(grid[i][2])])
        
        #self.num_workers = mp.cpu.count()
        self.pool = mp.pool()
        #self.taskQueue = mp.Queue()
        #pool.map(self.function, list_of_args (e.g. range(self.numTrials)))
        
        for r in range(numTrials):
            self.gaussianVector.append([])
            for i in range(numPoints):
                self.gaussianVector[i].append(rand.gauss(0,1))
        
        
        self.computeCoeffs()
        
    def computeCoeffs(self):
        self.pool.map(self.computeCoefficientVector, range(self.numPoints))
        
        print("Coefficients length is off by: ")
        print(len(self.coefficients) - self.numPoints)
        
        
    def computeCoefficientVector(self, r):
        coeffs_r = []
        
        for i in range(self.numPoints):
            coeffs_r.append(self.computeCoefficientPoint(r,i))
            
        self.coefficients[r] = coeffs_r
    
    def computeCoefficientPoint(self, r,i):
        return self.eigenVects[self.numPoints - 1][i][r] / (self.eigenVals[self.numPoints - 1][i] ** s)
    
    def evaluatePoint(self, i, sampleVector):
        result = 0
        
        for j in range(self.numPoints):
            result = result + self.coefficients[i][j] * sampleVector[j]
            
        return result
    
    def evaluate(self, r):
        evaluations = []
        
        for i in range(self.numPoints):
            evaluations.append([self.grid[i][0], self.grid[i][1], self.evaluatePoint(i, self.gaussianVector[i])])
            
        self.trialData[r] = evaluations
    
    def runTrials(self):
        self.pool.map(self.evaluate, range(self.numTrials))
        
        self.npTrialData = np.array(list(self.trialData.items()))
        self.computeMaxima()
        self.computeEmpMean()
        
        print("Trial data length is off by: ")
        print(len(self.trialData) - self.numTrials)
        
    def computeVectorMax(self, r):
        data = self.trialData[r]
        M = data[0]
        for i in range(1,len(data)):
            if M < data[i]:
                M = data[i]
        
        self.maxima[r] = M
        
    def computeMaxima(self):
        self.pool.map(self.computeVectorMax, range(len(self.trialData)))
        
        
    def computeMeanOfMaxima(self):
        self.meanOfMaxima = sum(self.maxima) / len(self.maxima)
                      
    
    def getGrid(self):
        return self.grid
                      
    def getCoefficients(self):
        return self.coefficients
    
    def getTrialData(self):
        return self.trialData
    
    def npTrialData(self):
        return self.npTrialData
                      
    def getMaxima(self):
        return self.maxima
                      
    def getMeanOfMaxima(self):
        return self.meanOfMaxima