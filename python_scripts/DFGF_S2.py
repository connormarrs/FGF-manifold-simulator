import numpy as np
import random as rand
import multiprocessing as mp
import DFGF
from Laplace_S2 import LaplaceS2

class DFGF_S2(DFGF.DFGF):
    eigenVals = []
    eigenVects = []
    grid = []
    coefficients = {}
    gaussianVector = []
    trialData = {}
    npTrialData = []
    maxima = {}
    meanOfMaxima = 0
    
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
            gaussianVector.append([])
            for i in range(numPoints):
                gaussianVector.append(rand.normal)
        
        
        self.computeCoeffs()
        
    def computeCoeffs():
        self.pool.map(computeCoefficientVector, range(numPoints))
        
        print("Coefficients length is off by: ")
        print(len(coefficients) - numPoints)
        
        
    def computeCoefficientVector(r):
        coeffs_r = []
        
        for i in range(numPoints):
            coeffs_r.append(self.computeCoefficientPoint(r,i))
            
        self.coefficients[r] = coeffs_r
    
    def computeCoefficientPoint(r,i):
        return eigenVects[numPoints - 1][i][r] / (eigenVals[numPoints - 1][i] ** s)
    
    def evaluatePoint(i, sampleVector):
        result = 0
        
        for j in range(numPoints):
            result = result + coefficients[i][j] * sampleVector[j]
            
        return result
    
    def evaluate(r):
        evaluations = {}
        
        for i in range(numPoints):
            evaluations[i] = (self.grid[i][0], self.grid[i][1], evaluatePoint(i, self.gaussianVector[i]))
            
        self.trialData[r] = evaluations
    
    def runTrials():
        self.pool.map(evaluate, range(numTrials))
        
        self.npTrialData = np.array(list(self.trialData.items()))
        self.computeMaxima()
        self.computeEmpMean()
        
        print("Trial data length is off by: ")
        print(len(self.trialData) - numTrials)
        
    def computeVectorMax(r):
        data = self.trialData[r]
        M = data[0]
        for i in range(1,len(data)):
            if M < data[i]:
                M = data[i]
        
        maxima[r] = M
        
    def computeMaxima():
        self.pool.map(computeVectorMax, range(len(self.trialData)))
        
        
    def computeEmpMean():
        return sum(maxima) / len(maxima)
                      
    
    def getGrid():
        return self.grid
                      
    def getCoefficients():
        return self.coefficients
    
    def getTrialData():
        return self.trialData
    
    def npTrialData():
        return self.npTrialData
                      
    def getMaxima():
        return self.maxima
                      
    def getMeanOfMaxima():
        return self.meanOfMaxima