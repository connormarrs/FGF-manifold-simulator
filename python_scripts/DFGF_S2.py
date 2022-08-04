import numpy as np
import math
import random as rand
import multiprocessing as mp
import DFGF

#from Laplace_S2 import LaplaceS2

class DFGF_S2(DFGF.DFGF):
    grid = []
    gaussianVector = []
    npTrialData = []
    maxima = {}
    meanOfMaxima = 0.0

    
    def __init__(self, numPoints, s, numTrials, eigenVals, eigenVects, grid):
        self.numPoints = numPoints
        self.s = s
        self.numTrials = numTrials
        self.eigenVals = eigenVals
        print(self.eigenVals)
        self.eigenVects = eigenVects

        #convert cartesian grid into spherical coordinates:
        for i in range(len(grid)):
            self.grid.append([np.arctan(grid[i][1] / grid[i][0]) * np.sign(grid[i][0]), np.arccos(grid[i][2])])
        
        #fill gaussian vector with nstandard ormally-distributed random values
        for r in range(self.numTrials):
            self.gaussianVector.append([])
            for i in range(self.numPoints):
                self.gaussianVector[r].append(rand.gauss(0,1))
        
        self.computeCoeffs()

    def computeCoefficientPoint(self, r, i):
        return self.eigenVects[i][r] / math.pow(self.eigenVals[i], self.s)

    def computeCoefficientVector(self, r):
        coeffs_r = []
        
        for i in range(1, self.numPoints):
            coeffs_r.append(self.computeCoefficientPoint(r,i))

        self.coefficientsQueue.put([r, coeffs_r])
        
    def computeCoeffs(self):
        numWorkers = mp.cpu_count()
        pool = mp.Pool(numWorkers)
        pool.map(self.computeCoefficientVector, [*range(self.numPoints)])

        for vector in range(self.numPoints):
            temp = self.coefficientsQueue.get()
            self.coefficientsDict[temp[0]] = temp[1]

        pool.close()
        pool.join()
    
    def evaluatePoint(self, i, sampleVector):
        result = 0
        print(type(self.coefficientsDict))
        print(type(sampleVector))
        for j in range(self.numPoints - 1):
            result = result + self.coefficientsDict[i][j] * sampleVector[j]
            
        return result
    
    def evaluate(self, r):
        print(self.numTrials - r)
        evaluations = []
        
        for i in range(self.numPoints):
            evaluations.append([self.evaluatePoint(i, self.gaussianVector[r])])
            
        self.trialDataQueue.put([r, evaluations])
    
    def runTrials(self):
        pool = mp.Pool()
        pool.map(self.evaluate, [*range(self.numTrials)])

        for vector in range(self.numTrials):
            temp = self.trialDataQueue.get()
            self.trialDataDict[temp[0]] = temp[1]
        
        pool.close()
        pool.join()
        
        self.npTrialData = np.array(list(self.trialDataDict.items()))
        self.computeMaxima()
        self.computeMeanOfMaxima()
        
    def computeVectorMax(self, r):
        data = self.trialDataDict[r]
        M = data[0]
        for i in range(1,len(data)):
            if M < data[i]:
                M = data[i]
        
        self.maxima[r] = M
        
    def computeMaxima(self):

        for r in range(len(self.trialDataDict)):
            self.computeVectorMax(r)
        
        
    def computeMeanOfMaxima(self):
        self.meanOfMaxima = sum(self.maxima) / len(self.maxima)
                      
    
    def getGrid(self):
        return self.grid
                      
    def getCoefficients(self):
        return self.coefficientsDict
    
    def getTrialData(self):
        return self.trialDataDict
    
    def npTrialData(self):
        return self.npTrialData
                      
    def getMaxima(self):
        return self.maxima
                      
    def getMeanOfMaxima(self):
        return self.meanOfMaxima