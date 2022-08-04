import numpy as np
import math
import random as rand
import multiprocessing as mp
import DFGF

#from Laplace_S2 import LaplaceS2

class DFGF_S2(DFGF.DFGF):
    grid = []
    coefficients = {}
    gaussianVector = []
    trialData = {}
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
        
        #self.num_workers = mp.cpu.count()
        #self.pool = mp.Pool()
        #self.taskQueue = mp.Queue()
        #pool.map(self.function, list_of_args (e.g. range(self.numTrials)))
        
        for r in range(self.numTrials):
            self.gaussianVector.append([])
            for i in range(self.numPoints):
                self.gaussianVector[r].append(rand.gauss(0,1))
        
        self.computeCoeffs()

    def check_process(self):
        print('starting '+mp.current_process().name)

    def computeCoefficientPoint(self, r, i):
        return self.eigenVects[i][r] / math.pow(self.eigenVals[i], self.s)

    def computeCoefficientVector(self, r):
        coeffs_r = []
        #print(r)
        
        for i in range(1, self.numPoints):
            coeffs_r.append(self.computeCoefficientPoint(r,i))

        self.coefficientsQueue.put([r, coeffs_r])
        print(self.coefficientsQueue.qsize())
        
    def computeCoeffs(self):
        numWorkers = mp.cpu_count()
        pool = mp.Pool(numWorkers, initializer=self.check_process)
        pool.map(self.computeCoefficientVector, [*range(self.numPoints)])

        print(self.coefficientsQueue.qsize())
        print("jobs completed")

        for vector in range(self.numPoints):
            temp = self.coefficientsQueue.get()
            print(temp)
            self.coefficientsDict[temp[0]] = temp[1]
            print("vector #"+str(vector)+" added")

        pool.close()
        pool.join()
        
        print("Coefficients length is off by: ")
        print(len(self.coefficientsDict) - self.numPoints)
    
    def evaluatePoint(self, i, sampleVector):
        result = 0
        
        for j in range(self.numPoints - 1):
            result = result + self.coefficients[i][j] * sampleVector[j]
            
        return result
    
    def evaluate(self, r):
        print(self.numTrials - r)
        evaluations = []
        
        for i in range(self.numPoints):
            evaluations.append([self.evaluatePoint(i, self.gaussianVector[r])])
            
        self.trialData[r] = evaluations
    
    def runTrials(self):
        pool = mp.Pool()
        pool.map(self.evaluate, [*range(self.numTrials)])
        pool.close()
        pool.join()
        
        self.npTrialData = np.array(list(self.trialData.items()))
        self.computeMaxima()
        self.computeMeanOfMaxima()
        
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
        pool = mp.Pool()
        pool.map(self.computeVectorMax, range(len(self.trialData)))
        pool.close()
        pool.join()
        
        
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