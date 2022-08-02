import numpy as np
import random as rand

class LaplaceS2:
    
    grid = []
    bandwidths = []
    eigenVals = []
    eigenVects = []
    
    def __init__(self, numPoints, boundingParam, bandwidthParam):
        self.numPoints = numPoints
        self.boundinParam = boundingParam
        self.bandwidthParam = bandwidthParam
        
        self.makeGrid()
        self.setBandwidths()
        
    def makePoint():
        theta = 2 * np.pi * random.uniform(0,1)
        phi = np.arcsin(1 - 2 * random.uniform(0,1))
        
        return [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
    
    def makeGrid():
        for i in range(numPoints):
            grid.append(self.makePoint())
            
    def setBandwidths():
        for i in range(1, numPoints+1):
            bandwidths.append(bandwidthParam / (i ** 0.25))
            
    def computeLegendre(x, l):
        if l == 0:
            return 1
        elif l == 1:
            return x
        else:
            prepreValue = 1
            preValue = x
            value = 0
            
            for i in range(2, l):
                holdValue = value
                holdPreValue = preValue
                
                value = ((2*i - 1) * x * preValue / i) - (i - 1) * prepreValue
                preValue = holdValue
                prepreValue = holdPreValue
            
            return value
        
    def computeEdgeWeight(p1, p2, t, n):
        cosine = np.cos(np.dot(p1,p2))
        
        term = 1;
        partSum = term
        l = 1
        
        while term >= 1e-8:
            term = (2*l + 1) * np.exp(-l * (l+1) * t) * self.computeLegendre(cosine, l)
            partSum = partSum + term
            l = l + 1
            
        return partSum / (n * t * 4 * np.pi)
    
    def computeLaplacian(k):
        laplacian = []
        
        for i in range(k):
            laplacian_i = []
            laplacian.append(laplacian_i)
            
            for j in range(k):
                if i != j:
                    laplacian[i].append(computeEdgeWeight(grid[i], grid[j], bandwidths[k], k))
                    
                else:
                    laplacian[i].append(0)
                    
            
        for i in range(k):
            laplacian[i][i] = -sum(laplacian[i])
            
        return laplacian
    
    def findSpectralGap(eVals):
        
        currentMin = eVals[0]
        prevMin = None
        
        for i in range(1, len(eVals)):
            if eVals[i] < currentMin:
                prevMin = currentMin
                currentMin = eVals[i]
                
        return prevMin
    
    def computeEigensystems():
        for k in range(1, numPoints + 1):
            bool computeNewLaplacian = True
            
            while computeNewLaplacian:
                laplacian = computelaplacian(k)
                
                [eVals, eVects] = np.linalg.eig(laplacian)
                
                if self.findSpectralGap(eVals) > self.boundingParam:
                    computeNewLaplacian = False
                else:
                    bandwidths[k] = 0.9 * bandwidths[k] + 0.1 * bandwidths[k-1]
                    
            eigenVals.append(eVals)
            eigenVects.append(eVects)