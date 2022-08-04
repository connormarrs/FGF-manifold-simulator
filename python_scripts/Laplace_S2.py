import numpy as np
import random as rand

#Class for precomputing random grid points and eigensystem of graph laplacians
#For use in DFGF_S2 driver

class Laplace_S2:
    
    numPoints = 0         #number of points on the sphere that should be generated 
    step_size = 0         #increment between size of laplacians
    grid = []             #list of all randomly generated points
    bandwidths = []       #list of parameters that determine edge weights
    boundingParam = 0.0   #how far the spectral gap should stay bounded from 0
    bandWidthParam = 0.0  #constant which helps determine size of bandwidths
    eigenVals = []        #list of lists of eigenvalues for each laplacian
    eigenVects = []       #list of lists of eigenvectors for each laplacian
    spectralGaps = []
    computeCounts = []
    
    #set parameters and initialize grid and bandwidths:
    def __init__(self, numPoints, step, boundingParam, bandwidthParam):
        self.numPoints = numPoints
        self.step_size = step
        self.boundingParam = boundingParam
        self.bandwidthParam = bandwidthParam
        
        self.makeGrid()
        self.setBandwidths()
        
    #chooses a single random point uniformly on the sphere
    def makePoint(self):
        theta = 2 * np.pi * rand.uniform(0,1)
        phi = np.arcsin(1 - 2 * rand.uniform(0,1))
        
        return [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
    
    #fills list of randomly chosen points on sphere
    def makeGrid(self):
        for i in range(self.numPoints):
            self.grid.append(self.makePoint())
            
    #initializes the bandwidth parameters for each laplacian
    def setBandwidths(self):
        for i in range(self.numPoints + 1):
            self.bandwidths.append(0)

        for k in range(self.step_size, self.numPoints+1, self.step_size):
            self.bandwidths[k] = self.bandwidthParam / (i ** (1 / 16))
            
    #helper function for computinig the l-th legendre polynomial at input x
    def computeLegendre(self, x, l):
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
    
    #Calculates the edge weights needed between each pair of points on the sphere
    def computeEdgeWeight(self, p1, p2, t, n):
        cosine = np.cos(np.dot(p1,p2))
        
        term = 1
        partSum = term
        l = 1
        
        while term >= 1e-8:
            term = (2*l + 1) * np.exp(-l * (l+1) * t) * self.computeLegendre(cosine, l)
            partSum = partSum + term
            l = l + 1
            
        return partSum / (n * t * 4 * np.pi)
    
    #Fills laplacian matrix for a certain number of points k
    def computeLaplacian(self, k):
        laplacian = []
        
        for i in range(k):
            laplacian_i = []
            laplacian.append(laplacian_i)
            
            for j in range(k):
                if i != j:
                    laplacian[i].append(-1 * self.computeEdgeWeight(self.grid[i], self.grid[j], self.bandwidths[k], k))
                    
                else:
                    laplacian[i].append(0)
                    
            
        for i in range(k):
            laplacian[i][i] = -1 * sum(laplacian[i])
            
        return laplacian
    
    #Calculates 2nd smallest eigenvalue of laplacian; 
    #this value should stay bounded away from 0
    def findSpectralGap(self, eVals):

        mins = [eVals[0], eVals[1]]
        
        currentMin = eVals[0]
        prevMin = eVals[1]
        
        for i in range(2, len(eVals)):
            if eVals[i] < mins[0]:
                mins[0] = eVals[i]
            elif eVals[i] < mins[1]:
                mins[1] = eVals[i]
        
        return max(mins)
    
    #Calculates eigenvectors and eigenvalues of each laplacian
    def computeEigenSystems(self):
        for k in range(self.step_size, self.numPoints + 1, self.step_size):
            computeNewLaplacian = True
            
            computeCount = 1
            while computeNewLaplacian:
                laplacian = self.computeLaplacian(k)
                
                [eVals, eVects] = np.linalg.eig(laplacian)

                spectralGap = self.findSpectralGap(eVals)
                
                #make sure the spectral gap stays bounded away from 0;
                #if not, increase bandwidth and try again
                if spectralGap > self.boundingParam:
                    computeNewLaplacian = False
                    self.spectralGaps.append(spectralGap)
                    self.computeCounts.append(computeCount)
                else:
                    computeCount += 1
                    self.bandwidths[k] = 0.9 * self.bandwidths[k] + 0.1 * self.bandwidths[k-1]
                    print(str(computeCount)+", "+str(spectralGap)+", "+str(k))
                    print(eVals)
                    
            self.eigenVals.append(eVals)
            self.eigenVects.append(eVects)

    def getSpectralGaps(self):
        return self.spectralGaps

    def getEigenValues(self):
        return self.eigenVals

    def getEigenVectors(self):
        return self.eigenVects

    def getComputeCounts(self):
        return self.computeCounts