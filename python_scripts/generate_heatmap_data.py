import multiprocessing as mp
import numpy as np
import DFGF_S1

from python_scripts.DFGF_S1 import DFGF_S1

###################################################################
#                   Set up Parameters for the Run
###################################################################
# create Queue from mp to store threads
output_queue=mp.Queue()
outputDict = {}

# instantiate s, n values
n_vals=np.array([100, 1000, 10000, 50000, 100000])
epsilon = 0.005
s_vals=np.array([0.0, 0.225, 0.25-epsilon, 0.25, 0.25+epsilon, 0.275, 0.5, 1.0])
numTrials = 2

# flags to pass in
dirichlet=True

###################################################################
#                         Generate the Data
###################################################################
# instantiate one initial DFGF to generate the eigenvalues and eigenvectors
initial_DFGF = DFGF_S1.DFGF_S1(s_vals[0], n_vals[0], numTrials, dirichlet, True)

# function that should take in a tuple (nval, sval)
def sampleDFGF(inputs):
    (nval, sval) = inputs
    # if we can, just use the old dfgf object
    if nval==n_vals[0] and sval==s_vals[0]:
        DFGF_object = initial_DFGF
    else:
        # reuse old sample, eigenvectors, eigenvalues to save compute power
        DFGF_object = DFGF_S1.DFGF_S1(s_vals[0], n_vals[0], numTrials, dirichlet, False)
        DFGF_object.setParams(
            initial_DFGF.getSample(), 
            initial_DFGF.getEigenVectors(),
            initial_DFGF.getEigenValues()
        )

    # run the simulation - multithread this
    DFGF_object.runTrials()
    output_queue.put([
        nval, sval, DFGF_object.getTrialData()
    ])

# create a pool of threads and use map to launch a thread for each n value
pool_object=mp.Pool(mp.cpu_count())
# format inputs into array of tuples
inputs=[(n,s) for n in n_vals for s in s_vals]
# pass in objects as inputs
pool_object.map(sampleDFGF, [*inputs])

###################################################################
#                         Write the Data to CSVs
###################################################################
# loop through 
for index in range(n_vals.size*s_vals.size):
    temp = output_queue.get()
    outputDict[(temp[0], temp[1])] = temp[2]

# loop through and data from dictionary to numpy array nvalues over
for nval in n_vals:
    for sval in s_vals:
        filename='output/'+str(nval)+'_'+"{:.6f}".format(sval)+'.csv'
        np.savetxt(filename, outputDict[(nval,sval)], delimiter=",")
