import numpy as np
import DFGF_S1

###################################################################
#                   Set up Parameters for the Run
###################################################################
# create Queue from mp to store threads

# instantiate s, n values
n_vals = [100, 1000, 5000, 10000, 15000, 20000]
epsilon = 0.005
s_vals = [0.0, 0.225, 0.25-epsilon, 0.25, 0.25+epsilon, 0.275, 0.5, 1.0]
numTrials = 1

# flags to pass in
dirichlet=True

###################################################################
#                         Generate the Data
###################################################################
# loop through and data from dictionary to numpy array nvalues over
for nval in n_vals:
	# instantiate one initial DFGF to generate the eigenvalues and eigenvectors
	initial_DFGF = DFGF_S1.DFGF_S1(s_vals[0], nval, numTrials, dirichlet, True)

	for sval in s_vals:
		if nval==n_vals[0] and sval==s_vals[0]:
			DFGF_object = initial_DFGF
		else:
			# reuse old sample, eigenvectors, eigenvalues to save compute power
			DFGF_object = DFGF_S1.DFGF_S1(sval, nval, numTrials, dirichlet, False)
			DFGF_object.setParams(
			    initial_DFGF.getSample(), 
			    initial_DFGF.getEigenValues(),
			    initial_DFGF.getEigenVectors()
			)
			
			# run the simulation - multithread this
			DFGF_object.runTrials()

		filename='output/'+str(nval)+'_'+"{:.6f}".format(sval)+'.csv'
		np.savetxt(filename, DFGF_object.getTrialData(), delimiter=",")
