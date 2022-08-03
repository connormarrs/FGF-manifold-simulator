from scipy.stats import qmc
import multiprocessing as mp
import numpy as np
import csv

compute_distribution=False
compute_expval_max=True

###################################################################
#	    Generate Data to Test Distribution of Max
###################################################################
if compute_distribution:
	nval = 10000
	numTrials = 20000
	
	# instantiate a multivariate normal distribution with SciPy
	dist = qmc.MultivariateNormalQMC(
		mean=np.zeros(nval)
	)
	
	# generate an array of numTrials samples
	sample = np.array(
		dist.random(numTrials)
	)
	
	with open('samples_SCIPY_MultivariateNormalQMC_n_10000_trials_20000.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerows(sample)

###################################################################
#	    Generate Data to Test Distribution of Max
###################################################################
if compute_expval_max:
	# make a queue to hold data from the threads
	output_queue=mp.Queue()
	outputDict = {}
	# make array of n values
	n_start=100
	n_stop=21000
	n_step=50
	numTrials = 4000

	def computeEmpMean(n):
		# compute the sample
		dist = qmc.MultivariateNormalQMC(mean=np.zeros(n))
		sample = np.array(dist.random(numTrials))
		
		# compute the maximum
		output_queue.put([n, np.mean(np.max(sample, axis=1))])

	# make array of n values and array to hold the computed emp means
	n_vals = np.arange(start = n_start, stop = n_stop+n_step, step = n_step)
	emp_mean_array = np.zeros(n_vals.shape)

	# create a pool of threads and use map to launch a thread for each n value
	pool_object=mp.Pool(mp.cpu_count())
	pool_object.map(computeEmpMean, [*n_vals])

	# place objects in dictionary to 
	for i in range(len(n_vals)):
		temp = output_queue.get()
		outputDict[temp[0]] = temp[1]
	for index in range(len(n_vals)):
		# compute the maximum
		emp_mean_array[index] = outputDict[n_vals[index]]

	print(emp_mean_array.size)

	filename='output/empiricalMeans_SCIPY_MultivariateNormalQMC_nstart_'+str(n_start)+'_n_stop_'+str(n_stop)+'_n_step_'+str(n_step)+'.csv'
	np.savetxt(filename, emp_mean_array, delimiter=",")

