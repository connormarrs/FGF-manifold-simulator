import sys
import numpy as np
import math
import multiprocessing as mp
import os
from scipy.stats import qmc

sys.path.append('../FGF_Classes')

import DFGF_S1




# driver file to compute the approximate pdf of the maxima of the DFGF on S1