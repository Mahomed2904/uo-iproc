import numpy as np
import math
e = math.e

def global_Stretching_ab(a,height, width):
    return a * (1.3 ** (1 - np.abs(a / 128)))
