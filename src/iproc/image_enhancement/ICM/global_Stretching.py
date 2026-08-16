import numpy as np

def global_stretching(img_L,height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)

    # Multiply by the reciprocal rather than dividing directly — matches the
    # original's exact operation order; the two aren't always bit-identical
    # in floating point, and this can shift which side of a truncation
    # boundary a pixel lands on downstream.
    return (img_L - I_min) * (1 / (I_max - I_min))

