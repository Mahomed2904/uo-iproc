import numpy as np

def global_stretching(img_L,height, width):
    length = height * width
    R_rray = np.sort(img_L.flatten())
    I_min = R_rray[int(length / 100)]
    I_max = R_rray[-int(length / 100)]

    below = img_L < I_min
    above = img_L > I_max
    mid = ~below & ~above

    array_Global_histogram_stretching_L = np.zeros((height, width))
    array_Global_histogram_stretching_L[below] = img_L[below]
    array_Global_histogram_stretching_L[above] = img_L[above]
    # Multiply by the reciprocal rather than dividing directly — matches the
    # original's exact operation order.
    array_Global_histogram_stretching_L[mid] = (img_L[mid] - I_min) * (1 / (I_max - I_min))
    return array_Global_histogram_stretching_L

