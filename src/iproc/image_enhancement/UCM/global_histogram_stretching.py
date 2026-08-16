import numpy as np

def histogram_r(r_array,height, width):
    length = height * width
    R_rray = np.sort(r_array.flatten())
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    below = r_array < I_min
    above = r_array > I_max
    mid = ~below & ~above

    array_Global_histogram_stretching = np.zeros((height, width))
    array_Global_histogram_stretching[below] = I_min
    array_Global_histogram_stretching[above] = 255
    array_Global_histogram_stretching[mid] = (
        (r_array[mid] - I_min) * ((255 - I_min) / (I_max - I_min))
    ).astype(int) + I_min
    return array_Global_histogram_stretching

def histogram_g(r_array,height, width):
    length = height * width
    R_rray = np.sort(r_array.flatten())
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    above = r_array > I_max
    mid = (r_array >= I_min) & ~above

    array_Global_histogram_stretching = np.zeros((height, width))
    array_Global_histogram_stretching[above] = 255
    array_Global_histogram_stretching[mid] = (
        (r_array[mid] - I_min) * (255 / (I_max - I_min))
    ).astype(int)
    return array_Global_histogram_stretching

def histogram_b(r_array,height, width):
    length = height * width
    R_rray = np.sort(r_array.flatten())
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])

    above = r_array > I_max
    mid = (r_array >= I_min) & ~above

    array_Global_histogram_stretching = np.zeros((height, width))
    array_Global_histogram_stretching[above] = I_max
    array_Global_histogram_stretching[mid] = (
        (r_array[mid] - I_min) * (I_max / (I_max - I_min))
    ).astype(int)
    return array_Global_histogram_stretching

def stretching(img):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
    img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
    img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
    return img


