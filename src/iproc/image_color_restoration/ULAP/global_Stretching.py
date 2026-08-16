import numpy as np

def global_stretching(img_L):
    height = len(img_L)
    width = len(img_L[0])
    length = height * width
    R_rray = np.sort(img_L.flatten())
    I_min = R_rray[int(length / 2000)]
    I_max = R_rray[-int(length / 2000)]

    below = img_L < I_min
    above = img_L > I_max
    mid = ~below & ~above

    array_Global_histogram_stretching_L = np.zeros((height, width))
    array_Global_histogram_stretching_L[above] = 1
    array_Global_histogram_stretching_L[mid] = (img_L[mid] - I_min) * (1 / (I_max - I_min))
    return array_Global_histogram_stretching_L














# def histogram_r(r_array,height, width):
#     length = height * width
#     R_rray = []
#     for i in range(height):
#         for j in range(width):
#             R_rray.append(r_array[i][j])
#     R_rray.sort()
#     I_min = int(R_rray[int(length / 20000)])
#     I_max = int(R_rray[-int(length / 20000)])
#     array_Global_histogram_stretching = np.zeros((height, width))
#     for i in range(0, height):
#         for j in range(0, width):
#             if r_array[i][j] < I_min:
#                 p_out = r_array[i][j]
#                 array_Global_histogram_stretching[i][j] = p_out
#             elif (r_array[i][j] > I_max):
#                 p_out = r_array[i][j]
#                 array_Global_histogram_stretching[i][j] = p_out
#             else:
#                 p_out = int((r_array[i][j] - I_min) * (
#                     (255) / (I_max - I_min)))
#                 array_Global_histogram_stretching[i][j] = p_out
#     return (array_Global_histogram_stretching)