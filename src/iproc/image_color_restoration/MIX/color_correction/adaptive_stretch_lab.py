

from iproc.image_color_restoration.MIX.color_correction.rgb2cie import *
from iproc.image_color_restoration.MIX.color_correction.cie2rgb import *
from scipy import stats
from skimage import color


# S_model curve function to stretch 'a' and 'b' values
def S_model_curve(x):
    return x * (1.3**(1-np.abs(x/128)))

# Rayleigh Distriution function to stretch 
def rayleigh_distribution(x,a):
    return (x/a**2)* np.exp(-x**2 / (2*a**2))
    
# Adaptive Stretching of 'L' values based on Rayleigh Distribution to range [0,100]
def adaptive_stretching_L(L, height, width):
    O_min = 0
    O_max = 100
    stretched_L = np.zeros(L.shape, 'float64')

    L_flatten = L.flatten()
    L_flatten.sort()
    mode = stats.mode(L_flatten).mode[0]
    # np.searchsorted(..., side='left') finds the same first-occurrence
    # index as list(L_flatten).index(mode) on this already-sorted array,
    # without the O(n) Python-level linear search.
    mode_index = int(np.searchsorted(L_flatten, mode, side='left'))
    L_min = L_flatten[int(mode_index * 0.001)]
    L_max = L_flatten[int(-(len(L_flatten) - mode_index) * 0.001)]

    # NOTE: the middle (in-range) branch computes `x` (the properly stretched
    # value) but then discards it and assigns the ORIGINAL L[i,j] through
    # unchanged instead — a pre-existing bug in this algorithm, not touched
    # here since this is a performance-only pass; only the below/above
    # branches actually clip to 0/100.
    above = L > L_max
    mid = (L >= L_min) & ~above
    stretched_L[above] = 100
    stretched_L[mid] = L[mid]

    return stretched_L


# Function to stretch 'a' or 'b' values to range [-128,127] according to S-model Curve Function
def adaptive_stretching_ab(a_or_b_values,height,width):
    # S_model_curve is already built from numpy ops (np.abs/**), so it works
    # directly on the whole array — no per-pixel loop needed.
    return S_model_curve(a_or_b_values).astype('float64')

def  LAB_Stretching(img_rgb):

    #Processing of Image
    height = len(img_rgb)
    width = len(img_rgb[0])

    #Converting RGB to CIE Lab
    L,a,b = RGB_to_CIE_Lab(img_rgb)


    #Stretching
    L_stretched = adaptive_stretching_L(L, height, width)

    a_stretched = adaptive_stretching_ab(a, height, width)

    b_stretched = adaptive_stretching_ab(b, height, width)

    #Merging Back
    lab_img = np.zeros((height, width, 3), 'float64')
    
    lab_img[:, :, 0] = L_stretched
    lab_img[:, :, 1] = a_stretched
    lab_img[:, :, 2] = b_stretched        
    
    img_rgb_color_correction = CIE_Lab_to_RGB(lab_img)

    return img_rgb_color_correction


