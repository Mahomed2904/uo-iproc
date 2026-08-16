import numpy as np
import math

e = math.e
def S(img,sigma):
    height = img.shape[0]
    width = img.shape[1]
    Length_more_half = np.count_nonzero(img > (0.5 * 255))
    a = Length_more_half/(height * width)
    FinalS = (1 + e ** (-32 * (a - sigma))) ** -1
    return FinalS


def ThreeAtomsphericLightFusion(AtomsphericLightOne,AtomsphericLightTwo,AtomsphericLightThree,img):
    FialAtomsphericLightFusion = np.zeros(3)
    for i in range(0,3):
        Max = np.max([AtomsphericLightOne[i],AtomsphericLightTwo[i],AtomsphericLightThree[i]])
        Min = np.min([AtomsphericLightOne[i],AtomsphericLightTwo[i],AtomsphericLightThree[i]])
        alpha = S(img[:,:,i],sigma = 0.2)
        # print('alpha',alpha)
        AtomsphericLightFusion = alpha * Max +  (1-alpha) * Min
        FialAtomsphericLightFusion[i]= AtomsphericLightFusion
    return FialAtomsphericLightFusion

