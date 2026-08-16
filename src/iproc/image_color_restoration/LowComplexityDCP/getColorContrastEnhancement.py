import numpy as np
def ColorContrastEnhancement(sceneRadiance):
    # print('Before_sceneRadiance',sceneRadiance)
    bgrAvg = np.zeros(3)
    sceneRadiance = np.float16(sceneRadiance)
    for k in range(0, 3):
        bgrAvg[k] = np.mean(sceneRadiance[:,:,k])
    # print('bgrAvg',bgrAvg)
    for k in range(1, 3):
        sceneRadiance[:, :, k] = (bgrAvg[0]/bgrAvg[k])*sceneRadiance[:,:,k]
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    # print('sceneRadiance',sceneRadiance)
    # print(np.mean(sceneRadiance[:,:,0]),np.mean(sceneRadiance[:,:,1]),np.mean(sceneRadiance[:,:,2]))
    return sceneRadiance



