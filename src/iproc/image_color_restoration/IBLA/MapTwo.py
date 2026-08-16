import numpy as np
from iproc.image_color_restoration.IBLA.getOneChannelMax import getMaxChannel

def getGBMAxChannel(img):
    # Channels 0,1 only, matching the original's range(0,2) exactly (same
    # BGR-assuming index choice as GBdehazingRCorrection/NewOpticalModel —
    # preserved as-is, out of scope for this performance-only pass).
    return img[:, :, :2].max(axis=2).astype(np.float64)


def R_minus_GB(img,blockSize,R_map):
    img = getGBMAxChannel(img)
    mip_map =  R_map - getMaxChannel(img, blockSize)
    return mip_map
