import numpy as np


def stretching(img):
    # The original assigns each stretched value back into `img` in place, so
    # each assignment gets coerced to whatever dtype `img` actually is as it
    # goes (each channel's Min/Max are read once up front, so this never
    # contaminates the channel's own stats — only the final returned values
    # are affected). This function is called both directly on the uint8
    # interface input (ICM/RGHS-style callers) AND on a float64 intermediate
    # here (RGB_equalisation's output) — so the result must preserve
    # `img`'s own dtype rather than assuming uint8.
    result = img.copy()
    img_f = img.astype(np.float64)
    for k in range(0, 3):
        Max_channel = np.max(img[:, :, k])
        Min_channel = np.min(img[:, :, k])
        result[:, :, k] = (img_f[:, :, k] - Min_channel) * 255 / (Max_channel - Min_channel)
    return result



