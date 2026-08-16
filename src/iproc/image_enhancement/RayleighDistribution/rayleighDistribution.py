import numpy as np

e = np.e
esp = 2.2204e-16



def _rayleighLUT(NumPixel, count):
    """
    Builds the 256-entry Rayleigh-CDF-matching lookup table shared by both
    the lower and upper partitions: normalize the value histogram into a
    CDF, then map each cumulative probability through the inverse Rayleigh
    CDF. A pure function of a value histogram — never depends on pixel
    position — so it can be applied as a vectorized lookup instead of a
    per-pixel loop.
    """
    alpha = 0.4
    selectedRange = [0, 255]
    ProbPixel = NumPixel / count
    CumuPixel = np.cumsum(ProbPixel)

    valSpread = selectedRange[1] - selectedRange[0]
    hconst = 2 * alpha ** 2
    vmax = 1 - e ** (-1 / hconst)
    val = vmax * CumuPixel
    val = np.where(val >= 1, val - esp, val)

    temp = np.sqrt(-hconst * np.log(1 - val))
    normalization = temp * valSpread
    return np.where(normalization > 255, 255, normalization)


def uperLower(r, height, width):
    allSize = height * width
    R__middle = np.mean(r)

    # The original sorts nodes by their RAW value (whatever precision `r`
    # actually is — this function is called both on integer channels and,
    # via a caller upstream of this one, on float64 intermediates) and only
    # truncates to int afterward (node.value = np.int(node.value), applied
    # to every node right before histogram-building). Truncating before the
    # sort/boundary determination — as an earlier version of this rewrite
    # did — changes both the sort order and the lower_Position boundary
    # whenever raw values have a fractional part, corrupting the whole
    # partition. Sort and find the boundary on the raw values first, then
    # truncate only for the histogram/LUT indexing that actually needs ints.
    flat_r_raw = r.reshape(-1)
    # Stable ascending sort, matching Python's stable sorted(..., reverse=False)
    # over the row-major-appended node list — ties keep their original
    # (row-major) relative order.
    order = np.argsort(flat_r_raw, kind='stable')

    # lower_Position = count of pixels with value <= R__middle, exactly the
    # first index (in ascending sorted order) whose value exceeds R__middle
    # — the same boundary the original found by scanning for the first
    # sorted node with value > R__middle.
    lower_Position = int(np.sum(flat_r_raw <= R__middle))

    flat_r = flat_r_raw.astype(np.int64)

    lower_idx = order[:lower_Position]
    # The original's Lower loops (histogram build AND value mutation) both
    # run range(0, lower_Position) — strictly excluding the boundary
    # position. Its Upper loops (histogram build AND value mutation) both
    # run range(lower_Position, allSize) — INCLUDING the boundary position,
    # so the boundary pixel's value does feed into the Upper histogram/LUT
    # even though (see below) the transformed value it gets is then never
    # actually used. upper_hist_idx mirrors that inclusive range.
    upper_hist_idx = order[lower_Position:]
    # The final assembly step's condition is strictly "i > lower_Position",
    # so only sorted-positions AFTER the boundary actually receive the
    # Upper-transformed value in the output array — upper_out_idx mirrors
    # that exclusive range.
    upper_out_idx = order[lower_Position + 1:]
    # The single sorted-position exactly AT lower_Position is therefore
    # never actually assigned a transformed value by either side: Lower's
    # mutation loop stops before reaching it, and Upper's mutated value for
    # it is computed but never read by the final assembly (which takes the
    # untouched Lower value there instead, per its "else" branch). A real
    # (if minor) quirk of the original algorithm, reproduced explicitly
    # rather than glossed over.
    boundary_idx = order[lower_Position] if lower_Position < allSize else None

    NumPixel_lower = np.bincount(flat_r[lower_idx], minlength=256)[:256]
    LUT_lower = _rayleighLUT(NumPixel_lower, lower_Position)

    NumPixel_upper = np.bincount(flat_r[upper_hist_idx], minlength=256)[:256]
    LUT_upper = _rayleighLUT(NumPixel_upper, allSize - lower_Position)

    array_lower_histogram_stretching = np.zeros(allSize)
    array_upper_histogram_stretching = np.zeros(allSize)

    array_lower_histogram_stretching[lower_idx] = LUT_lower[flat_r[lower_idx]]
    array_upper_histogram_stretching[lower_idx] = 0

    array_upper_histogram_stretching[upper_out_idx] = LUT_upper[flat_r[upper_out_idx]]
    array_lower_histogram_stretching[upper_out_idx] = 255

    if boundary_idx is not None:
        array_lower_histogram_stretching[boundary_idx] = flat_r[boundary_idx]
        array_upper_histogram_stretching[boundary_idx] = 0

    array_lower_histogram_stretching = array_lower_histogram_stretching.reshape(height, width)
    array_upper_histogram_stretching = array_upper_histogram_stretching.reshape(height, width)

    return array_lower_histogram_stretching,array_upper_histogram_stretching

def rayleighStretching(sceneRadiance, height, width):

    R_array_lower_histogram_stretching, R_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 2], height, width)
    G_array_lower_histogram_stretching, G_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 1], height, width)
    B_array_lower_histogram_stretching, B_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 0], height, width)

    sceneRadiance_Lower = np.zeros((height, width, 3), )
    sceneRadiance_Lower[:, :, 0] = B_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 1] = G_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 2] = R_array_lower_histogram_stretching
    sceneRadiance_Lower = np.uint8(sceneRadiance_Lower)

    sceneRadiance_Upper = np.zeros((height, width, 3))
    sceneRadiance_Upper[:, :, 0] = B_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 1] = G_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 2] = R_array_upper_histogram_stretching
    sceneRadiance_Upper = np.uint8(sceneRadiance_Upper)

    return sceneRadiance_Lower, sceneRadiance_Upper




# def rayleighStr(r, height, width):
#     alpha = 0.4
#     selectedRange = [0, 255]
#     NumPixel = np.zeros(256)
#     temp = np.zeros(256)
#     for i in range(0, height):
#         for j in range(0, width):
#             NumPixel[r[i, j]] = NumPixel[r[i, j]] + 1
#     ProbPixel = NumPixel / (height * width)
#     CumuPixel = np.cumsum(ProbPixel)
#     valSpread = selectedRange[1] - selectedRange[0]
#     hconst = 2 * alpha ** 2
#     vmax = 1 - e ** (-1 / hconst)
#     val = vmax * (CumuPixel)
#     val = np.array(val)
#
#     for i in range(256):
#         if (val[i] >= 1):
#             val[i] = val[i] - esp
#     for i in range(256):
#         temp[i] = np.sqrt(-hconst * math.log((1 - val[i]), e))
#         normalization = temp[i] * valSpread
#         if(normalization > 255):
#             CumuPixel[i] = 255
#         else:
#             CumuPixel[i] = normalization
#     for i in range(0, height):
#         for j in range(0, width):
#             r[i, j] = CumuPixel[(r[i, j])]
#     return r


