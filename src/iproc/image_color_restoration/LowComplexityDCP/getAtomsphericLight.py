import numpy as np

def getAtomsphericLight(darkChannel, img, meanMode, percent):
    # meanMode is accepted but was never actually read by the original
    # function body either — preserved as an unused parameter to keep the
    # call signature intact.
    size = darkChannel.size
    k = int(percent * size)

    # Full stable sort by descending dark-channel value (ties broken by
    # ascending row-major index) — not argpartition, which picks an
    # arbitrary subset among tied boundary values and can silently select a
    # different top-k candidate set than the original's stable Python sort
    # (confirmed on real images for the analogous IBLA case). Sorting this
    # way also directly gives candidates in the same walk order the
    # original's node-list iteration used, so a tie in RGB-sum resolves to
    # the candidate with the higher dark-channel value — np.argmax below
    # returns the first occurrence on ties, matching that walk order.
    # darkChannel is uint8 — negating it directly wraps around instead of
    # producing negative values, so cast to a signed type first.
    flat_dark = darkChannel.reshape(-1).astype(np.int32)
    flat_img = img.reshape(-1, 3)
    candidate_idx = np.argsort(-flat_dark, kind='stable')[:k]

    sums = flat_img[candidate_idx].astype(np.int32).sum(axis=1)
    best = candidate_idx[np.argmax(sums)]
    return flat_img[best]

