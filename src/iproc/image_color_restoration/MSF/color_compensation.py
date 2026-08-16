import numpy as np


def compensate_and_balance(img: np.ndarray, a: float = 1.0) -> np.ndarray:
    """
    Adaptive red/blue channel compensation against the green channel (green is
    used as reference since it attenuates least underwater), followed by a
    gray-world normalization. Expects and returns an RGB image with values in
    [0, 1].
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()

    r_comp = r + a * (mean_g - mean_r) * (1 - r * g)
    b_comp = b + a * (mean_g - mean_b) * (1 - b * g)

    compensated = np.stack([r_comp, g, b_comp], axis=2)
    compensated = np.clip(compensated, 0, 1)

    return _gray_world_normalize(compensated)


def _gray_world_normalize(img: np.ndarray) -> np.ndarray:
    channel_means = img.reshape(-1, 3).mean(axis=0)
    channel_means = np.clip(channel_means, 1e-6, None)
    gray_mean = channel_means.mean()
    scale = gray_mean / channel_means
    return np.clip(img * scale, 0, 1)
