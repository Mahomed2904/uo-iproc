# iproc

Python library of classic (non-learned) underwater image color-restoration and enhancement
algorithms, implemented against a single, uniform interface so any of them can be swapped in as a
drop-in preprocessing step.

**This library is not tied to any one application.** It was extracted from
`UnderwaterObservserApp` (a reef-monitoring device that runs one of these algorithms before
on-device fish detection), but there is nothing in the code that depends on that use case, on a
detection pipeline, or on any particular hardware. Underwater image color restoration/enhancement
is a general problem — turbidity and color-cast correction, contrast/dehazing, dataset
preprocessing before training a vision model, or improving footage for human review are all valid
uses. Any project that needs to correct the color cast, haze, or low contrast typical of
underwater photos can depend on `iproc` directly.

## Algorithms

All algorithms take a `numpy.ndarray` image (BGR, as read by OpenCV) and return a corrected
`numpy.ndarray` of the same shape via a static `apply(img)` method.

**Color restoration** (`iproc.image_color_restoration`) — physics-based dehazing/restoration
models that estimate a transmission map and atmospheric/background light to recover scene
radiance:
- `DCP` — Dark Channel Prior
- `GBdehazingRCorrection` — DCP variant with green-blue dehazing and red-channel correction
- `IBLA` — Image Blurriness and Light Absorption
- `LowComplexityDCP` — reduced-cost DCP variant
- `MIP` — Minimum Information Prior
- `MIX` — combined color + contrast correction pipeline
- `MSF` — Multi-Scale Fusion
- `NewOpticalModel` — revised underwater optical (formation) model
- `RoWS` — Rank-One prior underwater dehazing
- `UDCP` — Underwater DCP
- `ULAP` — Underwater Light Attenuation Prior

**Enhancement** (`iproc.image_enhancement`) — direct pixel-value/histogram methods that don't
model the physical formation process:
- `CLAHE` — Contrast Limited Adaptive Histogram Equalization
- `GC` — Gamma Correction
- `HE` — Histogram Equalization
- `ICM` — Integrated Color Model
- `RGHS` — Relative Global Histogram Stretching
- `RayleighDistribution` — Rayleigh-distribution-based stretching
- `UCM` — Unsupervised Color Correction Method

`iproc.NONE` provides a no-op `NONE_ALG` (returns the input unchanged) so callers can select
"no preprocessing" through the same interface as every real algorithm.

## Usage

Every algorithm implements `iproc.interface.ImageAlgorithmInterface`:

```python
from abc import ABC, abstractmethod
import numpy as np

class ImageAlgorithmInterface(ABC):
    @staticmethod
    @abstractmethod
    def apply(img: np.ndarray) -> np.ndarray: ...
```

which makes every algorithm interchangeable behind that one method:

```python
import cv2
from iproc.image_color_restoration.UDCP.main import UDCP_ICR
from iproc.image_enhancement.CLAHE.main import CLAHE_IE

img = cv2.imread("underwater_frame.jpg")

restored = UDCP_ICR.apply(img)     # physics-based restoration
enhanced = CLAHE_IE.apply(img)     # histogram-based enhancement
```

Because every implementation shares this interface, an algorithm can be chosen at runtime (e.g.
by name/config) without the caller needing an `if/elif` per algorithm — this is how
`UnderwaterObservserApp`'s `ImagesPreprocessorFactory` selects one of them before running
detection, but the same pattern applies to any pipeline that wants a swappable preprocessing
step.

## Install

```
pip install -e .
```

## Origin

Extracted from `UnderwaterObservserApp/src/main/libs/iproc/`, where it originated as vendored,
in-tree code, and pulled out into its own package so it can be reused and versioned independently
of that application.
