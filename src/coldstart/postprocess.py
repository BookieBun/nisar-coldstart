"""Minimum-patch rule, applied identically to every thresholded map (never to raw scores)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

_EIGHT = np.ones((3, 3), dtype=bool)


def remove_small_patches(flag: np.ndarray, min_cells: int) -> np.ndarray:
    """Drop 8-connected components of `flag` smaller than min_cells."""
    flag = np.asarray(flag, dtype=bool)
    if min_cells <= 1 or not flag.any():
        return flag.copy()
    lab, n = label(flag, structure=_EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_cells
    keep[0] = False
    return keep[lab]
