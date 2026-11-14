import numpy as np

from coldstart.postprocess import remove_small_patches


def test_patch_sizes_and_diagonals():
    f = np.zeros((10, 10), bool)
    f[0, 0:3] = True                       # 3 cells -> removed
    f[5, 5] = f[6, 6] = f[7, 7] = f[8, 8] = True   # 4 diagonal cells -> one component, kept
    out = remove_small_patches(f, 4)
    assert not out[0, 0:3].any()
    assert out[5, 5] and out[8, 8]
    f2 = np.zeros((5, 5), bool)
    f2[1, 1:5] = True                      # 4 in a row -> kept
    assert remove_small_patches(f2, 4).sum() == 4
