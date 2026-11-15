"""app/places.py: patch list for the 'Places to check first' table."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from places import COLUMNS, flagged_places  # noqa: E402

BOUNDS = [[44.0, -120.0], [45.0, -119.0]]


def test_patches_sizes_order_and_overlap():
    flags = np.zeros((100, 100), bool)
    flags[10:20, 10:20] = True          # 100 cells, overlaps the perimeter
    flags[60:63, 60:63] = True          # 9 cells, outside
    flags[80, 80] = flags[81, 81] = True  # diagonal neighbours: one patch of 2
    labels = np.zeros((100, 100), int)
    labels[15:30, 15:30] = 1
    score = np.full((100, 100), 2.0)
    df = flagged_places(flags, labels, score, BOUNDS, [100, 100])
    assert list(df.columns) == COLUMNS
    assert df["cells"].tolist() == [100, 9, 2]
    assert df["overlaps_perimeter"].tolist() == [True, False, False]
    assert np.allclose(df["mean_score"], 2.0)
    r = df.iloc[0]
    assert 44.0 < r["lat"] < 45.0 and -120.0 < r["lon"] < -119.0
    assert np.isclose(r["lat"], 45.0 - (14.5 + 0.5) / 100)      # centroid row 14.5
    assert df["area_km2"].is_monotonic_decreasing


def test_no_flags_gives_empty_table():
    z = np.zeros((10, 10))
    df = flagged_places(z.astype(bool), z.astype(int), z, BOUNDS, [10, 10])
    assert df.empty and list(df.columns) == COLUMNS
