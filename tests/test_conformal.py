import math

import numpy as np

from coldstart.conformal import pvalues, threshold


def test_threshold_index_matches_formula():
    s = np.arange(1, 1001, dtype=float)[::-1]          # unsorted on purpose
    for a in (0.01, 0.05, 0.1):
        i = math.ceil((1000 + 1) * (1 - a))
        assert threshold(s, a) == float(i)             # sorted values equal their rank


def test_threshold_inf_when_too_few():
    assert threshold(np.arange(5.0), 0.01) == float("inf")


def test_realized_rate_on_iid_scores():
    rng = np.random.default_rng(0)
    cal, test = rng.standard_normal(100_000), rng.standard_normal(100_000)
    for a in (0.01, 0.05):
        r = (test > threshold(cal, a)).mean()
        assert abs(r - a) <= 0.2 * a


def test_pvalues():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(pvalues(np.array([0.0, 2.5, 5.0]), cal), [5 / 5, 3 / 5, 1 / 5])
