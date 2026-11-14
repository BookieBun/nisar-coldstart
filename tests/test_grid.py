import numpy as np

from coldstart.grid import aggregate_linear, assign_folds, to_db


def test_aggregate_known_block():
    x = np.array([[1.0, 2.0, 10.0, 10.0], [3.0, 4.0, 10.0, 10.0]])
    out = aggregate_linear(x, 2, 0.6)
    assert out.shape == (1, 2)
    assert np.isclose(out[0, 0], 2.5) and np.isclose(out[0, 1], 10.0)


def test_aggregate_validity_threshold():
    x = np.array([[1.0, np.nan], [3.0, 4.0]])          # 3 of 4 valid -> mean of valid
    assert np.isclose(aggregate_linear(x, 2, 0.6)[0, 0], 8 / 3)
    y = np.array([[1.0, np.nan], [0.0, 4.0]])          # 2 of 4 valid (0 is invalid) -> nan
    assert np.isnan(aggregate_linear(y, 2, 0.6)[0, 0])


def test_to_db():
    assert to_db(np.array([1.0]))[0] == 0.0
    assert np.isclose(to_db(np.array([0.1]))[0], -10.0)
    assert np.isnan(to_db(np.array([0.0, -1.0, np.nan]))).all()


def test_tiles_map_to_one_fold_and_are_seeded():
    tile, fold = assign_folds(95, 130, 20, 5, seed=3)
    for t in np.unique(tile):
        assert len(np.unique(fold[tile == t])) == 1
    _, fold2 = assign_folds(95, 130, 20, 5, seed=3)
    assert np.array_equal(fold, fold2)
    assert set(np.unique(fold)) == set(range(5))
