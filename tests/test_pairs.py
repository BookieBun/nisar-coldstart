import logging

from coldstart.log import get_logger
from coldstart.pairs import EVENT, define_pairs

DATES = ["2026-07-01", "2026-07-13", "2026-07-25", "2026-08-06", "2026-08-14", "2026-08-18", "2026-08-30"]


def test_one_event_pair_and_no_spanning(cfg_small):
    # ignition 2026-08-10, end 2026-08-18: 2026-08-14 lies between and must be unused
    pairs = define_pairs(DATES, cfg_small)
    ev = [p for p in pairs if p.kind == EVENT]
    assert len(ev) == 1 and (ev[0].date_a, ev[0].date_b) == ("2026-08-06", "2026-08-18")
    for p in pairs:
        if p.kind != EVENT:
            assert not (p.date_a < "2026-08-10" <= p.date_b)
        assert "2026-08-14" not in (p.date_a, p.date_b)


def test_b2_history_is_same_regime_and_fixed_length(cfg_small):
    pairs = {p.id: p for p in define_pairs(DATES, cfg_small)}
    post = pairs["20260818_20260830"]
    assert post.b2_history_dates == []                     # only one post-regime date <= date_a
    ev = pairs["20260806_20260818"]
    assert ev.b2_history_dates == ["2026-07-25", "2026-08-06"]


def test_fallback_warning_with_few_no_event_pairs(cfg_small):
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    h = H()
    get_logger().addHandler(h)
    try:
        pairs = define_pairs(["2026-07-25", "2026-08-06", "2026-08-18"], cfg_small)
    finally:
        get_logger().removeHandler(h)
    assert sum(p.kind != EVENT for p in pairs) == 1
    assert any("spatial" in m for m in records)
