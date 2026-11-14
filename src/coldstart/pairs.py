"""Date pairs: one event pair, and no-event pairs before ignition and after event.end."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .log import get_logger

log = get_logger()

EVENT, PRE, POST = "event", "no_event_pre", "no_event_post"


@dataclass
class Pair:
    id: str
    date_a: str
    date_b: str
    kind: str
    history_dates: list[str]                                  # all dates <= date_a (handoff 5.5)
    b2_history_dates: list[str] = field(default_factory=list)  # B2's fixed-length same-regime history

    @property
    def is_event(self) -> bool:
        return self.kind == EVENT


def define_pairs(dates: list[str], cfg: Config) -> list[Pair]:
    ign, end = cfg.event.ignition, cfg.event.end
    if end is None:
        raise ValueError("event.end must be set (first date treated as post-event)")
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError(f"dates must be sorted and unique: {dates}")
    pre = [d for d in dates if d < ign]
    post = [d for d in dates if d >= end]
    if not pre or not post:
        raise ValueError(f"need dates before ignition {ign} and on/after event.end {end}; got {dates}")
    n = cfg.baselines.b2_history_dates

    def mk(a: str, b: str, kind: str) -> Pair:
        # B2 history comes from the same regime as date_a, so a post-fire pair never
        # averages pre-fire values (which would flag the burn itself as "change").
        regime = pre if kind in (EVENT, PRE) else post
        same = [d for d in regime if d <= a]
        return Pair(id=f"{a.replace('-', '')}_{b.replace('-', '')}", date_a=a, date_b=b, kind=kind,
                    history_dates=[d for d in dates if d <= a],
                    b2_history_dates=same[-n:] if len(same) >= n else [])

    pairs = [mk(pre[-1], post[0], EVENT)]
    pairs += [mk(a, b, PRE) for a, b in zip(pre, pre[1:])]
    pairs += [mk(a, b, POST) for a, b in zip(post, post[1:])]
    skipped = [d for d in dates if ign <= d < end]
    if skipped:
        log.info("dates between ignition and event.end, used in no pair: %s", skipped)
    n_no = sum(not p.is_event for p in pairs)
    if n_no < 2:
        log.warning("only %d no-event pair(s): calibration falls back to the spatial hold-out", n_no)
    return pairs


def save_pairs(pairs: list[Pair], path: Path) -> None:
    path.write_text(json.dumps([asdict(p) for p in pairs], indent=2))


def load_pairs(path: Path) -> list[Pair]:
    return [Pair(**d) for d in json.loads(Path(path).read_text())]


def event_pair(pairs: list[Pair]) -> Pair:
    ev = [p for p in pairs if p.is_event]
    if len(ev) != 1:
        raise ValueError(f"expected exactly one event pair, found {len(ev)}")
    return ev[0]
