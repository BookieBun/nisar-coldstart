"""Logging: one logger for the package and a timing context manager for stages."""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Iterator


def get_logger(name: str = "coldstart") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextlib.contextmanager
def stage(name: str) -> Iterator[list[Path]]:
    """Log start, end, wall time and the output paths appended to the yielded list."""
    log = get_logger()
    t0 = time.perf_counter()
    log.info("[%s] start", name)
    outputs: list[Path] = []
    yield outputs
    shown = ", ".join(str(p) for p in outputs[:6]) + (f" (+{len(outputs) - 6} more)" if len(outputs) > 6 else "")
    log.info("[%s] done in %.1f s; outputs: %s", name, time.perf_counter() - t0, shown or "-")


def require_inputs(stage_name: str, *paths: Path) -> None:
    """Refuse to run a stage whose inputs are missing (rule 9: fail loudly)."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"stage '{stage_name}' cannot run; missing inputs:\n  " + "\n  ".join(missing)
            + "\nRun the earlier stages first (or `python -m coldstart run CONFIG`)."
        )
