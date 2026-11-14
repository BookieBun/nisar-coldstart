"""Download the selected scenes. Refuses to start if they would not fit on disk.

With io.stream: true (the default for the real site) the grid stage reads windows over
HTTPS and no download is needed; this command is for offline work only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .config import Config
from .log import get_logger, require_inputs, stage

log = get_logger()


def download_scenes(cfg: Config) -> list[Path]:
    import asf_search as asf

    man = cfg.interim / "manifest.csv"
    require_inputs("download", man)
    with stage("download") as outputs:
        df = pd.read_csv(man, dtype=str)
        df = df[df.selected.str.lower() == "true"]
        if df.empty:
            raise ValueError("no selected scenes in manifest.csv; set search.track/frame/... and rerun search")
        dest = cfg.raw_dir
        dest.mkdir(parents=True, exist_ok=True)
        todo, need = [], 0
        for _, r in df.iterrows():
            f = dest / Path(r.url).name
            size = int(float(r.bytes)) if r.bytes not in (None, "", "nan") else None
            if f.exists() and size is not None and f.stat().st_size == size:
                outputs.append(f)
                continue
            todo.append((r.url, f))
            need += size or 0
        free = shutil.disk_usage(dest).free
        gb = need / 1e9
        if gb > cfg.io.max_download_gb:
            raise RuntimeError(f"{gb:.1f} GB to download exceeds io.max_download_gb={cfg.io.max_download_gb}; "
                               "use io.stream: true instead")
        if free - need < cfg.io.min_free_gb * 1e9:
            raise RuntimeError(f"{gb:.1f} GB needed but only {free / 1e9:.1f} GB free "
                               f"(io.min_free_gb={cfg.io.min_free_gb}); use io.stream: true")
        session = asf.ASFSession()
        for url, f in todo:
            log.info("downloading %s", f.name)
            asf.download_url(url=url, path=str(dest), filename=f.name, session=session)
            outputs.append(f)
        return outputs
