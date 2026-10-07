"""Causal per-frame features: data/processed/<stem>.parquet -> data/features/<stem>.parquet.

One implementation for batch and live use: CausalFeatureExtractor.step() sees one frame at a time
and only state built from earlier frames, so features are causal by construction and a live
stream cannot drift from training. Batch compute_features() loops step() in (timestamp, row_idx)
order. One stream = one file/session (reset() between them).

Conventions
- Sentinel -1 everywhere (every real value is >= 0): first frame of an ID, first frame of the
  stream, payload never seen before.
- Windows count PAST frames (processed earlier) with t' > t - W; the current frame is excluded.
- IDs not in train_stats use global_ref_period_s as ref period, with no flag and no special
  sentinel, so dt_ratio stays purely behavioral. Knowing the ID exists only in ID_INCLUSIVE.
- Payload history is bounded to BATCH_PAYLOAD_HISTORY_MAX_AGE_S (10 s) for the feature files and
  the live demo alike: older entries are pruned and treated as never seen. Unbounded history made
  the Normal "seen before" rate climb with time since stream start (an implicit position-in-file
  feature); 10 s removes that drift (chosen on train data, notebooks/02_features.ipynb).
  The class default (None = unbounded) exists for tests and analysis only.
Usage: python -m src.features
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.load import BYTE_COLS, PROCESSED_DIR, ROOT
from src.train_stats import load_splits, load_train_stats

FEATURES_DIR = ROOT / "data" / "features"

ID_WINDOW_S = 0.100
BUS_WINDOWS_S = (0.050, 0.100, 0.200)
MODE_CODE = {"S": 0, "D": 1}
BATCH_PAYLOAD_HISTORY_MAX_AGE_S = 10.0

BEHAVIORAL = ([
    "dt_same_id", "dt_ratio", "is_first_of_id", "dt_prev_frame", "id_count_last_100ms",
    "frame_count_last_50ms", "frame_count_last_100ms", "frame_count_last_200ms",
    "payload_changed", "n_bytes_changed", "payload_repeat_age", "payload_seen_before",
] + BYTE_COLS + ["dlc", "mode"])
ID_INCLUSIVE = BEHAVIORAL + ["arb_id", "id_seen_in_training"]
ALL_FEATURES = ID_INCLUSIVE  # column order of CausalFeatureExtractor.step() output

# Kept in the features parquet for evaluation/joins; never model inputs.
META_COLS = ["source_file", "subset", "row_idx", "timestamp", "class", "subclass", "is_attack"]
INPUT_COLS = (["source_file", "subset", "mode", "row_idx", "timestamp", "arb_id", "dlc"]
              + BYTE_COLS + ["class", "subclass", "is_attack"])

FEATURE_DTYPES = {
    **dict.fromkeys(["dt_same_id", "dt_ratio", "dt_prev_frame", "payload_repeat_age"], np.float32),
    **dict.fromkeys(["is_first_of_id", "payload_changed", "n_bytes_changed", "payload_seen_before",
                     "dlc", "mode", "id_seen_in_training"], np.int8),
    **dict.fromkeys(["id_count_last_100ms", "frame_count_last_50ms", "frame_count_last_100ms",
                     "frame_count_last_200ms", "arb_id"] + BYTE_COLS, np.int16),
}

# Absolute time/position and labels leak the label (attack schedules repeat across files).
_FORBIDDEN = set(META_COLS) | {"ts_relative"}
if _FORBIDDEN & (set(BEHAVIORAL) | set(ID_INCLUSIVE)):
    raise RuntimeError(f"forbidden columns in feature lists: {_FORBIDDEN & set(ID_INCLUSIVE)}")
if set(FEATURE_DTYPES) != set(ALL_FEATURES):
    raise RuntimeError("FEATURE_DTYPES must cover exactly ALL_FEATURES")


def _count_and_push(dq: deque, ts: float, window: float) -> int:
    """Count past timestamps in (ts - window, ...], then add ts. Assumes non-decreasing ts."""
    cutoff = ts - window
    while dq and dq[0] <= cutoff:
        dq.popleft()
    n = len(dq)
    dq.append(ts)
    return n


class CausalFeatureExtractor:
    """Streaming per-frame features from past frames only. step() returns values in ALL_FEATURES order."""

    def __init__(self, stats: dict, mode: str, payload_history_max_age_s: float | None = None):
        if mode not in MODE_CODE:
            raise ValueError(f"mode must be one of {list(MODE_CODE)}, got {mode!r}")
        if payload_history_max_age_s is not None and payload_history_max_age_s <= 0:
            raise ValueError("payload_history_max_age_s must be > 0 or None")
        self.mode_code = MODE_CODE[mode]
        self.known_ids = frozenset(stats["known_ids"])
        self.global_ref = float(stats["global_ref_period_s"])
        self.ref_period = {int(i): float(s["median_period_s"]) for i, s in stats["per_id"].items()}
        self.max_age = payload_history_max_age_s
        self.reset()

    def reset(self) -> None:
        self.prev_ts = None
        self.last_ts: dict[int, float] = {}
        self.last_payload: dict[int, tuple] = {}
        self.id_windows: dict[int, deque] = {}
        self.bus_windows = [deque() for _ in BUS_WINDOWS_S]
        self.payload_last_seen: dict[tuple, float] = {}
        self.next_prune = None

    def _prune_payload_history(self, ts: float) -> None:
        if self.next_prune is None:
            self.next_prune = ts + self.max_age
        elif ts >= self.next_prune:
            cutoff = ts - self.max_age
            self.payload_last_seen = {k: v for k, v in self.payload_last_seen.items() if v >= cutoff}
            self.next_prune = ts + self.max_age

    def step(self, ts: float, arb_id: int, dlc: int, bytes8) -> tuple:
        bytes8 = tuple(bytes8)

        last = self.last_ts.get(arb_id)
        if last is None:
            is_first, dt, dt_ratio = 1, -1.0, -1.0
        else:
            dt = ts - last
            is_first, dt_ratio = 0, dt / self.ref_period.get(arb_id, self.global_ref)
        dt_prev = -1.0 if self.prev_ts is None else ts - self.prev_ts

        id_window = self.id_windows.get(arb_id)
        if id_window is None:
            id_window = self.id_windows[arb_id] = deque()
        id_count = _count_and_push(id_window, ts, ID_WINDOW_S)
        bus_counts = [_count_and_push(dq, ts, w) for dq, w in zip(self.bus_windows, BUS_WINDOWS_S)]

        prev = self.last_payload.get(arb_id)
        if prev is None:
            changed, n_changed = -1, -1
        else:
            prev_dlc, prev_bytes = prev
            n_changed = sum(a != b for a, b in zip(prev_bytes, bytes8))
            changed = int(n_changed > 0 or prev_dlc != dlc)

        if self.max_age is not None:
            self._prune_payload_history(ts)
        key = (arb_id, dlc) + bytes8
        seen = self.payload_last_seen.get(key)
        age = -1.0 if seen is None else ts - seen
        if self.max_age is not None and age > self.max_age:
            age = -1.0

        self.payload_last_seen[key] = ts
        self.last_ts[arb_id] = ts
        self.last_payload[arb_id] = (dlc, bytes8)
        self.prev_ts = ts

        return (dt, dt_ratio, is_first, dt_prev, id_count, *bus_counts,
                changed, n_changed, age, int(age >= 0), *bytes8, dlc, self.mode_code,
                arb_id, int(arb_id in self.known_ids))


def compute_features(df: pd.DataFrame, stats: dict,
                     payload_history_max_age_s: float | None = None) -> pd.DataFrame:
    """One stream (file) -> META_COLS present in df + ALL_FEATURES, sorted by (timestamp, row_idx)."""
    modes = df["mode"].astype(str).unique()
    if len(modes) != 1:
        raise ValueError(f"one stream must have one mode, got {list(modes)}")
    order = np.lexsort((df["row_idx"].to_numpy(), df["timestamp"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)

    ext = CausalFeatureExtractor(stats, mode=modes[0], payload_history_max_age_s=payload_history_max_age_s)
    step = ext.step
    ts = df["timestamp"].to_numpy(np.float64).tolist()
    ids = df["arb_id"].to_numpy().tolist()
    dlcs = df["dlc"].to_numpy().tolist()
    payloads = zip(*(df[c].to_numpy().tolist() for c in BYTE_COLS))

    # float32 holds every feature exactly (ints < 2^24; dt features are float32 anyway).
    out = np.empty((len(df), len(ALL_FEATURES)), dtype=np.float32)
    for i, (t, a, d, b) in enumerate(zip(ts, ids, dlcs, payloads)):
        out[i] = step(t, a, d, b)

    feats = pd.DataFrame({name: out[:, j].astype(FEATURE_DTYPES[name])
                          for j, name in enumerate(ALL_FEATURES)})
    meta = df[[c for c in META_COLS if c in df.columns]]
    return pd.concat([meta, feats], axis=1)


def process_file(stem: str, stats: dict, processed_dir: Path = PROCESSED_DIR,
                 out_dir: Path = FEATURES_DIR,
                 payload_history_max_age_s: float | None = BATCH_PAYLOAD_HISTORY_MAX_AGE_S) -> dict:
    src = Path(processed_dir) / f"{stem}.parquet"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")

    t0 = time.perf_counter()
    feats = compute_features(pd.read_parquet(src, columns=INPUT_COLS), stats, payload_history_max_age_s)
    elapsed = time.perf_counter() - t0

    n_in = pq.ParquetFile(src).metadata.num_rows
    if len(feats) != n_in:
        raise RuntimeError(f"{stem}: row count mismatch (in {n_in}, out {len(feats)})")
    if feats[ALL_FEATURES].isna().any().any():
        raise RuntimeError(f"{stem}: NaN in feature columns")
    feats.to_parquet(tmp_path, compression="zstd", index=False)
    tmp_path.replace(out_path)
    return {"source_file": stem, "rows": n_in, "seconds": elapsed, "out": str(out_path)}


if __name__ == "__main__":
    splits = load_splits()
    stats = load_train_stats()
    stems = splits["train"] + splits["test"] + splits["extra_test"]
    t_all = time.perf_counter()
    print(f"payload_history_max_age_s = {BATCH_PAYLOAD_HISTORY_MAX_AGE_S}")
    for stem in stems:
        r = process_file(stem, stats)
        print(f"{stem:<28} rows={r['rows']:>9,}  {r['seconds']:6.1f}s  "
              f"({r['seconds'] / r['rows'] * 1e6:.1f} us/frame)")
    print(f"\n{len(stems)} files in {time.perf_counter() - t_all:.0f}s -> {FEATURES_DIR}")
    try:
        import psutil
        mi = psutil.Process().memory_info()
        print(f"peak memory: {getattr(mi, 'peak_wset', mi.rss) / 2**20:.0f} MB")
    except ImportError:
        pass
