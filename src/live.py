"""Live replay engine for the dashboard (app/dashboard.py): frames -> causal features -> model -> alerts.

No Streamlit import, so everything here is unit-testable (tests/test_live.py).

- Features come from the same CausalFeatureExtractor as the feature files, with the same 10 s payload
  history bound, so the live stream cannot drift from training. step() runs per frame; the model scores
  each chunk of frames in one batched call (one row at a time would be ~1 ms/frame, latency.csv).
- A frame is flagged when score > threshold (src/evaluate.py convention). An ALERT fires when at least
  n_alert flagged frames fall within ALERT_WINDOW_S; triggers less than ALERT_GAP_S apart belong to one
  incident (one alert-log row).
- Stream time is shown relative to the first frame; it is display/pacing only, never a feature.
- load_detector() never raises: the final XGBoost behavioral model if its files are complete, else the
  dt_ratio rule, with the reason recorded in Detector.source.
"""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import PRIMARY_FPR
from src.features import ALL_FEATURES, BATCH_PAYLOAD_HISTORY_MAX_AGE_S, BEHAVIORAL, CausalFeatureExtractor
from src.load import BYTE_COLS, PROCESSED_DIR, SUBCLASS_CATS
from src.train import FINAL_DIR, REPORT_DIR, SELECTED_PATH, TEST_METRICS_PATH, DtRatioRule

CV_THRESHOLDS_PATH = REPORT_DIR / "cv_thresholds.csv"
RULE_DEFAULT_THRESHOLD = -0.031  # rule/behavioral fold-mean threshold at 0.1% FPR (cv_thresholds.csv)

ALERT_WINDOW_S = 0.100
ALERT_GAP_S = 1.0     # triggers closer than this are one incident
STATUS_HOLD_S = 1.0   # status stays ATTACK this long after the last trigger (no flicker)
BIN_S = 0.100         # chart resolution (bus load = frames per bin)
N_BINS = 1500         # rolling chart window: 150 s of stream time
SCORE_CLIP = 20.0     # plotting only (the rule's first-frame sentinel is -1e9)
LOG_MAX = 200
NORMAL = SUBCLASS_CATS.index("Normal")
ATTACK_NAMES = [c for c in SUBCLASS_CATS if c != "Normal"]


# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class Frames:
    """One stream in (timestamp, row_idx) order, as numpy arrays."""
    stem: str
    mode: str
    ts: np.ndarray       # float64, absolute (fed to the extractor exactly as in compute_features)
    ids: np.ndarray
    dlc: np.ndarray
    payload: np.ndarray  # (n, 8), -1 beyond DLC
    sub: np.ndarray      # int8 codes into SUBCLASS_CATS (ground truth, display only)

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def rel(self) -> np.ndarray:
        return self.ts - self.ts[0]

    @property
    def duration(self) -> float:
        return float(self.ts[-1] - self.ts[0]) if len(self.ts) else 0.0


def frames_from_df(df: pd.DataFrame, stem: str) -> Frames:
    modes = df["mode"].astype(str).unique()
    if len(modes) != 1:
        raise ValueError(f"{stem}: one stream must have one mode, got {list(modes)}")
    order = np.lexsort((df["row_idx"].to_numpy(), df["timestamp"].to_numpy()))
    df = df.iloc[order]
    sub = pd.Categorical(df["subclass"].astype(str), categories=SUBCLASS_CATS).codes
    if (sub < 0).any():
        raise ValueError(f"{stem}: unknown subclass labels")
    ts = df["timestamp"].to_numpy(np.float64)
    ts.flags.writeable = False
    # Narrow dtypes (~28 B/frame, 8 GB laptop); .tolist() hands the extractor plain Python ints anyway.
    return Frames(stem=stem, mode=modes[0], ts=ts, ids=df["arb_id"].to_numpy(np.int32),
                  dlc=df["dlc"].to_numpy(np.int8), payload=df[BYTE_COLS].to_numpy(np.int16),
                  sub=sub.astype(np.int8))


def load_frames(stem: str, processed_dir: Path = PROCESSED_DIR) -> Frames:
    cols = ["mode", "row_idx", "timestamp", "arb_id", "dlc"] + BYTE_COLS + ["subclass"]
    return frames_from_df(pd.read_parquet(Path(processed_dir) / f"{stem}.parquet", columns=cols), stem)


def list_streams(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    return sorted(p.stem for p in Path(processed_dir).glob("*.parquet"))


# ---------------------------------------------------------------- model + threshold

@dataclass
class Detector:
    model: object          # src.train.Detector: .features, .score(X) higher = more attack-like
    threshold: float
    name: str
    source: str
    target_fpr: float = PRIMARY_FPR
    fallback: bool = False
    feature_idx: list = field(init=False)

    def __post_init__(self):
        self.feature_idx = [ALL_FEATURES.index(f) for f in self.model.features]

    def score(self, X_all: np.ndarray) -> np.ndarray:
        """X_all: (n, len(ALL_FEATURES)) float32 in extractor output order."""
        return np.asarray(self.model.score(np.ascontiguousarray(X_all[:, self.feature_idx])), dtype=np.float64)


def final_threshold(path: Path, model: str, feature_set: str, config: str,
                    target_fpr: float = PRIMARY_FPR) -> float:
    """The threshold run_final applied (threshold column of test_metrics.csv); one finite value or raise."""
    d = pd.read_csv(path)
    rows = d[(d["model"] == model) & (d["feature_set"] == feature_set) & (d["config"] == config)
             & np.isclose(d["target_fpr"], target_fpr)]
    vals = rows["threshold"].unique()
    if len(vals) != 1 or not np.isfinite(vals[0]):
        raise ValueError(f"expected one finite threshold for {model}/{feature_set}/{config}, got {list(vals)}")
    return float(vals[0])


def _load_xgb(final_dir: Path, selected_path: Path, test_metrics_path: Path, target_fpr: float) -> Detector:
    sel = json.loads(Path(selected_path).read_text(encoding="utf-8"))
    m = next(m for m in sel["models"] if m["model"] == "xgboost" and m["feature_set"] == "behavioral")
    path = Path(final_dir) / f"xgboost__behavioral__{m['config']}.joblib"
    model = joblib.load(path)
    if list(model.features) != BEHAVIORAL:
        raise ValueError(f"{path.name}: feature list is not BEHAVIORAL")
    thr = final_threshold(test_metrics_path, "xgboost", "behavioral", m["config"], target_fpr)
    p = m["params"]
    return Detector(model, thr, name=f"XGBoost behavioral (depth {p['max_depth']}, w_spoof {p['w_spoof']})",
                    source=f"{path.name} + {Path(test_metrics_path).name}", target_fpr=target_fpr)


def load_detector(final_dir: Path = FINAL_DIR, selected_path: Path = SELECTED_PATH,
                  test_metrics_path: Path = TEST_METRICS_PATH, cv_thresholds_path: Path = CV_THRESHOLDS_PATH,
                  target_fpr: float = PRIMARY_FPR) -> Detector:
    """Final XGBoost behavioral model + its final threshold; on any failure the dt_ratio rule. Never raises."""
    try:
        return _load_xgb(final_dir, selected_path, test_metrics_path, target_fpr)
    except Exception as e:  # noqa: BLE001 - any failure means fallback, shown in the UI
        why = f"XGBoost unavailable ({type(e).__name__}: {e})"
    rule = DtRatioRule(BEHAVIORAL)  # score needs no fit, so no file is required
    try:
        thr, src = final_threshold(test_metrics_path, "rule", "behavioral", "default", target_fpr), "test_metrics.csv"
    except Exception:  # noqa: BLE001
        try:
            d = pd.read_csv(cv_thresholds_path)
            d = d[(d["model"] == "rule") & (d["feature_set"] == "behavioral") & np.isclose(d["target_fpr"], target_fpr)]
            thr, src = float(d["threshold"].mean()), "CV fold-mean threshold"
            if not np.isfinite(thr):
                raise ValueError("no CV threshold")
        except Exception:  # noqa: BLE001
            thr, src = RULE_DEFAULT_THRESHOLD, "built-in threshold"
    return Detector(rule, thr, name="dt_ratio rule (behavioral)", source=f"FALLBACK: {why}; {src}",
                    target_fpr=target_fpr, fallback=True)


# ---------------------------------------------------------------- engine

@dataclass
class TickStats:
    n_frames: int
    feature_us: float  # per frame
    model_us: float    # per frame (batched)
    capped: bool       # time budget ran out before reaching the requested stream time


class ReplayEngine:
    def __init__(self, frames: Frames, detector: Detector, stats: dict, n_alert: int = 3,
                 payload_history_max_age_s: float = BATCH_PAYLOAD_HISTORY_MAX_AGE_S):
        self.frames, self.detector = frames, detector
        self.rel = frames.rel
        self.ext = CausalFeatureExtractor(stats, mode=frames.mode, payload_history_max_age_s=payload_history_max_age_s)
        self.set_n_alert(n_alert)
        self.reset()

    def set_n_alert(self, n: int) -> None:
        self.n_alert = max(1, int(n))

    def reset(self) -> None:
        self.ext.reset()
        self.i = 0
        self.sim_t = 0.0
        self.window: deque = deque()  # flagged frames in the alert window: (t, arb_id, sub, score)
        self.last_trigger_t = -np.inf
        self.incident: dict | None = None
        self.alerts: deque = deque(maxlen=LOG_MAX)
        self.n_alerts = 0
        self.total = np.zeros(len(SUBCLASS_CATS), np.int64)
        self.flagged = np.zeros(len(SUBCLASS_CATS), np.int64)
        self.bins: deque = deque(maxlen=N_BINS)  # closed bins: (t_start, max_score, n_frames, true_attack, alert)
        self.open_bin: list | None = None        # [bin_idx, max_score, n_frames, true_attack, alert]
        self.last_tick = TickStats(0, 0.0, 0.0, False)

    @property
    def done(self) -> bool:
        return self.i >= len(self.frames)

    @property
    def status(self) -> str:
        return "ATTACK" if self.sim_t - self.last_trigger_t <= STATUS_HOLD_S else "NORMAL"

    def advance(self, sim_target: float, budget_s: float = 0.25, chunk: int = 2000) -> TickStats:
        """Process frames with stream time <= sim_target, stopping early when budget_s is used up.

        At least one chunk always runs. When stopped early, the stream clock is clamped to the last
        processed frame, so a backlog never builds up (the effective speed drops instead)."""
        t_start = time.perf_counter()
        end = int(np.searchsorted(self.rel, sim_target, side="right"))
        n, feat_s, model_s, capped = 0, 0.0, 0.0, False
        while self.i < end:
            if n and time.perf_counter() - t_start >= budget_s:
                capped = True
                break
            k = min(end, self.i + chunk)
            f, m = self._process(self.i, k)
            feat_s, model_s, n = feat_s + f, model_s + m, n + (k - self.i)
            self.i = k
        if capped:
            self.sim_t = float(self.rel[self.i - 1])
        else:
            self.sim_t = max(self.sim_t, min(float(sim_target), self.frames.duration))
        self.last_tick = TickStats(n, feat_s / n * 1e6 if n else 0.0, model_s / n * 1e6 if n else 0.0, capped)
        return self.last_tick

    def _process(self, a: int, b: int) -> tuple[float, float]:
        fr = self.frames
        out = np.empty((b - a, len(ALL_FEATURES)), dtype=np.float32)
        step = self.ext.step
        t0 = time.perf_counter()
        for j, (t, i, d, p) in enumerate(zip(fr.ts[a:b].tolist(), fr.ids[a:b].tolist(),
                                             fr.dlc[a:b].tolist(), fr.payload[a:b].tolist())):
            out[j] = step(t, i, d, p)
        t1 = time.perf_counter()
        scores = self.detector.score(out)
        t2 = time.perf_counter()
        self._update(a, b, scores, scores > self.detector.threshold)
        return t1 - t0, t2 - t1

    def _update(self, a: int, b: int, scores: np.ndarray, flags: np.ndarray) -> None:
        rel, sub, ids = self.rel[a:b], self.frames.sub[a:b], self.frames.ids[a:b]
        k = len(SUBCLASS_CATS)
        self.total += np.bincount(sub, minlength=k)
        self.flagged += np.bincount(sub[flags], minlength=k)

        trigger_times = []
        w = self.window
        for j in np.flatnonzero(flags).tolist():
            t = float(rel[j])
            w.append((t, int(ids[j]), int(sub[j]), float(scores[j])))
            while w[0][0] <= t - ALERT_WINDOW_S:
                w.popleft()
            if len(w) >= self.n_alert:
                self._trigger(t, w)
                trigger_times.append(t)

        bidx = np.floor(rel / BIN_S).astype(np.int64)
        starts = np.flatnonzero(np.r_[True, bidx[1:] != bidx[:-1]])
        peak = np.maximum.reduceat(np.clip(scores, -SCORE_CLIP, SCORE_CLIP), starts)
        count = np.diff(np.r_[starts, len(bidx)])
        attack = np.maximum.reduceat((sub != NORMAL).astype(np.int8), starts).astype(bool)
        alert_bins = set(np.floor(np.asarray(trigger_times) / BIN_S).astype(np.int64).tolist())
        for s, pk, c, at in zip(bidx[starts].tolist(), peak.tolist(), count.tolist(), attack.tolist()):
            ob = self.open_bin
            if ob is not None and ob[0] == s:
                ob[1], ob[2], ob[3], ob[4] = max(ob[1], pk), ob[2] + c, ob[3] or at, ob[4] or s in alert_bins
                continue
            if ob is not None:
                self.bins.append((ob[0] * BIN_S, *ob[1:]))
            self.open_bin = [s, pk, c, at, s in alert_bins]

    def _trigger(self, t: float, w: deque) -> None:
        peak = max(x[3] for x in w)
        inc = self.incident
        if inc is not None and t - self.last_trigger_t < ALERT_GAP_S:
            inc["end_s"], inc["peak_score"] = t, max(inc["peak_score"], peak)
        else:
            true_sub = Counter(x[2] for x in w).most_common(1)[0][0]
            self.incident = {"time_s": t, "end_s": t,
                             "can_id": f"0x{Counter(x[1] for x in w).most_common(1)[0][0]:03X}",
                             "predicted": "ATTACK", "peak_score": peak,
                             "true": "Normal (false alarm)" if true_sub == NORMAL else SUBCLASS_CATS[true_sub]}
            self.alerts.append(self.incident)
            self.n_alerts += 1
        self.last_trigger_t = t

    # ------------------------------------------------------------ views for the UI

    def chart_data(self) -> dict[str, np.ndarray]:
        rows = list(self.bins)
        if self.open_bin is not None:
            ob = self.open_bin
            rows.append((ob[0] * BIN_S, *ob[1:]))
        if not rows:
            return {"t": np.empty(0), "score": np.empty(0), "load": np.empty(0, np.int64),
                    "attack": np.empty(0, bool), "alert": np.empty(0, bool)}
        t, score, load, attack, alert = map(np.asarray, zip(*rows))
        return {"t": t, "score": score, "load": load, "attack": attack.astype(bool), "alert": alert.astype(bool)}

    def per_attack(self) -> pd.DataFrame:
        """Frame-level detected / missed per attack SubClass so far."""
        idx = [SUBCLASS_CATS.index(c) for c in ATTACK_NAMES]
        det, tot = self.flagged[idx], self.total[idx]
        return pd.DataFrame({"subclass": ATTACK_NAMES, "detected": det, "missed": tot - det, "total": tot})

    @property
    def normal_fpr(self) -> float:
        n = self.total[NORMAL]
        return float(self.flagged[NORMAL] / n) if n else 0.0
