"""Thresholds, metrics and latency for the Phase 4 detectors (src/train.py).

Conventions
- score(X) is higher = more attack-like for every model. A frame is flagged when score > t, so ties at
  the threshold are never flagged and the FPR target is never exceeded on the set t was chosen on.
- Per-SubClass metrics compare that SubClass with ALL Normal frames at their real ratio (never resampled):
  recall = TP_sc / n_sc, precision = TP_sc / (TP_sc + FP), FPR = FP / n_normal.
- q-transfer: q = fraction of a reference Normal population scoring <= t. The same population (all Normal
  frames of the model's training files, src.train.reference_normal_mask) is used in every CV fold and for
  the final model, so q means the same thing in both places.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.load import BYTE_COLS

ATTACKS = ["Flooding", "Spoofing", "Replay", "Fuzzing"]
ALL_ATTACKS = "All attacks"
TARGET_FPRS = (0.001, 0.005, 0.01)
PRIMARY_FPR = 0.001  # drives threshold choice and model selection


def threshold_at_fpr(normal_scores, target_fpr: float) -> float:
    """Smallest t with mean(normal_scores > t) <= target_fpr."""
    s = np.sort(np.asarray(normal_scores, dtype=np.float64))
    if not len(s):
        raise ValueError("no Normal scores")
    k = int(np.floor(target_fpr * len(s) + 1e-9))  # false positives allowed
    return float(s[len(s) - 1 - k]) if k < len(s) else -np.inf


def quantile_of(ref_scores, t: float) -> float:
    """q = fraction of ref_scores <= t."""
    s = np.asarray(ref_scores, dtype=np.float64)
    return float(np.count_nonzero(s <= t) / len(s))


def threshold_at_quantile(ref_scores, q: float) -> float:
    """Inverse of quantile_of: the ref score at rank ceil(q * n), so quantile_of(ref, result) == q on ref."""
    s = np.sort(np.asarray(ref_scores, dtype=np.float64))
    j = int(np.ceil(q * len(s) - 1e-9))
    return float(s[j - 1]) if j > 0 else -np.inf


def _subclass_array(subclass) -> np.ndarray:
    return np.asarray(subclass, dtype=object).astype(str)


def _positive_masks(sub: np.ndarray) -> dict[str, np.ndarray]:
    normal = sub == "Normal"
    return {**{name: sub == name for name in ATTACKS}, ALL_ATTACKS: ~normal}


def metrics_at_threshold(scores, subclass, t: float) -> pd.DataFrame:
    """One row per attack SubClass (+ All attacks): n, tp, fp, recall, precision, f1, fpr for score > t."""
    sub = _subclass_array(subclass)
    flag = np.asarray(scores) > t
    normal = sub == "Normal"
    fp = int(np.count_nonzero(flag & normal))
    fpr = fp / int(normal.sum())
    rows = []
    for name, pos in _positive_masks(sub).items():
        n, tp = int(pos.sum()), int(np.count_nonzero(flag & pos))
        recall = tp / n if n else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if n else np.nan)
        rows.append({"subclass": name, "n": n, "tp": tp, "fp": fp, "recall": recall,
                     "precision": precision, "f1": f1, "fpr": fpr})
    return pd.DataFrame(rows)


def auc_metrics(scores, subclass) -> pd.DataFrame:
    """Threshold-free ROC-AUC and PR-AUC (average precision) of each SubClass vs all Normal."""
    scores = np.asarray(scores, dtype=np.float64)
    sub = _subclass_array(subclass)
    normal = sub == "Normal"
    rows = []
    for name, pos in _positive_masks(sub).items():
        if not pos.any():
            rows.append({"subclass": name, "roc_auc": np.nan, "pr_auc": np.nan})
            continue
        m = normal | pos
        rows.append({"subclass": name, "roc_auc": roc_auc_score(pos[m], scores[m]),
                     "pr_auc": average_precision_score(pos[m], scores[m])})
    return pd.DataFrame(rows)


def evaluate_scores(scores, subclass, thresholds: dict[float, float]) -> pd.DataFrame:
    """Long table: one row per (target_fpr, subclass) with threshold metrics plus AUCs."""
    parts = [metrics_at_threshold(scores, subclass, t).assign(target_fpr=f, threshold=t)
             for f, t in thresholds.items()]
    return pd.concat(parts, ignore_index=True).merge(auc_metrics(scores, subclass), on="subclass")


def macro_recall(table: pd.DataFrame, target_fpr: float = PRIMARY_FPR) -> float:
    """Mean recall over the 4 attack SubClasses at target_fpr (the model-selection criterion)."""
    rows = table[(table["target_fpr"] == target_fpr) & table["subclass"].isin(ATTACKS)]
    return float(rows["recall"].mean())


def _percentiles_us(ns: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_p{q}_us": float(np.percentile(ns, q) / 1e3) for q in (50, 95, 99)}


def measure_latency(frames: pd.DataFrame, extractor, score_one, feature_idx,
                    n_frames: int = 5000, warmup: int = 500) -> dict:
    """Live per-frame latency: CausalFeatureExtractor.step() then single-row score(), timed separately.

    frames: consecutive frames of one stream in stream order (timestamp, arb_id, dlc, byte_0..7).
    score_one: callable on a (1, n_features) float32 array. The first `warmup` frames run untimed
    (they fill the per-ID history and warm caches).
    """
    rows = frames.iloc[:warmup + n_frames]
    if len(rows) < warmup + n_frames:
        raise ValueError("not enough frames")
    ts = rows["timestamp"].to_numpy(np.float64).tolist()
    ids = rows["arb_id"].to_numpy().tolist()
    dlcs = rows["dlc"].to_numpy().tolist()
    payloads = list(zip(*(rows[c].to_numpy().tolist() for c in BYTE_COLS)))
    idx = list(feature_idx)
    x = np.empty((1, len(idx)), dtype=np.float32)
    t_feat = np.empty(n_frames, dtype=np.int64)
    t_model = np.empty(n_frames, dtype=np.int64)

    extractor.reset()
    clock = time.perf_counter_ns
    for i in range(warmup + n_frames):
        t0 = clock()
        out = extractor.step(ts[i], ids[i], dlcs[i], payloads[i])
        t1 = clock()
        x[0] = [out[j] for j in idx]
        score_one(x)
        t2 = clock()
        if i >= warmup:
            t_feat[i - warmup], t_model[i - warmup] = t1 - t0, t2 - t1
    return {"n_frames": n_frames, **_percentiles_us(t_feat, "feature"),
            **_percentiles_us(t_model, "model"), **_percentiles_us(t_feat + t_model, "total")}


def batch_throughput(score_fn, X: np.ndarray) -> float:
    """Frames per second of vectorised score() over X (one pass)."""
    t0 = time.perf_counter()
    score_fn(X)
    return len(X) / (time.perf_counter() - t0)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
