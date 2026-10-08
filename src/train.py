"""Phase 4 training: dt_ratio rule, Isolation Forest and XGBoost on BEHAVIORAL / ID_INCLUSIVE features.

Leave-one-day-out CV (splits.json cv_default) picks one config per (model, feature set) by mean macro
recall over the 4 attack SubClasses at PRIMARY_FPR (ties: all-attack PR-AUC). Only train files are
loaded here; held-out files need final=True, which only the post-approval final step passes.

Training rows per model (all from the fold's training files only):
- rule, xgboost: every attack frame + Normal subsampled to NORMAL_PER_ATTACK:1 per attack file
  (Normal-only _0 files kept whole). The per-file seed makes a file's subsample identical in every fold.
- iforest: Normal frames more than NEAR_ATTACK_MARGIN_S from any attack frame (near-attack Normal
  frames have inflated bus counts, notebooks/02_features.ipynb section 5).
- reference R (q-transfer, src/evaluate.py): ALL Normal frames of the training files, whatever the
  model fit on, so q means the same thing in every fold and in the final model.
Usage:
  python -m src.train cv [--smoke]   # LODO CV -> reports/phase4/*.csv, selected.json; models/cv/
  python -m src.train latency        # live per-frame latency of the selected configs
  python -m src.train final --rehearse  # final code path on the first LODO fold; writes nothing
  python -m src.train final          # approved configs: refit on all train files, score held-out ONCE
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.evaluate import (ALL_ATTACKS, ATTACKS, PRIMARY_FPR, TARGET_FPRS, batch_throughput,
                          evaluate_scores, macro_recall, measure_latency, quantile_of, threshold_at_fpr,
                          threshold_at_quantile)
from src.features import (ALL_FEATURES, BATCH_PAYLOAD_HISTORY_MAX_AGE_S, BEHAVIORAL, FEATURES_DIR,
                          ID_INCLUSIVE, META_COLS, MODE_CODE, CausalFeatureExtractor)
from src.load import BYTE_COLS, ROOT, SUBCLASS_CATS
from src.train_stats import load_splits, load_train_stats

MODELS_DIR = ROOT / "models"
CV_DIR = MODELS_DIR / "cv"
FINAL_DIR = MODELS_DIR / "final"
REPORT_DIR = ROOT / "reports" / "phase4"
SELECTED_PATH = REPORT_DIR / "selected.json"
TEST_METRICS_PATH = REPORT_DIR / "test_metrics.csv"

FEATURE_SETS = {"behavioral": BEHAVIORAL, "id_inclusive": ID_INCLUSIVE}
LOAD_META = ["timestamp", "subclass", "is_attack"]  # timestamp: near-attack mask only, never a feature
NEAR_ATTACK_MARGIN_S = 1.0
NORMAL_PER_ATTACK = 4
SEED = 0
KEY = ["model", "feature_set", "config"]

if set(META_COLS) & set(ALL_FEATURES):
    raise RuntimeError("META_COLS overlap the feature columns")


# ---------------------------------------------------------------- data

def load_feature_files(stems, final: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Feature files -> (X float32 in ALL_FEATURES order, meta: source_file/timestamp/subclass/is_attack)."""
    stems = list(stems)
    splits = load_splits()
    held_out = set(splits["test"]) | set(splits["extra_test"])
    unknown = set(stems) - set(splits["train"]) - held_out
    if unknown:
        raise ValueError(f"unknown stems {sorted(unknown)}")
    if held_out & set(stems) and not final:
        raise PermissionError(f"held-out files {sorted(held_out & set(stems))} need final=True "
                              "(only after the validation results are approved)")
    Xs, metas = [], []
    for i, stem in enumerate(stems):
        d = pd.read_parquet(FEATURES_DIR / f"{stem}.parquet", columns=LOAD_META + ALL_FEATURES)
        Xs.append(d[ALL_FEATURES].to_numpy(np.float32))
        metas.append(pd.DataFrame({
            "source_file": pd.Categorical.from_codes(np.full(len(d), i), categories=stems),
            "timestamp": d["timestamp"].to_numpy(np.float64),
            "subclass": pd.Categorical(d["subclass"].astype(str), categories=SUBCLASS_CATS),
            "is_attack": d["is_attack"].to_numpy(np.int8)}))
    return np.concatenate(Xs), pd.concat(metas, ignore_index=True)


def near_attack_mask(meta: pd.DataFrame, margin_s: float = NEAR_ATTACK_MARGIN_S) -> np.ndarray:
    """True where a frame is within margin_s of an attack frame of the same file (attack frames included)."""
    near = np.zeros(len(meta), dtype=bool)
    ts, att = meta["timestamp"].to_numpy(), meta["is_attack"].to_numpy() == 1
    for ix in meta.groupby("source_file", observed=True).indices.values():
        att_t = np.sort(ts[ix][att[ix]])
        if not len(att_t):
            continue
        t = ts[ix]
        j = np.searchsorted(att_t, t)
        d_prev = np.where(j > 0, t - att_t[np.maximum(j - 1, 0)], np.inf)
        d_next = np.where(j < len(att_t), att_t[np.minimum(j, len(att_t) - 1)] - t, np.inf)
        near[ix] = np.minimum(d_prev, d_next) <= margin_s
    return near


def reference_normal_mask(meta: pd.DataFrame) -> np.ndarray:
    """R for the q-transfer: every Normal frame of the given (training) rows. Defined only here."""
    return meta["is_attack"].to_numpy() == 0


def anomaly_train_mask(meta: pd.DataFrame) -> np.ndarray:
    return reference_normal_mask(meta) & ~near_attack_mask(meta)


def supervised_train_mask(meta: pd.DataFrame, ratio: int = NORMAL_PER_ATTACK, seed: int = SEED) -> np.ndarray:
    """All attack frames + Normal subsampled to ratio:1 per attack file; Normal-only files kept whole."""
    att = meta["is_attack"].to_numpy() == 1
    keep = att.copy()
    for stem, ix in meta.groupby("source_file", observed=True).indices.items():
        normal_ix, n_att = ix[~att[ix]], int(att[ix].sum())
        if n_att == 0 or len(normal_ix) <= ratio * n_att:
            keep[normal_ix] = True
        else:
            rng = np.random.default_rng([seed, zlib.crc32(str(stem).encode())])
            keep[rng.choice(normal_ix, ratio * n_att, replace=False)] = True
    return keep


# ---------------------------------------------------------------- models (score: higher = more attack-like)

class Detector:
    kind = ""

    def __init__(self, features, **params):
        self.features, self.params = list(features), params

    def fit(self, X, y, subclass):
        raise NotImplementedError

    def score(self, X) -> np.ndarray:
        raise NotImplementedError

    def score_one(self, x) -> np.ndarray:
        return self.score(x)


class DtRatioRule(Detector):
    """score = -dt_ratio (early frames are suspicious); first frame of an ID scores lowest.
    With ID features, an ID never seen in Pre_train_*_0 scores highest."""
    kind = "rule"
    LOW, HIGH = np.float32(-1e9), np.float32(1e9)

    def __init__(self, features):
        super().__init__(features)
        self._dt = self.features.index("dt_ratio")
        self._id = self.features.index("id_seen_in_training") if "id_seen_in_training" in self.features else None

    def fit(self, X, y, subclass=None):
        self.train_auc = float(roc_auc_score(y, self.score(X)))
        if self.train_auc <= 0.5:
            raise RuntimeError(f"dt_ratio direction flipped on training data (AUC {self.train_auc:.3f})")
        return self

    def score(self, X):
        dt = X[:, self._dt]
        s = np.where(dt < 0, self.LOW, -dt).astype(np.float32)
        if self._id is not None:
            s[X[:, self._id] == 0] = self.HIGH
        return s


class IForestModel(Detector):
    kind = "iforest"

    def __init__(self, features, n_estimators=100, max_samples=256):
        super().__init__(features, n_estimators=n_estimators, max_samples=max_samples)
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                     random_state=SEED, n_jobs=-1)
        self._single = None

    def fit(self, X, y=None, subclass=None):
        self.model.fit(X)
        self._single = None
        return self

    def score(self, X):
        return -self.model.score_samples(X)

    def score_one(self, x):
        # Single-threaded copy: joblib dispatch would dominate one-row latency.
        if self._single is None:
            self._single = copy.deepcopy(self.model).set_params(n_jobs=1)
        return -self._single.score_samples(x)


class XGBModel(Detector):
    kind = "xgboost"

    def __init__(self, features, max_depth=6, w_spoof=1.0, n_estimators=300, learning_rate=0.1):
        super().__init__(features, max_depth=max_depth, w_spoof=w_spoof,
                         n_estimators=n_estimators, learning_rate=learning_rate)
        self.model = XGBClassifier(tree_method="hist", objective="binary:logistic", max_depth=max_depth,
                                   n_estimators=n_estimators, learning_rate=learning_rate, subsample=0.8,
                                   colsample_bytree=0.8, n_jobs=4, random_state=SEED)
        self._single = None

    def fit(self, X, y, subclass):
        w = np.where(np.asarray(subclass) == "Spoofing", self.params["w_spoof"], 1.0).astype(np.float32)
        self.model.fit(X, y, sample_weight=w)
        self._single = None
        return self

    def score(self, X):
        # Raw margin: probabilities saturate near 1.0 and tie exactly in the low-FPR tail.
        return self.model.get_booster().inplace_predict(X, predict_type="margin")

    def score_one(self, x):
        # Single-threaded booster copy (same trees, same output): thread dispatch dominates one row.
        if getattr(self, "_single", None) is None:
            self._single = self.model.get_booster().copy()
            self._single.set_param({"nthread": 1})
        return self._single.inplace_predict(x, predict_type="margin")

    def importance(self) -> pd.Series:
        gain = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series({f: gain.get(f"f{i}", 0.0) for i, f in enumerate(self.features)}, name="gain")


MODEL_CLASSES = {c.kind: c for c in (DtRatioRule, IForestModel, XGBModel)}
GRIDS = {
    "rule": [{}],
    "iforest": [{"n_estimators": n, "max_samples": m} for n in (100, 300) for m in (256, 4096)],
    "xgboost": [{"max_depth": d, "w_spoof": w} for d in (4, 6, 8) for w in (1, 5)],
}
TRAIN_ROWS = {"rule": "supervised", "iforest": "anomaly", "xgboost": "supervised"}


def config_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in params.items()) or "default"


def model_path(fold: str, kind: str, feature_set: str, cfg: str):
    return CV_DIR / f"{fold}__{kind}__{feature_set}__{cfg}.joblib"


# ---------------------------------------------------------------- CV

def run_cv(kinds=tuple(GRIDS), feature_sets=tuple(FEATURE_SETS), smoke: bool = False) -> dict:
    """LODO CV on train files. smoke=True: first fold, first config per model, nothing written."""
    splits = load_splits()
    folds = splits["cv_schemes"][splits["cv_default"]][:1 if smoke else None]
    held_out = set(splits["test"]) | set(splits["extra_test"])
    X_all, meta = load_feature_files(splits["train"])
    is_attack, subclass = meta["is_attack"].to_numpy(), meta["subclass"].to_numpy()
    if not smoke:
        CV_DIR.mkdir(parents=True, exist_ok=True)

    metric_parts, thr_rows, imp_parts = [], [], []
    for fold in folds:
        name = fold["name"]
        if (set(fold["train"]) | set(fold["val"])) & held_out or set(fold["train"]) & set(fold["val"]):
            raise RuntimeError(f"{name}: fold overlaps held-out files or itself")
        src = meta["source_file"]
        tr_idx = np.flatnonzero(src.isin(fold["train"]).to_numpy())
        va_idx = np.flatnonzero(src.isin(fold["val"]).to_numpy())
        tr_meta = meta.iloc[tr_idx].reset_index(drop=True)
        rows = {"supervised": tr_idx[supervised_train_mask(tr_meta)],
                "anomaly": tr_idx[anomaly_train_mask(tr_meta)],
                "reference": tr_idx[reference_normal_mask(tr_meta)]}
        va_sub = subclass[va_idx]
        va_normal = va_sub == "Normal"
        print(f"\n== {name}: train={fold['train']} val={fold['val']}\n"
              f"   rows: val={len(va_idx):,} " + " ".join(f"{k}={len(v):,}" for k, v in rows.items()))
        if not smoke:
            meta.iloc[va_idx][["source_file", "subclass"]].to_parquet(CV_DIR / f"{name}__val_meta.parquet")

        for fs in feature_sets:
            cols = [ALL_FEATURES.index(f) for f in FEATURE_SETS[fs]]
            take = lambda idx: X_all[np.ix_(idx, cols)]
            X_va, X_ref = take(va_idx), take(rows["reference"])
            for kind in kinds:
                fit_idx = rows[TRAIN_ROWS[kind]]
                X_fit = take(fit_idx)
                val_scores = {}
                for params in GRIDS[kind][:1 if smoke else None]:
                    cfg = config_id(params)
                    model = MODEL_CLASSES[kind](FEATURE_SETS[fs], **params)
                    t0 = time.perf_counter()
                    model.fit(X_fit, is_attack[fit_idx], subclass[fit_idx])
                    t1 = time.perf_counter()
                    s_va = model.score(X_va)
                    t2 = time.perf_counter()
                    s_ref = model.score(X_ref)

                    thresholds = {f: threshold_at_fpr(s_va[va_normal], f) for f in TARGET_FPRS}
                    table = evaluate_scores(s_va, va_sub, thresholds)
                    key = {"fold": name, "model": kind, "feature_set": fs, "config": cfg}
                    metric_parts.append(table.assign(**key))
                    thr_rows += [{**key, "target_fpr": f, "threshold": t, "q": quantile_of(s_ref, t),
                                  "n_fit": len(fit_idx), "fit_s": t1 - t0, "score_val_s": t2 - t1}
                                 for f, t in thresholds.items()]
                    if kind == "xgboost":
                        imp_parts.append(model.importance().rename_axis("feature").reset_index().assign(**key))
                    print(f"   {fs:<12} {kind:<8} {cfg:<34} macro recall@{PRIMARY_FPR:.1%}="
                          f"{macro_recall(table):.3f}  fit {t1 - t0:5.0f}s  score {t2 - t1:4.0f}s")
                    if not smoke:
                        joblib.dump(model, model_path(name, kind, fs, cfg))
                    val_scores[cfg] = s_va
                if not smoke:
                    pd.DataFrame(val_scores).to_parquet(CV_DIR / f"{name}__{kind}__{fs}__val_scores.parquet")

    out = {"metrics": pd.concat(metric_parts, ignore_index=True), "thresholds": pd.DataFrame(thr_rows),
           "importance": pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()}
    out["summary"] = summarize_cv(out["metrics"], out["thresholds"])
    return out


def summarize_cv(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, feature_set, config): mean/worst-fold macro recall and all-attack PR-AUC at PRIMARY_FPR."""
    prim = metrics[metrics["target_fpr"] == PRIMARY_FPR]
    per_fold = (prim[prim["subclass"].isin(ATTACKS)].groupby(KEY + ["fold"])["recall"].mean()
                .rename("macro_recall").to_frame()
                .join(prim[prim["subclass"] == ALL_ATTACKS].set_index(KEY + ["fold"])[["pr_auc", "fpr"]]))
    q = thresholds[thresholds["target_fpr"] == PRIMARY_FPR].groupby(KEY)["q"]
    summary = per_fold.groupby(KEY).agg(macro_recall_mean=("macro_recall", "mean"),
                                        macro_recall_worst=("macro_recall", "min"),
                                        pr_auc_all_mean=("pr_auc", "mean"), val_fpr_mean=("fpr", "mean"),
                                        n_folds=("pr_auc", "size"))
    summary["q_mean"], summary["q_spread"] = q.mean(), q.max() - q.min()
    return summary.reset_index().sort_values(["model", "feature_set", "macro_recall_mean", "pr_auc_all_mean"],
                                             ascending=[True, True, False, False], ignore_index=True)


def select_configs(summary: pd.DataFrame, thresholds: pd.DataFrame) -> dict:
    """Best config per (model, feature_set): mean macro recall at PRIMARY_FPR, ties by all-attack PR-AUC."""
    params_by_cfg = {(k, config_id(p)): p for k, grid in GRIDS.items() for p in grid}
    best = summary.groupby(["model", "feature_set"], sort=False).head(1)  # summary is already ranked
    chosen = []
    for r in best.itertuples(index=False):
        t = thresholds[(thresholds["model"] == r.model) & (thresholds["feature_set"] == r.feature_set)
                       & (thresholds["config"] == r.config)]
        chosen.append({"model": r.model, "feature_set": r.feature_set, "config": r.config,
                       "params": params_by_cfg[(r.model, r.config)],
                       "macro_recall_mean": r.macro_recall_mean, "macro_recall_worst": r.macro_recall_worst,
                       "q": {str(f): float(g["q"].mean()) for f, g in t.groupby("target_fpr")},
                       "q_per_fold": {str(f): dict(zip(g["fold"], g["q"].astype(float)))
                                      for f, g in t.groupby("target_fpr")}})
    return {"approved": False, "primary_fpr": PRIMARY_FPR, "cv_scheme": load_splits()["cv_default"],
            "criterion": "mean over folds of macro recall (4 SubClasses) at primary_fpr; ties: all-attack PR-AUC",
            "models": chosen}


def load_selected(require_approved: bool = True) -> dict:
    sel = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    if require_approved and sel.get("approved") is not True:
        raise PermissionError(f"{SELECTED_PATH} is not approved; review validation results first")
    return sel


def write_cv_outputs(out: dict, force: bool = False) -> None:
    if SELECTED_PATH.exists() and load_selected(require_approved=False).get("approved") and not force:
        raise PermissionError("selected.json is already approved; refusing to overwrite (use --force)")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "thresholds", "importance", "summary"):
        out[name].to_csv(REPORT_DIR / f"cv_{name}.csv", index=False)
    sel = select_configs(out["summary"], out["thresholds"])
    SELECTED_PATH.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- latency

def run_latency(n_frames: int = 5000) -> pd.DataFrame:
    """Selected configs, first LODO fold's models, replayed on that fold's first val file (unseen by them)."""
    splits = load_splits()
    fold = splits["cv_schemes"][splits["cv_default"]][0]
    stem = fold["val"][0]
    frames = pd.read_parquet(FEATURES_DIR / f"{stem}.parquet", columns=["timestamp", "arb_id", "dlc", "mode"] + BYTE_COLS)
    mode = {v: k for k, v in MODE_CODE.items()}[int(frames["mode"].iloc[0])]
    stats = load_train_stats()
    X_file, _ = load_feature_files([stem])
    rows = []
    for m in load_selected(require_approved=False)["models"]:
        model = joblib.load(model_path(fold["name"], m["model"], m["feature_set"], m["config"]))
        idx = [ALL_FEATURES.index(f) for f in model.features]
        ext = CausalFeatureExtractor(stats, mode=mode, payload_history_max_age_s=BATCH_PAYLOAD_HISTORY_MAX_AGE_S)
        lat = measure_latency(frames, ext, model.score_one, idx, n_frames=n_frames)
        rows.append({"model": m["model"], "feature_set": m["feature_set"], "config": m["config"], **lat,
                     "batch_frames_per_s": batch_throughput(model.score, X_file[:, idx])})
        print(rows[-1])
    out = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "latency.csv", index=False)
    return out


# ---------------------------------------------------------------- final fit + held-out evaluation

def run_final(rehearse: bool = False, force: bool = False) -> pd.DataFrame:
    """Refit the selected configs on all train files and score the held-out files ONCE.

    Thresholds come from the CV fold-mean q applied to the refit model's own reference Normal scores
    (q-transfer), so no held-out label touches them. rehearse=True runs the same path with the first LODO
    fold standing in (its train files fit, its val files are the "test"): no approval needed, no held-out
    file read, nothing written.
    """
    splits = load_splits()
    if rehearse:
        fold = splits["cv_schemes"][splits["cv_default"]][0]
        train_stems, eval_sets = fold["train"], {"rehearsal": fold["val"]}
        sel = load_selected(require_approved=False)
    else:
        if TEST_METRICS_PATH.exists() and not force:
            raise PermissionError(f"{TEST_METRICS_PATH} exists: held-out files were already evaluated")
        sel = load_selected(require_approved=True)
        train_stems, eval_sets = splits["train"], {"test": splits["test"], "extra_test": splits["extra_test"]}
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

    X_all, meta = load_feature_files(train_stems)
    is_attack, subclass = meta["is_attack"].to_numpy(), meta["subclass"].to_numpy()
    rows = {"supervised": np.flatnonzero(supervised_train_mask(meta)),
            "anomaly": np.flatnonzero(anomaly_train_mask(meta)),
            "reference": np.flatnonzero(reference_normal_mask(meta))}
    print(f"train={train_stems}\n   rows: " + " ".join(f"{k}={len(v):,}" for k, v in rows.items()))
    fitted = []
    for m in sel["models"]:
        kind, fs, cfg = m["model"], m["feature_set"], m["config"]
        cols = [ALL_FEATURES.index(f) for f in FEATURE_SETS[fs]]
        fit_idx = rows[TRAIN_ROWS[kind]]
        model = MODEL_CLASSES[kind](FEATURE_SETS[fs], **m["params"])
        t0 = time.perf_counter()
        model.fit(X_all[np.ix_(fit_idx, cols)], is_attack[fit_idx], subclass[fit_idx])
        s_ref = model.score(X_all[np.ix_(rows["reference"], cols)])
        thresholds = {f: threshold_at_quantile(s_ref, m["q"][str(f)]) for f in TARGET_FPRS}
        print(f"   fit {fs:<12} {kind:<8} {cfg:<34} {time.perf_counter() - t0:5.0f}s")
        if not rehearse:
            joblib.dump(model, FINAL_DIR / f"{kind}__{fs}__{cfg}.joblib")
        fitted.append((m, cols, model, thresholds))
    del X_all, meta

    parts = []
    for eval_set, stems in eval_sets.items():
        X, emeta = load_feature_files(stems, final=not rehearse)
        sub = emeta["subclass"].to_numpy()
        groups = {"pooled": np.arange(len(X))} if len(stems) > 1 else {}
        groups.update({str(k): v for k, v in emeta.groupby("source_file", observed=True).indices.items()})
        print(f"\n== {eval_set}: {stems} ({len(X):,} frames)")
        for m, cols, model, thresholds in fitted:
            s = model.score(X[:, cols])
            for name, ix in groups.items():
                table = evaluate_scores(s[ix], sub[ix], thresholds)
                parts.append(table.assign(eval_set=eval_set, file=name, model=m["model"],
                                          feature_set=m["feature_set"], config=m["config"]))
            prim = parts[-len(groups)]  # pooled (or the only file)
            prim = prim[prim["target_fpr"] == PRIMARY_FPR]
            print(f"   {m['feature_set']:<12} {m['model']:<8} macro recall@{PRIMARY_FPR:.1%}="
                  f"{prim[prim['subclass'].isin(ATTACKS)]['recall'].mean():.3f}  fpr={prim['fpr'].iloc[0]:.5f}")
        del X
        if not rehearse:  # written after each set so a crash later cannot lose an evaluation already made
            pd.concat(parts, ignore_index=True).to_csv(TEST_METRICS_PATH, index=False)
    return pd.concat(parts, ignore_index=True)


def _peak_mb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except ImportError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--smoke", action="store_true", help="first fold, first config per model, write nothing")
    c.add_argument("--force", action="store_true", help="overwrite an approved selected.json")
    sub.add_parser("latency")
    f = sub.add_parser("final")
    f.add_argument("--rehearse", action="store_true", help="first LODO fold stands in for train/test; writes nothing")
    f.add_argument("--force", action="store_true", help="re-evaluate although test_metrics.csv exists")
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.cmd == "cv":
        res = run_cv(smoke=args.smoke)
        print("\n" + res["summary"].to_string())
        if not args.smoke:
            write_cv_outputs(res, force=args.force)
            print(f"-> {REPORT_DIR}, {CV_DIR}")
    elif args.cmd == "final":
        run_final(rehearse=args.rehearse, force=args.force)
        if not args.rehearse:
            print(f"-> {TEST_METRICS_PATH}, {FINAL_DIR}")
    else:
        run_latency()
    print(f"\n{time.perf_counter() - t0:.0f}s, peak memory {_peak_mb() or float('nan'):.0f} MB")


if __name__ == "__main__":
    # Run via the imported module so pickled models reference src.train.<Class>, not __main__.<Class>
    # (otherwise the notebook and src.evaluate cannot unpickle them).
    from src.train import main as _main
    _main()
