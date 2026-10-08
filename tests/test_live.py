import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import ALL_FEATURES, BEHAVIORAL, compute_features
from src.live import (ALERT_GAP_S, BIN_S, RULE_DEFAULT_THRESHOLD, Detector, ReplayEngine, frames_from_df,
                      load_detector)
from src.load import BYTE_COLS
from src.train import XGBModel
from tests.test_features import STATS, synthetic_stream

B0 = BEHAVIORAL.index("byte_0")


class RecordingModel:
    """score = byte_0, and every scored row is recorded (to compare live features with batch ones)."""
    features = BEHAVIORAL

    def __init__(self):
        self.rows = []

    def score(self, X):
        self.rows.append(X.copy())
        return X[:, B0].astype(np.float64)


def stream(byte0, dt=0.001, subclass=None, arb_id=0x130):
    """Regular stream with byte_0 controlling the fake score (threshold 100 below)."""
    n = len(byte0)
    df = pd.DataFrame({"mode": "D", "row_idx": np.arange(n), "timestamp": 1.6e9 + dt * np.arange(n),
                       "arb_id": arb_id, "dlc": 8,
                       "subclass": subclass if subclass is not None else "Normal"})
    for c in BYTE_COLS:
        df[c] = 0
    df["byte_0"] = byte0
    return frames_from_df(df, "synthetic")


def engine(frames, n_alert=3, model=None):
    det = Detector(model or RecordingModel(), threshold=100.0, name="fake", source="test")
    return ReplayEngine(frames, det, STATS, n_alert=n_alert)


def test_live_features_equal_batch_features():
    df = synthetic_stream(n=3000).assign(subclass="Normal")
    model = RecordingModel()
    eng = engine(frames_from_df(df, "s"), model=model)
    t = 0.0
    while not eng.done:  # many small, uneven ticks
        t += 0.37
        eng.advance(t, chunk=97)
    live = np.concatenate(model.rows)
    batch = compute_features(df, STATS, payload_history_max_age_s=10.0)[BEHAVIORAL].to_numpy(np.float32)
    np.testing.assert_array_equal(live, batch)


def test_alert_needs_n_flagged_frames_within_window():
    b = np.zeros(1000, int)
    b[[100, 150]] = 200          # 2 flagged within 100 ms: no alert at N=3
    b[[400, 450, 499]] = 200     # 3 within 100 ms (t=0.400..0.499): alert at frame 499
    eng = engine(stream(b))
    eng.advance(10.0)
    assert eng.n_alerts == 1
    assert eng.alerts[0]["time_s"] == pytest.approx(0.499)
    assert eng.flagged.sum() == 5

    eng2 = engine(stream(b), n_alert=2)
    eng2.advance(10.0)
    # frames 100/150 now trigger; the triggers at 0.45 s are < ALERT_GAP_S later, so same incident
    assert eng2.n_alerts == 1 and eng2.alerts[0]["time_s"] == pytest.approx(0.150)


def test_incidents_merge_within_gap_and_split_after():
    b = np.zeros(5000, int)
    b[1000:1100] = 200                       # incident 1 (t 1.0-1.1 s)
    b[1500:1600] = 200                       # 0.4 s later: same incident
    b[1600 + int(ALERT_GAP_S * 1000) + 500:4000] = 200  # > gap later: incident 2
    sub = np.where(b > 0, "Spoofing", "Normal")
    eng = engine(stream(b, subclass=sub, arb_id=0x553))
    eng.advance(100.0, chunk=333)
    assert eng.n_alerts == 2
    first = eng.alerts[0]
    assert first["can_id"] == "0x553" and first["true"] == "Spoofing" and first["predicted"] == "ATTACK"
    assert first["end_s"] == pytest.approx(1.599)


def test_false_alarm_label_and_status_hold():
    b = np.zeros(3000, int)
    b[1000:1010] = 200
    eng = engine(stream(b))
    eng.advance(1.05)
    assert eng.status == "ATTACK" and eng.alerts[0]["true"] == "Normal (false alarm)"
    eng.advance(2.9)
    assert eng.status == "NORMAL"


def test_bins_carry_over_chunk_boundaries():
    b = np.zeros(1000, int)
    b[::7] = 150
    chunked, whole = engine(stream(b)), engine(stream(b))
    chunked.advance(10.0, chunk=37)
    whole.advance(10.0, chunk=10_000)
    cd, ref = chunked.chart_data(), whole.chart_data()
    assert len(cd["t"]) == 10 and cd["load"].sum() == 1000
    np.testing.assert_allclose(cd["t"], np.arange(10) * BIN_S)
    for k in cd:
        np.testing.assert_array_equal(cd[k], ref[k])


def test_budget_clamps_stream_clock():
    eng = engine(stream(np.zeros(10_000, int)))
    st = eng.advance(9.0, budget_s=0.0, chunk=500)
    assert st.capped and st.n_frames == 500 and eng.i == 500
    assert eng.sim_t == pytest.approx(0.499)
    st = eng.advance(eng.sim_t + 0.0005)  # small step: not capped, clock moves to the target
    assert not st.capped and eng.sim_t == pytest.approx(0.4995)


def test_per_attack_counts_and_reset():
    b = np.zeros(2000, int)
    sub = np.array(["Normal"] * 2000, dtype=object)
    sub[100:200] = "Fuzzing"
    b[100:150] = 200
    b[1000] = 200  # one Normal false positive
    eng = engine(stream(b, subclass=sub))
    eng.advance(10.0)
    pa = eng.per_attack().set_index("subclass")
    assert pa.loc["Fuzzing", "detected"] == 50 and pa.loc["Fuzzing", "missed"] == 50
    assert pa.loc["Flooding", "total"] == 0
    assert eng.normal_fpr == pytest.approx(1 / 1900)
    eng.reset()
    assert eng.i == 0 and eng.n_alerts == 0 and eng.total.sum() == 0
    cd = eng.chart_data()  # empty before any frame; masks must stay usable as boolean indices
    assert not len(cd["t"]) and cd["attack"].dtype == bool and not len(cd["t"][cd["attack"]])


# ---------------------------------------------------------------- load_detector

def test_fallback_to_rule_with_cv_threshold(tmp_path):
    pd.DataFrame({"fold": ["a", "b"], "model": "rule", "feature_set": "behavioral", "config": "default",
                  "target_fpr": 0.001, "threshold": [-0.02, -0.04]}).to_csv(tmp_path / "cv.csv", index=False)
    det = load_detector(final_dir=tmp_path, selected_path=tmp_path / "missing.json",
                        test_metrics_path=tmp_path / "missing.csv", cv_thresholds_path=tmp_path / "cv.csv")
    assert det.fallback and det.threshold == pytest.approx(-0.03) and "FALLBACK" in det.source


def test_fallback_without_any_file(tmp_path):
    det = load_detector(final_dir=tmp_path, selected_path=tmp_path / "x", test_metrics_path=tmp_path / "y",
                        cv_thresholds_path=tmp_path / "z")
    assert det.fallback and det.threshold == RULE_DEFAULT_THRESHOLD
    X = np.zeros((3, len(ALL_FEATURES)), np.float32)
    assert det.score(X).shape == (3,)


def test_loads_final_xgboost_and_threshold(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.random((200, len(BEHAVIORAL))).astype(np.float32), rng.integers(0, 2, 200)
    model = XGBModel(BEHAVIORAL, max_depth=2, n_estimators=3).fit(X, y, np.where(y, "Spoofing", "Normal"))
    joblib.dump(model, tmp_path / "xgboost__behavioral__max_depth2_w_spoof1.joblib")
    (tmp_path / "sel.json").write_text(json.dumps({"models": [
        {"model": "xgboost", "feature_set": "behavioral", "config": "max_depth2_w_spoof1",
         "params": {"max_depth": 2, "w_spoof": 1}}]}))
    pd.DataFrame({"model": "xgboost", "feature_set": "behavioral", "config": "max_depth2_w_spoof1",
                  "target_fpr": [0.001, 0.001, 0.005], "threshold": [1.25, 1.25, 0.5]}
                 ).to_csv(tmp_path / "tm.csv", index=False)
    det = load_detector(final_dir=tmp_path, selected_path=tmp_path / "sel.json", test_metrics_path=tmp_path / "tm.csv")
    assert not det.fallback and det.threshold == 1.25 and "XGBoost" in det.name
    X_all = np.zeros((4, len(ALL_FEATURES)), np.float32)
    np.testing.assert_allclose(det.score(X_all), model.score(X_all[:, det.feature_idx]))
