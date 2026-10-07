import numpy as np
import pandas as pd
import pytest

from src.features import (ALL_FEATURES, BEHAVIORAL, BUS_WINDOWS_S, ID_INCLUSIVE, ID_WINDOW_S,
                          INPUT_COLS, META_COLS, CausalFeatureExtractor, compute_features)
from src.load import BYTE_COLS, PROCESSED_DIR

KNOWN_A, KNOWN_B, UNKNOWN_Z = 0x130, 0x260, 0x7DC
GLOBAL_REF = 0.01
# Known ID A has median period == global ref, so swapping A for an unknown ID must not change dt_ratio.
STATS = {"known_ids": frozenset({KNOWN_A, KNOWN_B}), "global_ref_period_s": GLOBAL_REF,
         "per_id": {KNOWN_A: {"median_period_s": GLOBAL_REF}, KNOWN_B: {"median_period_s": 0.02}}}

REAL_FILE = PROCESSED_DIR / "Pre_train_S_1.parquet"
needs_real = pytest.mark.skipif(not REAL_FILE.exists(), reason="processed data not available")


def synthetic_stream(n=3000, seed=0, ids=(KNOWN_A, KNOWN_B, 0x000, UNKNOWN_Z)):
    """Random frames with timestamp ties, repeated payloads, short DLCs and an unknown ID."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.0, 0.0002, 0.001, 0.004, 0.03], size=n)
    dlc = rng.choice([0, 2, 8, 8, 8], size=n)
    pool = rng.integers(0, 256, size=(5, 8))  # small pool -> exact payload repeats
    payload = np.where(rng.random((n, 1)) < 0.5, pool[rng.integers(0, 5, n)], rng.integers(0, 256, (n, 8)))
    payload = np.where(np.arange(8) < dlc[:, None], payload, -1)
    df = pd.DataFrame({"mode": "S", "row_idx": np.arange(n, dtype=np.int32),
                       "timestamp": 1.6e9 + np.cumsum(dt), "arb_id": rng.choice(ids, size=n).astype(np.int16),
                       "dlc": dlc.astype(np.int8)})
    for i, c in enumerate(BYTE_COLS):
        df[c] = payload[:, i].astype(np.int16)
    df["is_attack"] = rng.integers(0, 2, n).astype(np.int8)
    return df


@pytest.fixture(scope="module")
def real_slice():
    """30k real rows of Pre_train_S_1 starting shortly before the first attack frame."""
    df = pd.read_parquet(REAL_FILE, columns=INPUT_COLS)
    i0 = max(0, int(np.argmax(df["is_attack"].to_numpy())) - 10_000)
    out = df.iloc[i0:i0 + 30_000].reset_index(drop=True)
    assert out["is_attack"].any()
    return out


@pytest.fixture(scope="module")
def real_stats():
    from src.train_stats import STATS_PATH, load_train_stats
    if not STATS_PATH.exists():
        pytest.skip("artifacts/train_stats.json not available")
    return load_train_stats()


def assert_prefix_causal(df, stats, ks):
    full = compute_features(df, stats)
    for k in ks:
        pd.testing.assert_frame_equal(compute_features(df.iloc[:k], stats), full.iloc[:k])


def test_prefix_causality_synthetic():
    assert_prefix_causal(synthetic_stream(), STATS, ks=[1, 2, 17, 999, 2999])


@needs_real
def test_prefix_causality_real(real_slice, real_stats):
    assert_prefix_causal(real_slice, real_stats, ks=[1, 500, 10_123, 29_999])


@pytest.mark.parametrize("k", [10, 1500])
def test_randomizing_future_leaves_past_unchanged(k):
    df = synthetic_stream(seed=1)
    other = synthetic_stream(seed=99)
    fut = df.copy()
    for c in ["arb_id", "dlc"] + BYTE_COLS:
        fut.loc[k:, c] = other.loc[k:, c].to_numpy()
    fut.loc[k:, "timestamp"] = df["timestamp"].iloc[k] + np.sort(np.random.default_rng(5).random(len(df) - k))
    a, b = compute_features(df, STATS), compute_features(fut, STATS)
    pd.testing.assert_frame_equal(a.iloc[:k], b.iloc[:k])
    assert not a.iloc[k:].equals(b.iloc[k:])


def test_first_frame_sentinels():
    f = compute_features(synthetic_stream(), STATS)
    first = ~f.duplicated("arb_id")
    assert (f.loc[first, ["dt_same_id", "dt_ratio", "payload_changed", "n_bytes_changed"]] == -1).all().all()
    assert (f.loc[first, "is_first_of_id"] == 1).all()
    assert (f.loc[~first, "is_first_of_id"] == 0).all()
    assert (f.loc[~first, ["dt_same_id", "dt_ratio", "payload_changed", "n_bytes_changed"]] >= 0).all().all()
    assert f["dt_prev_frame"].iloc[0] == -1 and (f["dt_prev_frame"].iloc[1:] >= 0).all()
    assert f[["id_count_last_100ms", "frame_count_last_50ms"]].iloc[0].tolist() == [0, 0]


def test_unknown_id_does_not_leak_into_behavioral():
    known = synthetic_stream(ids=(KNOWN_A, KNOWN_B))
    unknown = known.copy()
    unknown.loc[unknown["arb_id"] == KNOWN_A, "arb_id"] = UNKNOWN_Z
    fk, fu = compute_features(known, STATS), compute_features(unknown, STATS)
    pd.testing.assert_frame_equal(fk[BEHAVIORAL], fu[BEHAVIORAL])
    swapped = known["arb_id"] == KNOWN_A
    assert (fk.loc[swapped, "id_seen_in_training"] == 1).all()
    assert (fu.loc[swapped, "id_seen_in_training"] == 0).all()
    assert not fu[ALL_FEATURES].isna().any().any()


@needs_real
def test_cross_check_against_pandas(real_slice, real_stats):
    df = real_slice.sort_values(["timestamp", "row_idx"], kind="stable").reset_index(drop=True)
    f = compute_features(df, real_stats)
    ts = df["timestamp"].to_numpy()

    dt = df.groupby("arb_id")["timestamp"].diff()
    np.testing.assert_array_equal(f["dt_same_id"], dt.fillna(-1).astype(np.float32))
    ref = df["arb_id"].map(lambda i: real_stats["per_id"].get(i, {}).get(
        "median_period_s", real_stats["global_ref_period_s"]))
    np.testing.assert_array_equal(f["dt_ratio"], (dt / ref).fillna(-1).astype(np.float32))

    idx = np.arange(len(df))
    for w in BUS_WINDOWS_S:
        expected = idx - np.searchsorted(ts, ts - w, side="right")
        np.testing.assert_array_equal(f[f"frame_count_last_{round(w * 1000)}ms"], expected)

    expected_id = np.empty(len(df), dtype=np.int64)
    for _, pos in df.groupby("arb_id").indices.items():
        t = ts[pos]
        expected_id[pos] = np.arange(len(t)) - np.searchsorted(t, t - ID_WINDOW_S, side="right")
    np.testing.assert_array_equal(f["id_count_last_100ms"], expected_id)

    age = df.groupby(["arb_id", "dlc"] + BYTE_COLS)["timestamp"].diff()
    np.testing.assert_array_equal(f["payload_repeat_age"], age.fillna(-1).astype(np.float32))
    np.testing.assert_array_equal(f["payload_seen_before"], age.notna().astype(np.int8))


def test_feature_lists_exclude_time_position_and_labels():
    forbidden = {"timestamp", "row_idx", "ts_relative", "class", "subclass", "is_attack", "source_file"}
    assert not forbidden & set(BEHAVIORAL)
    assert not forbidden & set(ID_INCLUSIVE)
    assert not {"arb_id", "id_seen_in_training"} & set(BEHAVIORAL)
    assert set(META_COLS).isdisjoint(ALL_FEATURES)


@pytest.mark.parametrize("max_age, expected", [(None, 2.0), (1.0, -1.0), (5.0, 2.0)])
def test_payload_history_max_age(max_age, expected):
    ext = CausalFeatureExtractor(STATS, mode="D", payload_history_max_age_s=max_age)
    payload = (1, 2, 3, 4, 5, 6, 7, 8)
    ext.step(100.0, KNOWN_A, 8, payload)
    ext.step(101.0, KNOWN_B, 8, payload)  # same bytes, different ID: a different key
    row = dict(zip(ALL_FEATURES, ext.step(102.0, KNOWN_A, 8, payload)))
    assert row["payload_repeat_age"] == expected
    assert row["payload_seen_before"] == int(expected >= 0)
    assert row["mode"] == 1


@pytest.mark.parametrize("max_age", [0.003, 0.03])
def test_bounded_history_equals_unbounded_age_capped(max_age):
    df = synthetic_stream(seed=3)
    full = compute_features(df, STATS)
    bounded = compute_features(df, STATS, payload_history_max_age_s=max_age)
    age = full["payload_repeat_age"]
    expected = age.where(age <= np.float32(max_age), np.float32(-1))
    np.testing.assert_array_equal(bounded["payload_repeat_age"], expected)
    assert (expected == -1).sum() > (age == -1).sum()  # the bound actually bit
    others = [c for c in ALL_FEATURES if c not in ("payload_repeat_age", "payload_seen_before")]
    pd.testing.assert_frame_equal(bounded[others], full[others])


def test_reset_starts_a_new_stream():
    ext = CausalFeatureExtractor(STATS, mode="S")
    first = ext.step(1.0, KNOWN_A, 2, (1, 2, -1, -1, -1, -1, -1, -1))
    ext.step(1.01, KNOWN_A, 2, (1, 3, -1, -1, -1, -1, -1, -1))
    ext.reset()
    assert ext.step(1.0, KNOWN_A, 2, (1, 2, -1, -1, -1, -1, -1, -1)) == first
