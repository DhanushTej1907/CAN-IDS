import numpy as np
import pandas as pd
import pytest

from src.features import ALL_FEATURES, BEHAVIORAL, FEATURES_DIR, ID_INCLUSIVE, META_COLS
from src.train import (DtRatioRule, IForestModel, XGBModel, anomaly_train_mask, load_feature_files,
                       near_attack_mask, reference_normal_mask, supervised_train_mask)
from src.train_stats import load_splits

SPLITS = load_splits()


def toy_meta(seed=0):
    """Two attack files + one Normal-only file over the SAME time range (so per-file grouping matters)."""
    rng = np.random.default_rng(seed)
    parts = []
    for stem, has_attack in [("A", True), ("B", True), ("N", False)]:
        n = 4000
        ts = np.sort(rng.uniform(0, 60, n))
        att = np.zeros(n, np.int8)
        if has_attack:
            for start in rng.uniform(5, 55, 3):
                att[(ts >= start) & (ts < start + 0.5)] = 1
        sub = np.where(att == 1, rng.choice(["Flooding", "Spoofing"], n), "Normal")
        parts.append(pd.DataFrame({"source_file": stem, "timestamp": ts, "subclass": sub, "is_attack": att}))
    meta = pd.concat(parts, ignore_index=True)
    meta["source_file"] = pd.Categorical(meta["source_file"], categories=["A", "B", "N"])
    return meta


def test_near_attack_mask_matches_brute_force():
    meta = toy_meta()
    got = near_attack_mask(meta, 1.0)
    expected = np.zeros(len(meta), dtype=bool)
    for _, g in meta.groupby("source_file", observed=True):
        att_t = g.loc[g["is_attack"] == 1, "timestamp"].to_numpy()
        if len(att_t):
            d = np.abs(g["timestamp"].to_numpy()[:, None] - att_t[None, :]).min(axis=1)
            expected[g.index] = d <= 1.0
    np.testing.assert_array_equal(got, expected)
    assert not got[(meta["source_file"] == "N").to_numpy()].any()


def test_supervised_mask_keeps_all_attacks_at_4_to_1():
    meta = toy_meta()
    keep = supervised_train_mask(meta, ratio=4)
    att = meta["is_attack"].to_numpy() == 1
    assert keep[att].all()
    for _, g in meta.groupby("source_file", observed=True):
        k, a = keep[g.index], att[g.index]
        if a.any():
            assert (k & ~a).sum() == min((~a).sum(), 4 * a.sum())
        else:
            assert k.all()


def test_supervised_subsample_is_identical_in_every_fold():
    meta = toy_meta()
    full = supervised_train_mask(meta)
    in_fold = meta["source_file"].isin(["A", "N"]).to_numpy()
    part = supervised_train_mask(meta[in_fold].reset_index(drop=True))
    np.testing.assert_array_equal(part, full[in_fold])


def test_reference_is_every_normal_frame_regardless_of_sampling():
    meta = toy_meta()
    R = reference_normal_mask(meta)
    np.testing.assert_array_equal(R, meta["subclass"].to_numpy() == "Normal")
    assert (anomaly_train_mask(meta) <= R).all()
    assert R[~supervised_train_mask(meta)].any()  # includes Normals the 4:1 subsample left out
    assert R[near_attack_mask(meta)].any()        # and near-attack Normals that IF excluded


def test_loader_rejects_held_out_and_unknown_files():
    for stem in SPLITS["test"] + SPLITS["extra_test"]:
        with pytest.raises(PermissionError):
            load_feature_files([stem])
    with pytest.raises(ValueError):
        load_feature_files(["not_a_file"])


@pytest.mark.skipif(not (FEATURES_DIR / f"{SPLITS['train_always'][0]}.parquet").exists(),
                    reason="feature files not available")
def test_loader_returns_features_without_meta():
    X, meta = load_feature_files([SPLITS["train_always"][0]])
    assert X.dtype == np.float32 and X.shape == (len(meta), len(ALL_FEATURES))
    assert set(META_COLS).isdisjoint(ALL_FEATURES)
    assert list(meta.columns) == ["source_file", "timestamp", "subclass", "is_attack"]


def test_dt_ratio_rule():
    X = np.zeros((4, len(ID_INCLUSIVE)), np.float32)
    X[:, ID_INCLUSIVE.index("dt_ratio")] = [1.0, 0.2, -1.0, 1.0]  # on time, early, first-of-ID, on time
    X[:, ID_INCLUSIVE.index("id_seen_in_training")] = [1, 1, 1, 0]
    s = DtRatioRule(ID_INCLUSIVE).score(X)
    assert s[3] > s[1] > s[0] > s[2]
    Xb = X[:, [ID_INCLUSIVE.index(f) for f in BEHAVIORAL]]
    sb = DtRatioRule(BEHAVIORAL).score(Xb)
    assert sb[1] > sb[0] == sb[3] > sb[2]  # unknown ID is invisible to the behavioral rule
    with pytest.raises(RuntimeError):      # attack on time, Normal early -> direction flipped
        DtRatioRule(BEHAVIORAL).fit(Xb[:2], np.array([1, 0]))


@pytest.mark.parametrize("cls, params", [(IForestModel, {"n_estimators": 20}),
                                         (XGBModel, {"n_estimators": 20, "max_depth": 3, "w_spoof": 5})])
def test_models_rank_attacks_higher_and_score_one_matches(cls, params):
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(np.int8)
    X = (rng.normal(size=(2000, len(BEHAVIORAL))) + 4 * y[:, None]).astype(np.float32)
    m = cls(BEHAVIORAL, **params).fit(X, y, np.where(y == 1, "Spoofing", "Normal"))
    s = m.score(X)
    assert s[y == 1].mean() > s[y == 0].mean()
    np.testing.assert_allclose(m.score_one(X[:1]), s[:1], rtol=1e-6)
