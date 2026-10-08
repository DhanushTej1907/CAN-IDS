import numpy as np
import pytest

from src.evaluate import (ALL_ATTACKS, auc_metrics, evaluate_scores, macro_recall, md_table,
                          metrics_at_threshold, quantile_of, threshold_at_fpr, threshold_at_quantile)


@pytest.mark.parametrize("target", [0.0, 0.001, 0.005, 0.01, 0.1])
@pytest.mark.parametrize("ties", [False, True])
def test_threshold_at_fpr_meets_target_and_is_smallest(target, ties):
    s = np.random.default_rng(0).normal(size=20_000)
    if ties:
        s = np.round(s, 1)  # heavy ties, like dt_ratio
    t = threshold_at_fpr(s, target)
    assert np.mean(s > t) <= target
    below = s[s < t]
    if len(below):  # the next lower candidate would break the target
        assert np.mean(s > below.max()) > target


def test_q_transfer_round_trip():
    ref = np.round(np.random.default_rng(1).normal(size=10_000), 2)
    for t in ref[:200]:
        assert threshold_at_quantile(ref, quantile_of(ref, t)) == t
    t = 0.12345  # between ref values: same flags on ref
    np.testing.assert_array_equal(ref > t, ref > threshold_at_quantile(ref, quantile_of(ref, t)))


def test_metrics_at_threshold_hand_example():
    sub = np.array(["Normal"] * 10 + ["Flooding"] * 4 + ["Spoofing"] * 2)
    scores = np.array([0] * 9 + [5] + [5, 5, 5, 0] + [5, 0], dtype=float)
    m = metrics_at_threshold(scores, sub, 1.0).set_index("subclass")
    assert (m.loc["Flooding", "n"], m.loc["Flooding", "tp"], m.loc["Flooding", "fp"]) == (4, 3, 1)
    assert m.loc["Flooding", "recall"] == 0.75 and m.loc["Flooding", "precision"] == 0.75
    assert m.loc["Spoofing", "recall"] == 0.5 and m.loc["Spoofing", "precision"] == 0.5
    assert (m["fpr"] == 0.1).all()
    assert np.isnan(m.loc["Replay", "recall"])
    assert m.loc[ALL_ATTACKS, "tp"] == 4


def test_auc_and_macro_recall():
    sub = np.array(["Normal"] * 6 + ["Flooding", "Spoofing", "Replay", "Fuzzing"])
    scores = np.array([0, 1, 2, 3, 4, 5, 9, 9, 9, -1], dtype=float)
    a = auc_metrics(scores, sub).set_index("subclass")
    assert a.loc["Flooding", "roc_auc"] == 1.0 and a.loc["Fuzzing", "roc_auc"] == 0.0
    table = evaluate_scores(scores, sub, {0.001: threshold_at_fpr(scores[:6], 0.001)})
    assert macro_recall(table, 0.001) == 0.75  # Fuzzing missed; "All attacks" row not averaged in


def test_md_table():
    import pandas as pd
    out = md_table(pd.DataFrame({"a": [1234], "b": [0.5], "c": [np.nan]}))
    assert out.splitlines() == ["| a | b | c |", "|---|---|---|", "| 1,234 | 0.500 | - |"]
