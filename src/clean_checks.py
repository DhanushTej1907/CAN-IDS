"""Cleaning checks on data/processed/*.parquet -> reports/cleaning_report.{md,json}.

Report-only: nothing is dropped or modified. Integrity checks (not distribution analysis),
so they run on every subset, including the held-out test files.
Duplicates among Attack rows are expected (Replay re-sends captured frames) and are never
treated as something to remove.
Usage: python -m src.clean_checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.load import (BYTE_COLS, CLASS_CATS, PROCESSED_DIR, ROOT, SUBCLASS_CATS,
                      processed_subset)

REPORTS_DIR = ROOT / "reports"
N_EXAMPLES = 3
EXAMPLE_COLS = ["row_idx", "timestamp", "arb_id", "dlc", "n_bytes"] + BYTE_COLS + ["class", "subclass"]


def _examples(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> list[dict]:
    ex = df.loc[mask, EXAMPLE_COLS].head(N_EXAMPLES).copy()
    ex["arb_id"] = ex["arb_id"].map(lambda v: f"0x{v:03X}")
    return json.loads(ex.to_json(orient="records", double_precision=15))


def _by_subclass(df: pd.DataFrame, mask) -> dict:
    return {k: int(v) for k, v in df.loc[mask, "subclass"].value_counts().items() if v}


def _check(name, count, status=None, note="", breakdown=None, examples=None) -> dict:
    return {"check": name, "count": int(count), "status": status or ("OK" if count == 0 else "FLAG"),
            "note": note, "breakdown": breakdown or {}, "examples": examples or []}


def _duplicate_check(df, name, cols, note) -> dict:
    """Rows that repeat an earlier row on `cols`. Attack-only duplicates -> INFO (expected)."""
    dup = df.duplicated(subset=cols, keep="first")
    n = int(dup.sum())
    n_normal = int((dup & (df["class"] == "Normal")).sum())
    if n == 0:
        status = "OK"
    elif n_normal == 0:
        status = "INFO"
        note += " All duplicates are Attack rows: expected attack traffic (e.g., Replay), keep as is."
    else:
        status = "FLAG"
        note += f" {n_normal:,} Normal-row duplicates: review only, no action proposed."
    return _check(name, n, status, note.strip(), _by_subclass(df, dup), _examples(df, dup))


def check_file(path: Path) -> dict:
    df = pd.read_parquet(path).sort_values("row_idx", kind="stable").reset_index(drop=True)
    checks = []

    nulls = df.isna().sum()
    nulls = {c: int(v) for c, v in nulls.items() if v}
    checks.append(_check("nulls", sum(nulls.values()), breakdown=nulls))

    content_cols = ["timestamp", "arb_id", "dlc"] + BYTE_COLS + ["class", "subclass"]
    checks.append(_duplicate_check(df, "exact_duplicate_rows", content_cols,
                                   "Same timestamp, ID, DLC, payload and labels as an earlier row."))
    checks.append(_duplicate_check(df, "duplicate_timestamp_id", ["timestamp", "arb_id"],
                                   "Same (timestamp, arb_id) as an earlier row; payload may differ."))

    mism = df["dlc"] != df["n_bytes"]
    checks.append(_check("dlc_vs_byte_count_mismatch", mism.sum(),
                         breakdown=_by_subclass(df, mism), examples=_examples(df, mism)))

    # -1 padding must occur exactly at positions >= n_bytes, and real bytes must be 0..255.
    pos = np.arange(len(BYTE_COLS))
    b = df[BYTE_COLS].to_numpy()
    should_pad = pos[None, :] >= df["n_bytes"].to_numpy()[:, None]
    bad_pad = ((b == -1) != should_pad).any(axis=1)
    out_rng = ((b < -1) | (b > 255)).any(axis=1)
    checks.append(_check("byte_padding_or_range", (bad_pad | out_rng).sum(),
                         examples=_examples(df, bad_pad | out_rng)))

    bad_id = (df["arb_id"] < 0) | (df["arb_id"] > 0x7FF)
    checks.append(_check("arb_id_outside_11bit", bad_id.sum(), examples=_examples(df, bad_id)))

    ts = df["timestamp"].to_numpy()
    d = np.diff(ts)
    dec = np.concatenate([[False], d < 0])
    ties = int((d == 0).sum())
    relative = bool(df["ts_relative"].all())
    checks.append(_check("timestamp_decreasing", dec.sum(), examples=_examples(df, dec),
                         note="Steps where timestamp goes backwards in file order."))
    checks.append(_check("timestamp_ties", ties, status="OK" if ties == 0 else "INFO",
                         note="Consecutive frames with identical timestamps (row_idx keeps order)."))
    if df["ts_relative"].nunique() > 1:
        checks.append(_check("timestamp_mixed_relative_unix", int(df["ts_relative"].sum())))

    unk_cls = ~df["class"].isin(CLASS_CATS)
    unk_sub = ~df["subclass"].isin(SUBCLASS_CATS)
    incons = ((df["class"] == "Normal") & (df["subclass"] != "Normal")) | \
             ((df["class"] == "Attack") & (df["subclass"] == "Normal"))
    checks.append(_check("unknown_class", unk_cls.sum(), examples=_examples(df, unk_cls)))
    checks.append(_check("unknown_subclass", unk_sub.sum(), examples=_examples(df, unk_sub)))
    checks.append(_check("class_subclass_inconsistent", incons.sum(), examples=_examples(df, incons)))

    return {
        "source_file": path.stem,
        "subset": processed_subset(path),
        "mode": str(df["mode"].iloc[0]),
        "rows": len(df),
        "timestamp_min": float(ts.min()),
        "timestamp_max": float(ts.max()),
        "duration_s": float(ts.max() - ts.min()),
        "timestamps_relative": relative,
        "checks": checks,
    }


def to_markdown(results: list[dict]) -> str:
    names = [c["check"] for c in results[0]["checks"]]
    lines = ["# Cleaning report", "",
             "Generated by `python -m src.clean_checks` from `data/processed/*.parquet`. "
             "**Report only: no rows were dropped or modified.**", "",
             "Status: OK = none found; INFO = present but expected or harmless; FLAG = worth a look "
             "(no action is proposed here). Duplicates among Attack rows are expected "
             "(Replay re-sends captured traffic) and are never candidates for removal.", "",
             "## Summary", "",
             "| file | subset | mode | rows | duration (s) | ts relative | " + " | ".join(names) + " |",
             "|" + "---|" * (6 + len(names))]
    for r in results:
        cells = [f"{c['status']} ({c['count']:,})" if c["count"] else c["status"] for c in r["checks"]]
        lines.append(f"| {r['source_file']} | {r['subset']} | {r['mode']} | {r['rows']:,} | "
                     f"{r['duration_s']:.1f} | {r['timestamps_relative']} | " + " | ".join(cells) + " |")
    lines += ["", "## Details (non-OK checks)", ""]
    any_detail = False
    for r in results:
        bad = [c for c in r["checks"] if c["status"] != "OK"]
        if not bad:
            continue
        any_detail = True
        lines.append(f"### {r['source_file']}")
        for c in bad:
            lines.append(f"- **{c['check']}**: {c['status']}, {c['count']:,}. {c['note']}")
            if c["breakdown"]:
                lines.append(f"  - by subclass/column: {c['breakdown']}")
            for e in c["examples"]:
                lines.append(f"  - e.g. `{e}`")
        lines.append("")
    if not any_detail:
        lines.append("All checks OK for all files.")
    return "\n".join(lines) + "\n"


def run_all(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    results = []
    for p in sorted(Path(processed_dir).glob("*.parquet")):
        r = check_file(p)
        status = ", ".join(f"{c['check']}={c['status']}" for c in r["checks"] if c["status"] != "OK") or "all OK"
        print(f"{r['source_file']:<28} {status}")
        results.append(r)
    return results


if __name__ == "__main__":
    results = run_all()
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "cleaning_report.json").write_text(json.dumps(results, indent=2))
    (REPORTS_DIR / "cleaning_report.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"\nwrote {REPORTS_DIR / 'cleaning_report.md'}")
