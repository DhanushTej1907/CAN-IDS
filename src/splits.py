"""File-level splits -> splits/splits.json (read by training code, incl. Colab).

Roles come from the subset tag in each processed parquet footer (written by src.load), not from
hand-typed lists, so a renamed or missing file fails loudly. Splitting is by FILE to avoid
temporal leakage between neighbouring frames.
Usage: python -m src.splits
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.load import PROCESSED_DIR, ROOT, SUBSET_CATS, processed_subset

SPLITS_PATH = ROOT / "splits" / "splits.json"

# Pre_train_<mode>_<day>: day 0 = Normal-only capture, days 1..n = attack captures.
TRAIN_NAME_RE = re.compile(r"^Pre_train_(?P<mode>[SD])_(?P<day>\d+)$")
NORMAL_ONLY_DAY = "0"

# Expected file counts per role; any drift (missing/renamed/extra file) is an error.
EXPECTED_COUNTS = {"train": 6, "train_always": 2, "attack": 4, "test": 2, "extra_test": 1}

NOTES = [
    "Split by FILE, never by row, to avoid temporal leakage.",
    "train_always (Pre_train_*_0, Normal-only) is in the training set of every CV fold.",
    "cv_default is 'lodo' (leave-one-day-out): Pre_train_{D,S}_1 and Pre_train_{D,S}_2 are "
    "same-day S/D twins that share the attack script, so leave-one-file-out (lofo) lets the "
    "twin of the validation file sit in training and can be optimistic. lofo is kept as a "
    "secondary scheme.",
    "stats_source: per-ID timing stats (src.train_stats) come from these files only. They are "
    "in training for every fold, so the stats cannot leak any validation fold.",
    "test (Pre_submit_*) is held out: no EDA, tuning or threshold selection on it.",
    "extra_test (Fin_*) is an optional extra held-out test: session-relative timestamps "
    "(not UNIX) and S mode only.",
]


def files_by_subset(processed_dir: Path = PROCESSED_DIR) -> dict[str, list[str]]:
    """Processed file stems grouped by the subset tag in their parquet footer."""
    groups: dict[str, list[str]] = {s: [] for s in SUBSET_CATS}
    for f in sorted(Path(processed_dir).glob("*.parquet")):
        groups[processed_subset(f)].append(f.stem)
    return groups


def build_splits(processed_dir: Path = PROCESSED_DIR) -> dict:
    groups = files_by_subset(processed_dir)
    train = groups["prelim_train"]

    parsed = {}
    for name in train:
        m = TRAIN_NAME_RE.match(name)
        if not m:
            raise ValueError(f"{name}: prelim_train file does not match Pre_train_<S|D>_<day>")
        parsed[name] = (m["mode"], m["day"])
    train_always = [n for n in train if parsed[n][1] == NORMAL_ONLY_DAY]
    attack = [n for n in train if parsed[n][1] != NORMAL_ONLY_DAY]

    # Normal-only is checked against the labels, not assumed from the filename.
    for name in train_always:
        is_attack = pd.read_parquet(Path(processed_dir) / f"{name}.parquet", columns=["is_attack"])
        if is_attack["is_attack"].any():
            raise ValueError(f"{name}: expected Normal-only but has Attack rows")

    lodo = []
    for day in sorted({parsed[n][1] for n in attack}):
        val = sorted(n for n in attack if parsed[n][1] == day)
        if sorted(parsed[n][0] for n in val) != ["D", "S"]:
            raise ValueError(f"day {day}: expected exactly one D and one S file, got {val}")
        lodo.append({"name": f"lodo_day{day}", "train": sorted(set(train) - set(val)), "val": val})
    lofo = [{"name": f"lofo_{n}", "train": sorted(set(train) - {n}), "val": [n]} for n in attack]

    splits = {
        "train": train,
        "train_always": train_always,
        "cv_default": "lodo",
        "cv_schemes": {"lodo": lodo, "lofo": lofo},
        "test": groups["prelim_submit"],
        "extra_test": groups["final"],
        "stats_source": train_always,
        "notes": NOTES,
    }
    validate_splits(splits, all_files=[n for g in groups.values() for n in g], attack=attack)
    return splits


def validate_splits(splits: dict, all_files: list[str], attack: list[str]) -> None:
    counts = {"train": len(splits["train"]), "train_always": len(splits["train_always"]),
              "attack": len(attack), "test": len(splits["test"]),
              "extra_test": len(splits["extra_test"])}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"file counts {counts} != expected {EXPECTED_COUNTS}")

    roles = splits["train"] + splits["test"] + splits["extra_test"]
    if sorted(roles) != sorted(all_files) or len(set(roles)) != len(roles):
        raise ValueError("every processed file must land in exactly one of train/test/extra_test")

    train, always = set(splits["train"]), set(splits["train_always"])
    for scheme, folds in splits["cv_schemes"].items():
        vals = [n for fold in folds for n in fold["val"]]
        if sorted(vals) != sorted(attack):
            raise ValueError(f"{scheme}: val files across folds must cover each attack file once")
        for fold in folds:
            ft, fv = set(fold["train"]), set(fold["val"])
            if ft & fv or ft | fv != train:
                raise ValueError(f"{scheme}/{fold['name']}: train/val must partition the train files")
            if not always <= ft:
                raise ValueError(f"{scheme}/{fold['name']}: train_always must be in fold train")
    if splits["cv_default"] not in splits["cv_schemes"]:
        raise ValueError(f"cv_default {splits['cv_default']!r} is not a defined scheme")


def write_splits(splits: dict, path: Path = SPLITS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    s = build_splits()
    out = write_splits(s)
    print(f"train       : {s['train']}")
    print(f"train_always: {s['train_always']}")
    for scheme, folds in s["cv_schemes"].items():
        tag = " (default)" if scheme == s["cv_default"] else ""
        for fold in folds:
            print(f"{scheme}{tag:<10} {fold['name']:<22} val={fold['val']}")
    print(f"test        : {s['test']}")
    print(f"extra_test  : {s['extra_test']}")
    print(f"-> {out}")
