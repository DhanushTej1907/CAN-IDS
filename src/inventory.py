"""Inventory of data/raw/: file sizes, row counts, columns, S/D mode and subset.

Integrity-level only (no label distributions), so it is safe to run on test files too.
Usage: python -m src.inventory
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.load import RAW_DIR, ROOT, count_csv_rows, list_raw_files, parse_file_meta, read_header

REPORTS_DIR = ROOT / "reports"


def inventory_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for p in list_raw_files(raw_dir):
        meta = parse_file_meta(p)
        rows.append({
            "source_file": meta["source_file"],
            "path": p.relative_to(raw_dir).as_posix(),
            "subset": meta["subset"],
            "mode": meta["mode"],
            "size_MB": round(p.stat().st_size / 2**20, 1),
            "rows": count_csv_rows(p),
            "columns": "|".join(read_header(p)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    inv = inventory_raw()
    REPORTS_DIR.mkdir(exist_ok=True)
    inv.to_csv(REPORTS_DIR / "raw_inventory.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(inv.drop(columns="path").to_string(index=False))
    print(f"\nTotal: {inv['rows'].sum():,} rows, {inv['size_MB'].sum():.0f} MB")
    print(inv.groupby(["subset", "mode"])["rows"].sum().to_string())
