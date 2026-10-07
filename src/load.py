"""Chunked loader: raw HCRL CAN CSVs -> typed, cleaned parquet (one file per source CSV).

No rows are dropped or modified here; anything unparseable raises with file + row context.
Usage: python -m src.load
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CHUNKSIZE = 500_000
N_BYTES = 8
BYTE_COLS = [f"byte_{i}" for i in range(N_BYTES)]

# Fixed category lists so every chunk/file shares one schema and unknown labels can't slip in.
CLASS_CATS = ["Normal", "Attack"]
SUBCLASS_CATS = ["Normal", "Flooding", "Spoofing", "Replay", "Fuzzing"]
MODE_CATS = ["S", "D"]
SUBSET_CATS = ["prelim_train", "prelim_submit", "final"]

# Raw folder name -> subset. Subset drives which files EDA is allowed to touch (prelim_train only).
SUBSET_BY_FOLDER = {"0_Training": "prelim_train", "1_Submission": "prelim_submit", "1_Final": "final"}

REQUIRED_RAW_COLS = ["Timestamp", "Arbitration_ID", "DLC", "Data", "Class"]
OPTIONAL_RAW_COLS = ["SubClass"]  # absent in Pre_train_*_0 (Normal-only files)

# UNIX seconds below this (~2001) means the file uses session-relative time (the Final round file).
UNIX_TS_MIN = 1e9

OUT_COLS = (["source_file", "subset", "mode", "row_idx", "timestamp", "ts_relative",
             "arb_id", "dlc", "n_bytes"] + BYTE_COLS + ["class", "subclass", "is_attack"])

PARQUET_META_SUBSET = b"can_ids.subset"


def parse_file_meta(path: Path | str) -> dict:
    """Derive mode (S/D), subset and source_file from a raw CSV path."""
    path = Path(path)
    m = re.search(r"_([SD])(?:_\d+)?$", path.stem)
    if not m:
        raise ValueError(f"{path.name}: cannot infer S/D mode from filename")
    subset = SUBSET_BY_FOLDER.get(path.parent.name)
    if subset is None:
        raise ValueError(f"{path.name}: unknown parent folder {path.parent.name!r} "
                         f"(expected one of {list(SUBSET_BY_FOLDER)})")
    return {"mode": m.group(1), "subset": subset, "source_file": path.stem}


def list_raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(Path(raw_dir).rglob("*.csv"))


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip().split(",")


def count_csv_rows(path: Path, block: int = 1 << 24) -> int:
    """Data rows by counting newlines (independent of pandas, so it catches silently skipped lines)."""
    n, last = 0, b""
    with open(path, "rb") as f:
        while chunk := f.read(block):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        n += 1  # final line without trailing newline
    return n - 1  # minus header


def load_raw_file(path: Path, chunksize: int = CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Yield raw chunks with safe dtypes: IDs/payload as str, full-precision float timestamps."""
    header = read_header(path)
    missing = [c for c in REQUIRED_RAW_COLS if c not in header]
    extra = [c for c in header if c not in REQUIRED_RAW_COLS + OPTIONAL_RAW_COLS]
    if missing or extra:
        raise ValueError(f"{path.name}: unexpected columns (missing={missing}, extra={extra})")
    dtypes = {"Timestamp": "float64", "Arbitration_ID": str, "DLC": "int16", "Data": str,
              "Class": str, "SubClass": str}
    yield from pd.read_csv(
        path,
        dtype={c: t for c, t in dtypes.items() if c in header},
        keep_default_na=False,         # DLC=0 frames have empty Data; keep "" rather than NaN
        float_precision="round_trip",  # exact float parse of the timestamp string
        chunksize=chunksize,
    )


def _parse_hex(values: pd.Series, max_value: int, what: str, ctx: str, row_offset: int) -> pd.Series:
    """Hex strings -> ints via a lookup over unique values (few uniques, many rows). NaN stays NaN."""
    lut = {}
    for u in pd.unique(values.dropna()):
        try:
            v = int(u, 16)
            if not 0 <= v <= max_value:
                raise ValueError
        except ValueError:
            pos = int(np.flatnonzero((values == u).to_numpy())[0])
            raise ValueError(f"{ctx} row {row_offset + pos}: bad {what} {u!r}") from None
        lut[u] = v
    return values.map(lut)


def _fail_first(mask: pd.Series, msg: str, ctx: str, row_offset: int) -> None:
    if mask.any():
        pos = int(np.flatnonzero(mask.to_numpy())[0])
        raise ValueError(f"{ctx} row {row_offset + pos}: {msg}")


def clean_chunk(df: pd.DataFrame, meta: dict, row_offset: int) -> pd.DataFrame:
    """Raw chunk -> typed frame. Validates strictly; never drops rows."""
    ctx = meta["source_file"]
    n = len(df)
    out = pd.DataFrame(index=pd.RangeIndex(n))

    out["source_file"] = pd.Categorical([ctx] * n, categories=[ctx])
    out["subset"] = pd.Categorical([meta["subset"]] * n, categories=SUBSET_CATS)
    out["mode"] = pd.Categorical([meta["mode"]] * n, categories=MODE_CATS)
    out["row_idx"] = np.arange(row_offset, row_offset + n, dtype=np.int32)

    ts = df["Timestamp"].to_numpy(dtype=np.float64)
    _fail_first(pd.Series(~np.isfinite(ts)), "non-finite timestamp", ctx, row_offset)
    out["timestamp"] = ts
    out["ts_relative"] = ts < UNIX_TS_MIN

    ids = df["Arbitration_ID"].reset_index(drop=True).str.strip()
    out["arb_id"] = _parse_hex(ids, 0x7FF, "Arbitration_ID", ctx, row_offset).astype(np.int16)

    dlc = df["DLC"].reset_index(drop=True)
    _fail_first((dlc < 0) | (dlc > N_BYTES), "DLC outside 0..8", ctx, row_offset)
    out["dlc"] = dlc.astype(np.int8)

    # Payload: whitespace split -> up to 8 tokens; missing positions padded with -1.
    parts = df["Data"].reset_index(drop=True).str.split(expand=True)
    if parts.shape[1] > N_BYTES:
        _fail_first(parts[N_BYTES].notna(), "more than 8 payload bytes", ctx, row_offset)
    parts = parts.reindex(columns=range(N_BYTES))
    out["n_bytes"] = parts.notna().sum(axis=1).astype(np.int8)
    for i, col in enumerate(BYTE_COLS):
        out[col] = _parse_hex(parts[i], 0xFF, f"payload byte {i}", ctx, row_offset).fillna(-1).astype(np.int16)

    cls = df["Class"].reset_index(drop=True)
    _fail_first(~cls.isin(CLASS_CATS), "unknown Class", ctx, row_offset)
    if "SubClass" in df.columns:
        sub = df["SubClass"].reset_index(drop=True)
    else:
        # Pre_train_*_0 have no SubClass column; they are Normal-only, verified here, not assumed.
        _fail_first(cls != "Normal", "Attack row in a file without SubClass", ctx, row_offset)
        sub = pd.Series(["Normal"] * n)
    _fail_first(~sub.isin(SUBCLASS_CATS), "unknown SubClass", ctx, row_offset)
    out["class"] = pd.Categorical(cls, categories=CLASS_CATS)
    out["subclass"] = pd.Categorical(sub, categories=SUBCLASS_CATS)
    out["is_attack"] = (cls == "Attack").to_numpy().astype(np.int8)

    return out[OUT_COLS]


def process_file(path: Path, out_dir: Path = PROCESSED_DIR, chunksize: int = CHUNKSIZE) -> dict:
    """Stream one raw CSV into data/processed/<stem>.parquet (one row group per chunk)."""
    path, out_dir = Path(path), Path(out_dir)
    if RAW_DIR.resolve() in out_dir.resolve().parents or out_dir.resolve() == RAW_DIR.resolve():
        raise ValueError("refusing to write inside data/raw/")
    meta = parse_file_meta(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{meta['source_file']}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")

    writer, schema, n_rows, n_attack = None, None, 0, 0
    try:
        for raw in load_raw_file(path, chunksize):
            df = clean_chunk(raw, meta, row_offset=n_rows)
            if writer is None:
                schema = pa.Schema.from_pandas(df, preserve_index=False)
                schema = schema.with_metadata({**schema.metadata,
                                               PARQUET_META_SUBSET: meta["subset"].encode()})
                writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
            writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
            n_rows += len(df)
            n_attack += int(df["is_attack"].sum())
    finally:
        if writer is not None:
            writer.close()

    expected = count_csv_rows(path)
    if n_rows != expected or pq.ParquetFile(tmp_path).metadata.num_rows != expected:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: row count mismatch (parsed {n_rows}, raw lines {expected})")
    tmp_path.replace(out_path)
    return {**meta, "rows": n_rows, "attack_rows": n_attack, "out": str(out_path)}


def process_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
                chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    results = []
    for p in list_raw_files(raw_dir):
        r = process_file(p, out_dir, chunksize)
        print(f"{r['source_file']:<28} {r['subset']:<14} {r['mode']}  rows={r['rows']:>9,}  "
              f"attack={r['attack_rows']:>7,}")
        results.append(r)
    return pd.DataFrame(results)


def processed_subset(path: Path) -> str:
    """Read a parquet file's subset from its footer metadata (cheap; no data read)."""
    return pq.read_schema(path).metadata[PARQUET_META_SUBSET].decode()


def load_processed(names: Iterable[str] | None = None,
                   subset: str | Iterable[str] | None = None,
                   columns: list[str] | None = None,
                   processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load processed parquet files, filtered by file stem and/or subset.

    EDA must call this with subset="prelim_train" so test-set distributions never inform design.
    """
    files = sorted(Path(processed_dir).glob("*.parquet"))
    if names is not None:
        names = set(names)
        files = [f for f in files if f.stem in names]
    if subset is not None:
        subsets = {subset} if isinstance(subset, str) else set(subset)
        unknown = subsets - set(SUBSET_CATS)
        if unknown:
            raise ValueError(f"unknown subset(s) {unknown}")
        files = [f for f in files if processed_subset(f) in subsets]
    if not files:
        raise FileNotFoundError(f"no processed files match names={names} subset={subset} in {processed_dir}")

    frames = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Concat of per-file categoricals with different categories falls back to strings; restore them.
    fixed = {"subset": SUBSET_CATS, "mode": MODE_CATS, "class": CLASS_CATS, "subclass": SUBCLASS_CATS}
    for col, cats in fixed.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=cats)
    if "source_file" in df.columns:
        df["source_file"] = pd.Categorical(df["source_file"], categories=[f.stem for f in files])
    return df


if __name__ == "__main__":
    import time

    t0 = time.time()
    summary = process_all()
    print(f"\n{len(summary)} files, {summary['rows'].sum():,} rows in {time.time() - t0:.0f}s -> {PROCESSED_DIR}")
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", mi.rss)
        print(f"peak memory: {peak / 2**20:.0f} MB")
    except ImportError:
        pass
