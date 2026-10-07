# CAN-IDS: Real-Time CAN Bus Intrusion Detection (ML project)

## Goal
Pure ML portfolio project. Detect CAN bus attacks (Flooding, Spoofing, Replay, Fuzzing)
using anomaly detection / classification on vehicle CAN traffic.

## Dataset: HCRL Car Hacking: Attack & Defense Challenge 2020 (Hyundai Avante CN7)
- Location: data/raw/ (NEVER modify files here)
- Columns: Timestamp (UNIX, seconds, float, keep full precision), Arbitration_ID (HEX string),
  DLC (0-8), Data (space-separated hex bytes, up to 8), Class (Normal/Attack),
  SubClass (Flooding/Spoofing/Replay/Fuzzing/Normal)
- Quirks: Pre_train_*_0 have no SubClass column (Normal-only). Final file timestamps are
  session-relative, not UNIX, and it has DLC=0 frames with empty Data.
- Filenames contain S = vehicle stationary, D = vehicle driving. Keep a `mode` column.
- Preliminary round = train/val/test. Final round = optional extra held-out test.

## Hard rules
- Load Arbitration_ID and Data as str (dtype). IDs are hex: int(x, 16).
- Laptop: i5, 8GB RAM, no GPU. Use pd.read_csv(chunksize=...) and downcast dtypes.
- Split by FILE, not random rows (prevents temporal leakage). Save split file lists.
- Anomaly models train on Normal only; labels used for evaluation.
- Features are computed per frame from history only (past frames), never future frames.
- NEVER use timestamp or row_idx (or any absolute time / position in file) as model features:
  attack schedules are nearly identical across files, so these leak the label.
- Validation: leave-one-file-out over the 4 attack train files (Pre_train_{S,D}_{1,2});
  Pre_train_*_0 (Normal-only) are always in training. Pre_submit_* / Fin_* stay held out.

## Architecture
Frame stream -> sliding window / per-ID history -> features -> model -> Normal/Attack
-> logger/DB -> Streamlit dashboard. Periodic (not continuous) retraining with a
validation gate: deploy new model only if it beats current one on a fixed test set.

## Planned features (per frame)
dt_same_id, frame_count_last_N_ms, payload_changed, payload_seen_before (replay),
id_seen_in_training, byte_0..byte_7 (padded), mode.
Ablation: ID-inclusive vs pure-behavioral feature sets. The behavioral-only result is the
headline result (ID features make Flooding trivial).
- Window: 100 ms default for frame_count_last_N_ms; compare 50 / 200 ms in Phase 3.

## EDA findings (Phase 2, prelim_train only; see notebooks/01_eda.ipynb)
- Fuzzing uses only IDs seen in Normal traffic (random payloads on known IDs), so
  id_seen_in_training will NOT catch it. Payload and dt features must carry Fuzzing.
- Flooding = ID 0x000 only (never in Normal), so ID features catch it trivially.
- S and D are nearly identical (same 80 IDs, ~2,400 frames/s, same periods), so use one
  shared model with a `mode` feature.

## Metrics
Per-SubClass precision/recall/F1, ROC-AUC, FPR at threshold.

## Working style
- Propose a plan before writing code. Explain WHY for each design choice briefly.
- Small steps. Code in src/ (reusable functions), notebooks/ for EDA only.
- Python venv, requirements.txt kept updated.