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
- Label quirk (known, NOT relabelled): 502 Normal-labelled frames in the attack train files use 7 IDs
  absent from Pre_train_*_0 (0x4A4, 0x7C4, 0x7CC, 0x7D0, 0x7D4, 0x7D8, 0x7DC). Most are UDS negative
  responses (`03 7F xx ...`) sent <= 4 ms after a fuzzing frame. Expected source of false positives
  (e.g. id_seen_in_training=0). See reports/train_stats_check.md.
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
- Validation (splits/splits.json, built by `python -m src.splits`): DEFAULT is leave-one-day-out
  (2 folds: val {D_1, S_1} / val {D_2, S_2}), because same-day S/D twins share the attack script
  and leave-one-file-out would put the val file's twin in training. The 4 LOFO folds are secondary.
  Pre_train_*_0 (Normal-only) are always in training. Pre_submit_* / Fin_* stay held out.
- Per-ID timing stats (artifacts/train_stats.json) come from Pre_train_*_0 only, so they cannot
  leak any validation fold. Never put information from attack-file labels into that artifact.

## Architecture
Frame stream -> sliding window / per-ID history -> features -> model -> Normal/Attack
-> logger/DB -> Streamlit dashboard. Periodic (not continuous) retraining with a
validation gate: deploy new model only if it beats current one on a fixed test set.

## Features (per frame; src/features.py -> data/features/<stem>.parquet)
One streaming `CausalFeatureExtractor` (.step() per frame, .reset() per stream) for batch AND live,
so the live demo cannot drift from training. Feature lists: BEHAVIORAL (dt_same_id, dt_ratio,
is_first_of_id, dt_prev_frame, id_count_last_100ms, frame_count_last_{50,100,200}ms,
payload_changed, n_bytes_changed, payload_repeat_age, payload_seen_before, byte_0..7, dlc, mode)
and ID_INCLUSIVE = BEHAVIORAL + arb_id, id_seen_in_training. META_COLS are never features.
Ablation: ID-inclusive vs pure-behavioral. The behavioral-only result is the headline result
(ID features make Flooding trivial).
- Sentinel -1 everywhere (all real values >= 0): first frame of an ID / of the stream, payload not seen.
- Windows count PAST frames (processed earlier) with t' > t - W; the current frame is excluded.
- dt_ratio = dt_same_id / per-ID median period; IDs not in train_stats use global_ref_period_s
  (~10.24 ms) with NO flag or special sentinel, so dt_ratio stays purely behavioral. ID knowledge
  lives only in ID_INCLUSIVE (a unit test enforces this).
- Payload history is bounded: payload_history_max_age_s = 10 s (BATCH_PAYLOAD_HISTORY_MAX_AGE_S),
  used for the feature files AND the live demo. Unbounded history made the Normal "seen before" rate
  climb with time since stream start (an implicit position-in-file feature); 10 s was the smallest
  bound that removes the drift without hurting Replay (notebooks/02_features.ipynb, section 4c).
- Window: all of 50 / 100 / 200 ms are features. 200 ms is the most robust for Spoofing/Replay
  (50 / 100 ms flip direction for Spoofing in one file); pick a single default in Phase 4 if needed.

## EDA findings (Phase 2, prelim_train only; see notebooks/01_eda.ipynb)
- Fuzzing uses IDs seen in Normal traffic (random payloads on known IDs). Refined in Phase 3:
  7.7% of Fuzzing frames use the 7 IDs absent from Pre_train_*_0, so id_seen_in_training (built
  from _0) flags only those. Payload and dt features must carry Fuzzing.
- Flooding = ID 0x000 only (never in Normal), so ID features catch it trivially.
- S and D are nearly identical (same 80 IDs, ~2,400 frames/s, same periods), so use one
  shared model with a `mode` feature.

## Feature diagnostics (Phase 3, train files only; see notebooks/02_features.ipynb)
- Behavioral features cover every attack (univariate separability, pooled): Flooding 1.00
  (dt_ratio / id_count_last_100ms), Spoofing 0.98 (dt_ratio), Replay 0.94 (dt_ratio),
  Fuzzing 0.99 (n_bytes_changed). dt_ratio is the strongest single feature.
- payload_seen_before / payload_repeat_age are WEAK for Replay (~0.55; direction flips in one file):
  Replay's repeat ages match Normal's (median 0.21 vs 0.20 s). dt_ratio is what detects Replay.
  The payload features do carry Fuzzing (never-seen payloads, ~0.87).
- The bounded Normal "seen before" rate differs by mode (S ~86-90%, D ~58-61%), so the `mode`
  feature matters for the payload features even though S and D look alike otherwise.
- Normal frames within +-1 s of an attack frame (42-45% of Normal in attack files) have inflated
  bus counts: exclude them when training anomaly models. Sampling proposal is in the notebook
  (written, not applied).

## Metrics
Per-SubClass precision/recall/F1, ROC-AUC, FPR at threshold.

## Working style
- Propose a plan before writing code. Explain WHY for each design choice briefly.
- Small steps. Code in src/ (reusable functions), notebooks/ for EDA only.
- Python venv, requirements.txt kept updated.