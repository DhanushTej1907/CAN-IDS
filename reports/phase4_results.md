# Phase 4 results

## Validation (prelim train files, leave-one-day-out CV)

**Setup.** Two folds from `splits/splits.json`: `lodo_day1` validates on {D_1, S_1}, `lodo_day2` on
{D_2, S_2}. Pre_train_*_0 (Normal-only) are always in training. Pre_submit_* / Fin_* were not used.
Models: Isolation Forest (trained on Normal only), a rule baseline, and XGBoost (supervised), each on
the behavioral and ID-inclusive feature sets. Selection rule: mean over folds of macro recall
(4 SubClasses) at 0.1% FPR; ties broken by all-attack PR-AUC. Status: `selected.json` has
`approved=false`; no test-set evaluation yet.

**Recall per SubClass at 0.1% FPR (selected config per model × feature set)**

| Model | Features | Fold | Flooding | Spoofing | Replay | Fuzzing | Macro |
|---|---|---|---|---|---|---|---|
| XGBoost (depth 8, w_spoof 1) | behavioral | day1 | 1.000 | 0.992 | 0.961 | 0.999 | 0.988 |
| | | day2 | 1.000 | 0.546 | 0.603 | 1.000 | 0.787 |
| XGBoost (depth 6, w_spoof 5) | ID-inclusive | day1 | 1.000 | 0.991 | 0.964 | 0.999 | 0.988 |
| | | day2 | 1.000 | 0.863 | 0.646 | 1.000 | 0.877 |
| Isolation Forest (300 trees, 4096 samples) | behavioral | day1 | 0.000 | 0.000 | 0.004 | 0.519 | 0.131 |
| | | day2 | 0.000 | 0.002 | 0.005 | 0.646 | 0.163 |
| Isolation Forest (300 trees, 4096 samples) | ID-inclusive | day1 | 0.000 | 0.000 | 0.004 | 0.587 | 0.148 |
| | | day2 | 0.000 | 0.000 | 0.007 | 0.669 | 0.169 |
| Rule baseline | behavioral | day1 | 0.000 | 0.045 | 0.045 | 0.081 | 0.043 |
| | | day2 | 0.000 | 0.059 | 0.049 | 0.082 | 0.048 |
| Rule baseline | ID-inclusive | day1 | 1.000 | 0.044 | 0.044 | 0.151 | 0.310 |
| | | day2 | 1.000 | 0.057 | 0.047 | 0.151 | 0.314 |

**Findings**
- **Main result (behavioral XGBoost):** macro recall 0.888 averaged over folds, 0.787 on the worse
  fold; all-attack PR-AUC 0.997. Adding ID features raises the average to 0.933; nearly all of
  the gain is day-2 Spoofing (0.55 → 0.86).
- Every XGBoost config catches nearly all Flooding and Fuzzing on both days (recall ≥ 0.999).
  Spoofing and Replay account for the whole gap between folds.
- Neither baseline is usable at 0.1% FPR. Isolation Forest catches only Fuzzing (0.52–0.67). The
  rule baseline catches Flooding only when it has the ID features (ID 0x000 is never seen in Normal
  traffic).
- **Why day 2 is harder.** Spoofing on 0x553 runs at about 3× the injection rate on day 2 (≈1,445 vs
  494 frames per file; median dt_ratio 0.05 vs 0.30), a timing range the day-1-trained model never
  saw. Replay timing on 0x130 is the same on both days (median dt_ratio ≈ 0.5), so the Replay drop
  is not explained yet. Replay is spread evenly over many IDs, and the mix of IDs differs by file.

**Caveat.** Each fold's threshold is the 0.1% quantile of that fold's own validation Normal scores.
That makes validation FPR exactly 0.1% and makes the recall above slightly optimistic. The deployed
threshold will be the fold-mean quantile (`q` in `selected.json`), so FPR on the test set may differ.

## Test (held-out, evaluated once)

**Setup.** The approved configs were refit on all 6 train files and evaluated once on Pre_submit_D/S
(test) and Fin_host_session_submit_S (extra test). Each threshold uses the CV fold-mean quantile `q`
at 0.1% FPR, applied to the refit model's own training-Normal scores; no test labels were used.
Nothing was retuned after scoring. Outputs: `reports/phase4/test_metrics.csv`, `models/final/`.

**Pre_submit (D+S pooled, 3.75M frames), target FPR 0.1%**

| Model | Features | Flooding | Spoofing | Replay | Fuzzing | Macro | FPR |
|---|---|---|---|---|---|---|---|
| XGBoost (depth 8, w_spoof 1) | behavioral (headline) | 1.000 | 0.457 | 0.899 | 0.999 | **0.839** | 0.07% |
| XGBoost (depth 6, w_spoof 5) | ID-inclusive | 1.000 | 0.434 | 0.901 | 0.999 | 0.833 | 0.07% |
| Rule baseline | ID-inclusive | 1.000 | 0.519 | 0.035 | 0.149 | 0.426 | 0.11% |
| Rule baseline | behavioral | 0.000 | 0.783 | 0.036 | 0.075 | 0.223 | 0.12% |
| Isolation Forest | ID-inclusive | 0.000 | 0.000 | 0.010 | 0.666 | 0.169 | 0.09% |
| Isolation Forest | behavioral | 0.000 | 0.000 | 0.007 | 0.580 | 0.147 | 0.07% |

**Fin (Spoofing and Fuzzing only, 1.27M frames)**

| Model | Features | Spoofing | Fuzzing | FPR (target 0.1%) |
|---|---|---|---|---|
| XGBoost | behavioral | 1.000 | 0.979 | 0.35% |
| XGBoost | ID-inclusive | 0.483 | 0.952 | 0.85% |
| Rule baseline | ID-inclusive | 0.749 | 0.719 | 1.50% |
| Rule baseline | behavioral | 0.755 | 0.027 | 0.90% |
| Isolation Forest | ID-inclusive | 0.000 | 0.526 | 0.06% |
| Isolation Forest | behavioral | 0.000 | 0.473 | 0.06% |

**Findings**
- **Headline:** behavioral XGBoost reaches 0.839 macro recall at 0.07% FPR on Pre_submit, between
  the two CV folds (0.988 / 0.787). Flooding, Fuzzing and Replay are strong.
- **Spoofing is the weak spot** (0.46), below the single-feature dt_ratio rule (0.78). This matches
  the CV day-2 finding: the model learned one Spoofing injection rate and does not generalise to
  others.
- **ID features do not help on test** (0.833 vs 0.839), so behavioral stays the headline.
- **The thresholds do not hold FPR on Fin:** 3.5× to 15× the target for XGBoost and the rule. Fin
  recall is therefore not a 0.1% FPR result; the XGBoost behavioral figure should be quoted as
  0.99 at 0.35% FPR. The cause has not been investigated. Fin is a separate session
  (session-relative timestamps, DLC=0 frames), so a shift in its Normal traffic is the likely
  reason.
