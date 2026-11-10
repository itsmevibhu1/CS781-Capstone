# Phase 5 results

Six saved runs are documented below. Scores use the 0–1 scale and development labels were used for checkpoint and threshold selection. No Codabench score or acceptance is established by these artifacts.

| Run | Origin | Method | Macro-F1 | Micro-F1 | Subset accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| [decoder_full_v1](decoder_full_v1/) | decoder_only | Full fine-tuning | 0.36727 | 0.46086 | 0.03791 |
| [encoder_partial_v1](encoder_partial_v1/) | encoder_decoder | Partial encoder fine-tuning | 0.28858 | 0.37688 | 0.01385 |
| [hybrid_adapters_v1](hybrid_adapters_v1/) | hybrid_prefixlm | Residual bottleneck adapters | 0.35353 | 0.44285 | 0.03015 |
| [hybrid_lorma_v1](hybrid_lorma_v1/) | hybrid_prefixlm | LoRMA+ | 0.34514 | 0.44130 | 0.02573 |
| [lightweight_kd_v1](lightweight_kd_v1/) | lightweight | Knowledge distillation | 0.38345 | 0.47414 | 0.03585 |
| [lightweight_lorma_v1](lightweight_lorma_v1/) | lightweight | LoRMA+ | 0.35435 | 0.44902 | 0.02848 |

Highest recorded development macro-F1: **0.38345** from `lightweight_kd_v1`. This ranking describes imported development selection scores, not private test performance. Architecture and adaptation vary together across several runs; these are descriptive comparisons and do not isolate one design choice.

All imported Phase 5 configurations use `reference_range` for macro-F1 over statute decisions. The per-run CSVs and metrics JSONs preserve label counts and alternative scope metrics. Do not compare these scores with Phase 4 binary-class macro-F1.

All saved training-complete flags and boolean verification checks pass. Historical training commits, standalone environment reports, and official test scores were not exported. Missing ensemble recipes are expected for these individual-model folders; no ensemble was imported.

## Phase-level evidence

- [best_per_architecture.json](best_per_architecture.json)
- [coverage.json](coverage.json)
- [experiment_summary.csv](experiment_summary.csv)
- [export_manifest.json](export_manifest.json)

The saved coverage report marks local completion/method/origin/distinct-prediction checks as passed and explicitly marks Codabench acceptance as unverified. These saved checks are not independently reproducible without the predictions and checkpoints retained in configured storage.

[Implementation and instructions](../../Phase5/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)

## Matched official best submission

Submission **1011686**, reported rank **80**, records displayed **Macro-F1 0.3622**, **Micro-F1 0.4565**, and **Subset Accuracy 0.0336**. These are rounded four-decimal leaderboard values, not the local-dev scores above. The downloaded archive contains **13,019 case predictions**. The author confirmed **6 distinct valid submissions**; the rank-recording date is unconfirmed.

[official_best.json](official_best.json) preserves the supplied metrics and prediction fingerprints. **Matched run: `lightweight_kd_v1`**, a lightweight student trained with knowledge distillation. The [match report](best_submission_match.json) compares 12 saved prediction artifacts with no errors. Both this run’s JSON and ZIP match all 13,019 official case predictions; no other compared run matches. Its saved configuration checksum matches the local `run_config.json`. This identifies the saved run without reloading checkpoint weights.
