# Phase 2 results

Fourteen saved runs cover six adaptation experiments: six `v1` runs, six `v3` runs, and decoder-only `v2` and `v4` variants. All configurations record 60 training and 10 held-out validation documents, with disjoint IDs.

| Run | Method | Epochs | Trainable parameters | Strict validation macro-F1 | Post-training / submission verification files |
| --- | --- | ---: | ---: | ---: | --- |
| [decoder_bitfit_v1](decoder_bitfit_v1/) | BitFit | 3 | 121,369 | 0.05199 | Not exported |
| [decoder_bitfit_v3](decoder_bitfit_v3/) | BitFit | 3 | 121,369 | 0.36200 | Present |
| [decoder_full_v1](decoder_full_v1/) | Full fine-tuning | 3 | 110,437,657 | 0.25517 | Not exported |
| [decoder_full_v2](decoder_full_v2/) | Full fine-tuning | 3 | 110,437,657 | 0.25517 | Not exported |
| [decoder_full_v3](decoder_full_v3/) | Full fine-tuning | 3 | 110,437,657 | 0.47663 | Present |
| [decoder_full_v4](decoder_full_v4/) | Full fine-tuning | 6 | 110,437,657 | 0.48045 | Present |
| [encoder_partial_v1](encoder_partial_v1/) | Partial fine-tuning | 3 | 14,190,361 | 0.00000 | Not exported |
| [encoder_partial_v3](encoder_partial_v3/) | Partial fine-tuning | 3 | 14,190,361 | 0.00000 | Present |
| [hybrid_lora_v1](hybrid_lora_v1/) | LoRA | 3 | 384,665 | 0.11463 | Not exported |
| [hybrid_lora_v3](hybrid_lora_v3/) | LoRA | 3 | 384,665 | 0.34674 | Present |
| [lightweight_kd_v1](lightweight_kd_v1/) | Knowledge distillation | 3 | 63,533,593 | 0.27944 | Not exported |
| [lightweight_kd_v3](lightweight_kd_v3/) | Knowledge distillation | 3 | 63,533,593 | 0.46162 | Present |
| [lightweight_lorma_v1](lightweight_lorma_v1/) | LoRMA+ | 3 | 387,865 | 0.09975 | Not exported |
| [lightweight_lorma_v3](lightweight_lorma_v3/) | LoRMA+ | 3 | 387,865 | 0.38958 | Present |

Highest saved validation macro-F1: **0.48045**, from `decoder_full_v4`. These are selection scores across 12 entity labels, not Codabench/private test results.

## Version comparisons and limitations

- `decoder_full_v1` and `decoder_full_v2` configurations differ only in their run tag and record the same macro-F1. Separate folder names alone do not establish distinct submission contents.
- Compared with older configurations, `v3` explicitly records class weighting, a higher head learning-rate multiplier, and decoder tuning. Several settings and code versions may differ, so the score changes do not isolate one cause.
- `decoder_full_v4` records six epochs versus three for `v3`; the other saved configuration fields match apart from the tag. Selection remains on the same held-out split.
- Both `encoder_partial` runs record zero strict F1 with no validation entity predictions in their saved counts. The artifacts do not explain the cause.
- All metrics record training complete. Older `v1`/`v2` exports lack the newer standalone verification and post-training reports; do not assume missing checks passed.
- Coverage files for `v1` and `v3` report six completed experiments and the required method/origin coverage. `v2` and `v4` report incomplete six-experiment coverage because they contain only one saved experiment each. These are local coverage reports, not evidence of accepted Codabench submissions.
- Exact training code commits and teacher checkpoint hashes are unavailable. Current implementation defaults must not be substituted for older saved settings.

## Phase-level artifacts

- [coverage_v1.json](coverage_v1.json)
- [coverage_v2.json](coverage_v2.json)
- [coverage_v3.json](coverage_v3.json)
- [coverage_v4.json](coverage_v4.json)
- [experiment_summary_v1.csv](experiment_summary_v1.csv)
- [experiment_summary_v2.csv](experiment_summary_v2.csv)
- [experiment_summary_v3.csv](experiment_summary_v3.csv)
- [experiment_summary_v4.csv](experiment_summary_v4.csv)
- [export_manifest.json](export_manifest.json)
- [labels.json](labels.json)

The export manifest records included and missing artifacts. Per-label scores are available in each run’s metrics JSON and README even though no standalone per-label CSV was exported.

[Phase 2 implementation](../../Phase2/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)

## Matched official best submission

The user supplied submission **978578**, reported rank **88**, with displayed **strict Macro-F1 0.4775**, **strict Macro-Precision 0.4685**, and **strict Macro-Recall 0.4946**. These are four-decimal leaderboard values; unrounded detailed results have not been supplied. The submitted JSON contains **35 documents** and **4,263 entity spans**.

[official_best.json](official_best.json) records the screenshots and ZIP/prediction fingerprints. **Matched run: `decoder_full_v3`, decoder-only architecture with full fine-tuning.** The [match report](best_submission_match.json) compares 28 saved prediction artifacts without errors. Both this run’s `prediction.json` and `submission.zip` match all 35 official document predictions; no other compared run matches. Its reported saved configuration checksum also matches the local `run_config.json`. Weights were not loaded for this identification. The author confirmed **16 distinct valid submissions**. The rank-recording date remains pending.
