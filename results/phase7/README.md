# Phase 7 results

Six trained retrieval runs plus one lexical baseline. Scores below are selected dev ranking scores over 118 queries and the corresponding candidate pool.

| Run | Origin | Method | Selected Micro-F1@7 | Dense-only Micro-F1@7 |
| --- | --- | --- | ---: | ---: |
| [decoder_partial_v1](decoder_partial_v1/) | decoder_only | Partial fine-tuning | 0.23013 | 0.19233 |
| [encoder_full_v1](encoder_full_v1/) | encoder_decoder | Full fine-tuning | 0.21679 | 0.17899 |
| [hybrid_lorma_v1](hybrid_lorma_v1/) | hybrid_prefixlm | LoRMA+ | 0.23013 | 0.18344 |
| [lightweight_ia3_v1](lightweight_ia3_v1/) | lightweight | IA³ | 0.19789 | 0.15675 |
| [lightweight_kd_v1](lightweight_kd_v1/) | lightweight | Knowledge distillation | 0.24458 | 0.21123 |
| [lightweight_prompt_v1](lightweight_prompt_v1/) | lightweight | Soft prompt tuning | 0.18566 | 0.11451 |
| [bm25_baseline](bm25_baseline/) | Lexical | BM25; no neural training | 0.11895 | Not applicable |

Highest recorded selected dev Micro-F1@7: **0.24458** from `lightweight_kd_v1`. This is a selection-score comparison among imported runs, not a private-test ranking. All saved trained-run completion and boolean verification flags pass. The official best submission uniquely matches `lightweight_kd_v1` by exact ranked prediction content; see the official result below.

Selected policies use dev-selected neural/BM25 rank fusion; the dense-only column prevents attributing all gains to the neural model. Late interaction is used in some policies. The KD teacher metadata records selection by dense Micro-F1@7. Architecture, adaptation and policy can vary together, so these comparisons do not isolate one change. Configurations, scores, ranking policies, histories, and plots are included in the run folders; model weights remain outside Git.

## Phase-level evidence

- [best_per_architecture.json](best_per_architecture.json)
- [coverage.json](coverage.json)
- [dataset_manifest.json](dataset_manifest.json)
- [document_lengths.json](document_lengths.json)
- [experiment_summary.csv](experiment_summary.csv)
- [export_manifest.json](export_manifest.json)

Coverage files report local method/origin/distinct-output checks, with competition acceptance explicitly unverified. Those saved checks are not independently rerun without the original runtime artifacts.

[Implementation](../../Phase7/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)

## Official best submission and verified run match

Submission **1039793**, reported rank **122**, records displayed **Micro-F1@7 0.2162**, **Micro-Precision@7 0.2206**, and **Micro-Recall@7 0.2119**. These are rounded official leaderboard values, separate from the local-dev metrics above. The author confirmed **6 distinct valid submissions**; rank-recording date remains unconfirmed.

[official_best.json](official_best.json) preserves the supplied screenshot values. The official ZIP contains 237 queries with seven distinct ranked candidate IDs each. Exact matching identifies **`lightweight_kd_v1` (lightweight student, knowledge distillation, tag `v1`)**. Both saved prediction files match; only this run matched among 14 compared artifacts, without errors. The configuration checksum agrees with the local copy. Evidence: [best_submission_match.json](best_submission_match.json). The saved policy combines dense neural and BM25 rankings; weights were not reloaded. Neither local-dev scores nor upload timestamps are used to guess the model.
