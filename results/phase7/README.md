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

Highest recorded selected dev Micro-F1@7: **0.24458** from `lightweight_kd_v1`. This is a selection-score comparison among imported runs, not a private-test ranking. All saved trained-run completion and boolean verification flags pass. Codabench IDs/scores and acceptance remain unrecorded.

Selected policies use dev-selected neural/BM25 rank fusion; the dense-only column prevents attributing all gains to the neural model. Late interaction is used in some policies. The KD teacher metadata records selection by dense Micro-F1@7. Architecture, adaptation and policy can vary together, so these comparisons do not isolate one change. Phase 7 weights remain in the friend’s Drive, not automatically in the user’s account.

## Phase-level evidence

- [best_per_architecture.json](best_per_architecture.json)
- [coverage.json](coverage.json)
- [dataset_manifest.json](dataset_manifest.json)
- [document_lengths.json](document_lengths.json)
- [experiment_summary.csv](experiment_summary.csv)
- [export_manifest.json](export_manifest.json)

Coverage files report local method/origin/distinct-output checks, with competition acceptance explicitly unverified. Those saved checks are not independently rerun without the Drive artifacts.

[Implementation](../../Phase7/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)
