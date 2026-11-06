# Phase 4 results

Six saved runs are documented below. Scores use the 0–1 scale and development labels were used for checkpoint and threshold selection. No Codabench score or acceptance is established by these artifacts.

| Run | Origin | Method | Macro-F1 | Class-1 precision | Class-1 recall |
| --- | --- | --- | ---: | ---: | ---: |
| [decoder_bitfit_v1](decoder_bitfit_v1/) | decoder_only | BitFit | 0.60625 | 0.65135 | 0.48491 |
| [decoder_full_v1](decoder_full_v1/) | decoder_only | Full fine-tuning | 0.65169 | 0.67352 | 0.59356 |
| [encoder_partial_v1](encoder_partial_v1/) | encoder_decoder | Partial encoder fine-tuning | 0.58567 | 0.61097 | 0.49296 |
| [hybrid_adapters_v1](hybrid_adapters_v1/) | hybrid_prefixlm | Residual bottleneck adapters | 0.64295 | 0.67813 | 0.55533 |
| [lightweight_kd_v1](lightweight_kd_v1/) | lightweight | Knowledge distillation | 0.64027 | 0.66357 | 0.57545 |
| [lightweight_lorma_v1](lightweight_lorma_v1/) | lightweight | LoRMA+ | 0.68233 | 0.70312 | 0.63380 |

Highest recorded development macro-F1: **0.68233** from `lightweight_lorma_v1`. This ranking describes imported development selection scores, not private test performance. Architecture and adaptation vary together across several runs; these are descriptive comparisons and do not isolate one design choice.

Phase 4 macro-F1 averages classes 0 and 1; precision/recall in the table are for class 1.

All saved training-complete flags and boolean verification checks pass. Historical training commits, standalone environment reports, and official test scores were not exported. Missing ensemble recipes are expected for these individual-model folders; no ensemble was imported.

## Phase-level evidence

- [export_manifest.json](export_manifest.json)

The Phase 4 export contains no root-level experiment summary or coverage report; the table above is built from individual run metrics. Six folders alone do not establish distinct accepted submissions.

[Implementation and instructions](../../Phase4/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)
