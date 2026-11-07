# Phase 6 results

Six trained summarization runs. Final scores below cover all 351 held-out documents; checkpoints were selected on a fixed 128-document subset. The public dataset has no official dev file.

| Run | Origin | Method | BERTScore F1 | ROUGE-1 F1 | ROUGE-L F1 |
| --- | --- | --- | ---: | ---: | ---: |
| [decoder_lora_v1](decoder_lora_v1/) | decoder_only | LoRA | 0.83919 | 0.33788 | 0.19092 |
| [encoder_adapters_v1](encoder_adapters_v1/) | encoder_decoder | Houlsby adapters | 0.82665 | 0.32482 | 0.17886 |
| [encoder_full_v1](encoder_full_v1/) | encoder_decoder | Full fine-tuning | 0.82825 | 0.32506 | 0.18014 |
| [hybrid_lorma_v1](hybrid_lorma_v1/) | hybrid_prefixlm | LoRMA+ | 0.82979 | 0.33932 | 0.18627 |
| [lightweight_kd_v1](lightweight_kd_v1/) | lightweight | Knowledge distillation | 0.83660 | 0.31707 | 0.17929 |
| [lightweight_prompt_v1](lightweight_prompt_v1/) | lightweight | Soft prompt tuning | 0.82659 | 0.32605 | 0.17827 |

Highest recorded full held-out BERTScore F1: **0.83919** from `decoder_lora_v1`. This is a selection-score comparison among imported runs, not a private-test ranking. All saved trained-run completion and boolean verification flags pass. Codabench IDs/scores and acceptance remain unrecorded.

BERTScore uses `roberta-large`, layer 17, no IDF and no baseline rescaling; scorer hashes are preserved per run. ROUGE uses stemming. Epoch-selection scores and full-held-out scores are distinct. Source selection is limited to 640 tokens and generation to 256 new tokens, so full validation does not imply full input coverage or full-length summaries. See the saved split/quality reports for counts and exclusions.

## Phase-level evidence

- [best_per_architecture.json](best_per_architecture.json)
- [coverage.json](coverage.json)
- [data_quality_report.json](data_quality_report.json)
- [excluded_training_rows.json](excluded_training_rows.json)
- [experiment_summary.csv](experiment_summary.csv)
- [export_manifest.json](export_manifest.json)
- [split_manifest.json](split_manifest.json)

Coverage files report local method/origin/distinct-output checks, with competition acceptance explicitly unverified. Those saved checks are not independently rerun without the Drive artifacts.

[Implementation](../../Phase6/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)
