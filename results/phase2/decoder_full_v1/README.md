# Phase 2: decoder_full_v1

Legal named entity recognition using Full fine-tuning. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

The backbone and BIO token-classification head are trainable.

The [current notebook](../../../CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `decoder_full` / `v1` |
| Phase 1 origin | Decoder-only Transformer |
| Method | Full fine-tuning |
| Original Phase 1 parameter count | 110,418,432 |
| Task model total / trainable parameters | 110,437,657 / 110,437,657 |
| Seed / configured epochs | 42 / 3 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 2e-05 |
| Maximum sequence length / window overlap | 384 / 96 |
| Training / held-out validation documents | 60 / 10 |
| Weight decay / warmup fraction | 0.01 / 0.1 |
| Head dropout | 0.1 |
| Training complete (saved flag) | True |

This older configuration does not record head learning-rate multiplication, class-weight settings, or decoder optimization. Do not fill these fields from the current notebook defaults.

Original Phase 1 configuration stored in the metadata:

```json
{
  "vocab_size": 32000,
  "n_layer": 12,
  "n_embd": 768,
  "n_head": 12,
  "d_ff": 3072,
  "max_len": 1024
}
```

Exact settings and split IDs: [run_config.json](run_config.json). Parameter counts, adapter targets, selected blocks, and teacher identifier: [model_metadata.json](model_metadata.json). Runtime details: [environment.json](environment.json). The original Phase 1 count may include components not used by the downstream task model.

## Held-out validation results

Strict entity matching requires the exact character span and entity label. Scores below use the 0–1 scale and average over all 12 entity labels. They are checkpoint/decoder selection scores, not private test estimates.

| Macro-F1 | Macro-precision | Macro-recall |
| ---: | ---: | ---: |
| 0.25517 | 0.23696 | 0.29334 |

A separate baseline/selected-decoder comparison was not exported for this run.

### Per-label results

The table is rendered from [metrics.json](metrics.json); no separate per-label CSV was exported.

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| APP | 0.02857 | 0.07143 | 0.04082 | 14 |
| RESP | 0.03846 | 0.07143 | 0.05000 | 14 |
| A.COUNSEL | 0.06061 | 0.11765 | 0.08000 | 17 |
| R.COUNSEL | 0.02857 | 0.07692 | 0.04167 | 13 |
| JUDGE | 0.22500 | 0.39130 | 0.28571 | 23 |
| WIT | 0.00000 | 0.00000 | 0.00000 | 23 |
| AUTH | 0.24576 | 0.24370 | 0.24473 | 119 |
| COURT | 0.51327 | 0.71605 | 0.59794 | 81 |
| STAT | 0.56757 | 0.44366 | 0.49802 | 142 |
| PREC | 0.12500 | 0.18519 | 0.14925 | 108 |
| DATE | 0.89837 | 0.93249 | 0.91511 | 237 |
| CASENO | 0.11236 | 0.27027 | 0.15873 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

No standalone `verification.json` was included in this export. A post-training validation report and decoder-selection artifacts were not included. The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Expected checkpoint: `MyDrive/cs781_phase2/decoder_full_v1/best_model.pt`, following the original folder layout. This path has not been independently verified. Weights, predictions, and submission ZIPs remain in Drive. Historical code commits, checkpoint checksums, and teacher checkpoint hashes where applicable still need recording.

[All Phase 2 results](../) · [All results](../../)
