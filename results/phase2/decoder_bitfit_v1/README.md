# Phase 2: decoder_bitfit_v1

Legal named entity recognition using BitFit. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

The current implementation updates backbone biases and the BIO classification head while freezing the other backbone parameters.

The [current notebook](../../../CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `decoder_bitfit` / `v1` |
| Phase 1 origin | Decoder-only Transformer |
| Method | BitFit |
| Original Phase 1 parameter count | 110,418,432 |
| Task model total / trainable parameters | 110,437,657 / 121,369 |
| Seed / configured epochs | 42 / 3 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 0.0002 |
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
| 0.05199 | 0.06561 | 0.05913 |

A separate baseline/selected-decoder comparison was not exported for this run.

### Per-label results

The table is rendered from [metrics.json](metrics.json); no separate per-label CSV was exported.

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| APP | 0.00000 | 0.00000 | 0.00000 | 14 |
| RESP | 0.00000 | 0.00000 | 0.00000 | 14 |
| A.COUNSEL | 0.00000 | 0.00000 | 0.00000 | 17 |
| R.COUNSEL | 0.00000 | 0.00000 | 0.00000 | 13 |
| JUDGE | 0.00000 | 0.00000 | 0.00000 | 23 |
| WIT | 0.00000 | 0.00000 | 0.00000 | 23 |
| AUTH | 0.25000 | 0.01681 | 0.03150 | 119 |
| COURT | 0.20909 | 0.28395 | 0.24084 | 81 |
| STAT | 0.03509 | 0.01408 | 0.02010 | 142 |
| PREC | 0.02920 | 0.07407 | 0.04188 | 108 |
| DATE | 0.26389 | 0.32068 | 0.28952 | 237 |
| CASENO | 0.00000 | 0.00000 | 0.00000 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

No standalone `verification.json` was included in this export. A post-training validation report and decoder-selection artifacts were not included. The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Expected checkpoint: `MyDrive/cs781_phase2/decoder_bitfit_v1/best_model.pt`, following the original folder layout. This path has not been independently verified. Weights, predictions, and submission ZIPs remain in Drive. Historical code commits, checkpoint checksums, and teacher checkpoint hashes where applicable still need recording.

[All Phase 2 results](../) · [All results](../../)
