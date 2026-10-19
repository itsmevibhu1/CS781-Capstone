# Phase 2: lightweight_kd_v1

Legal named entity recognition using Knowledge distillation. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

The smaller lightweight student uses the recorded subset of Phase 1 blocks and trains its backbone and BIO head with supervised loss and teacher-logit distillation. The saved teacher identifier is documented below; a historical teacher checkpoint hash was not exported.

The [current notebook](../../../Phase2/CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `lightweight_kd` / `v1` |
| Phase 1 origin | Lightweight Transformer |
| Method | Knowledge distillation |
| Original Phase 1 parameter count | 102,451,968 |
| Task model total / trainable parameters | 63,533,593 / 63,533,593 |
| Seed / configured epochs | 42 / 3 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 5e-05 |
| Maximum sequence length / window overlap | 384 / 96 |
| Training / held-out validation documents | 60 / 10 |
| Weight decay / warmup fraction | 0.01 / 0.1 |
| Head dropout | 0.1 |
| Training complete (saved flag) | True |
| Phase 1 block indices (zero based) | `[0, 2, 4, 6, 8, 10]` |
| Teacher identifier | `decoder_full` |
| KD weight / temperature | 0.5 / 2.0 |

This older configuration does not record head learning-rate multiplication, class-weight settings, or decoder optimization. Do not fill these fields from the current notebook defaults.

Original Phase 1 configuration stored in the metadata:

```json
{
  "vocab_size": 32000,
  "n_layer": 12,
  "d_model": 768,
  "n_head": 12,
  "n_kv_head": 3,
  "d_ff": 2176,
  "max_len": 1024
}
```

Exact settings and split IDs: [run_config.json](run_config.json). Parameter counts, adapter targets, selected blocks, and teacher identifier: [model_metadata.json](model_metadata.json). Runtime details: [environment.json](environment.json). The original Phase 1 count may include components not used by the downstream task model.

## Held-out validation results

Strict entity matching requires the exact character span and entity label. Scores below use the 0–1 scale and average over all 12 entity labels. They are checkpoint/decoder selection scores, not private test estimates.

| Macro-F1 | Macro-precision | Macro-recall |
| ---: | ---: | ---: |
| 0.27944 | 0.26811 | 0.31091 |

A separate baseline/selected-decoder comparison was not exported for this run.

### Per-label results

The table is rendered from [metrics.json](metrics.json); no separate per-label CSV was exported.

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| APP | 0.00000 | 0.00000 | 0.00000 | 14 |
| RESP | 0.09091 | 0.14286 | 0.11111 | 14 |
| A.COUNSEL | 0.11765 | 0.23529 | 0.15686 | 17 |
| R.COUNSEL | 0.06061 | 0.15385 | 0.08696 | 13 |
| JUDGE | 0.15152 | 0.21739 | 0.17857 | 23 |
| WIT | 0.00000 | 0.00000 | 0.00000 | 23 |
| AUTH | 0.40698 | 0.29412 | 0.34146 | 119 |
| COURT | 0.60204 | 0.72840 | 0.65922 | 81 |
| STAT | 0.58879 | 0.44366 | 0.50602 | 142 |
| PREC | 0.15432 | 0.23148 | 0.18519 | 108 |
| DATE | 0.88400 | 0.93249 | 0.90760 | 237 |
| CASENO | 0.16049 | 0.35135 | 0.22034 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

No standalone `verification.json` was included in this export. A post-training validation report and decoder-selection artifacts were not included. The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Expected checkpoint: `MyDrive/cs781_phase2/lightweight_kd_v1/best_model.pt`, following the original folder layout. This path has not been independently verified. Weights, predictions, and submission ZIPs remain in Drive. Historical code commits, checkpoint checksums, and teacher checkpoint hashes where applicable still need recording.

[All Phase 2 results](../) · [All results](../../)
