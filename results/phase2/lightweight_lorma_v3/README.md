# Phase 2: lightweight_lorma_v3

Legal named entity recognition using LoRMA+. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

Attention projections use the notebook’s LoRMA+ multiplicative low-rank adaptation while the backbone weights are frozen; the BIO classification head is trainable. Exact targets are preserved in the model metadata.

The [current notebook](../../../Phase2/CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `lightweight_lorma` / `v3` |
| Phase 1 origin | Lightweight Transformer |
| Method | LoRMA+ |
| Original Phase 1 parameter count | 102,451,968 |
| Task model total / trainable parameters | 102,839,833 / 387,865 |
| Seed / configured epochs | 42 / 3 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 0.0002 |
| Maximum sequence length / window overlap | 384 / 96 |
| Training / held-out validation documents | 60 / 10 |
| Weight decay / warmup fraction | 0.01 / 0.1 |
| Head dropout | 0.1 |
| Training complete (saved flag) | True |
| Head learning-rate multiplier | 5.0 |
| Class weights / O loss weight / maximum entity weight | True / 0.35 / 3.0 |
| Adapter rank / alpha | 8 / 8.0 |
| Adapter target modules | 48 |

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
| 0.38958 | 0.37038 | 0.43382 |

At the saved checkpoint, greedy macro-F1 is 0.25193; the selected decoder achieves 0.38958. This is a decoding comparison at one checkpoint.

Selected decoder:

```json
{
  "method": "viterbi",
  "entity_bias": 0.0
}
```

### Per-label results

The table is rendered from [metrics.json](metrics.json); no separate per-label CSV was exported.

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| APP | 0.26923 | 0.50000 | 0.35000 | 14 |
| RESP | 0.25000 | 0.35714 | 0.29412 | 14 |
| A.COUNSEL | 0.26667 | 0.47059 | 0.34043 | 17 |
| R.COUNSEL | 0.30000 | 0.23077 | 0.26087 | 13 |
| JUDGE | 0.36111 | 0.56522 | 0.44068 | 23 |
| WIT | 0.00000 | 0.00000 | 0.00000 | 23 |
| AUTH | 0.50847 | 0.50420 | 0.50633 | 119 |
| COURT | 0.60825 | 0.72840 | 0.66292 | 81 |
| STAT | 0.53846 | 0.34507 | 0.42060 | 142 |
| PREC | 0.24731 | 0.21296 | 0.22886 | 108 |
| DATE | 0.86066 | 0.88608 | 0.87318 | 237 |
| CASENO | 0.23438 | 0.40541 | 0.29703 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

The saved [verification.json](verification.json) reports passing validation, prediction ID/offset, and ZIP/JSON checks. The saved [post-training validation report](post_training_validation.json) reports checkpoint-reload agreement, validation schema/ID checks, JSON round-trip, and support-count verification. Decode evidence: [decoder_config.json](decoder_config.json) and [decoding_candidates.json](decoding_candidates.json). Saved class weights: [loss_class_weights.json](loss_class_weights.json). The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Weights and checkpoint files are excluded from Git; checkpoint contents have not been independently verified.

[All Phase 2 results](../) · [All results](../../)
