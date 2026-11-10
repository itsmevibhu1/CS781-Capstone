# Phase 2: decoder_full_v4

Legal named entity recognition using Full fine-tuning. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

The backbone and BIO token-classification head are trainable.

The [current notebook](../../../Phase2/CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `decoder_full` / `v4` |
| Phase 1 origin | Decoder-only Transformer |
| Method | Full fine-tuning |
| Original Phase 1 parameter count | 110,418,432 |
| Task model total / trainable parameters | 110,437,657 / 110,437,657 |
| Seed / configured epochs | 42 / 6 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 2e-05 |
| Maximum sequence length / window overlap | 384 / 96 |
| Training / held-out validation documents | 60 / 10 |
| Weight decay / warmup fraction | 0.01 / 0.1 |
| Head dropout | 0.1 |
| Training complete (saved flag) | True |
| Head learning-rate multiplier | 5.0 |
| Class weights / O loss weight / maximum entity weight | True / 0.35 / 3.0 |

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
| 0.48045 | 0.46459 | 0.52221 |

At the saved checkpoint, greedy macro-F1 is 0.36361; the selected decoder achieves 0.48045. This is a decoding comparison at one checkpoint.

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
| APP | 0.26471 | 0.64286 | 0.37500 | 14 |
| RESP | 0.50000 | 0.42857 | 0.46154 | 14 |
| A.COUNSEL | 0.39130 | 0.52941 | 0.45000 | 17 |
| R.COUNSEL | 0.58333 | 0.53846 | 0.56000 | 13 |
| JUDGE | 0.52174 | 0.52174 | 0.52174 | 23 |
| WIT | 0.00000 | 0.00000 | 0.00000 | 23 |
| AUTH | 0.38053 | 0.36134 | 0.37069 | 119 |
| COURT | 0.58879 | 0.77778 | 0.67021 | 81 |
| STAT | 0.71560 | 0.54930 | 0.62151 | 142 |
| PREC | 0.30631 | 0.31481 | 0.31050 | 108 |
| DATE | 0.95359 | 0.95359 | 0.95359 | 237 |
| CASENO | 0.36923 | 0.64865 | 0.47059 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

The saved [verification.json](verification.json) reports passing validation, prediction ID/offset, and ZIP/JSON checks. The saved [post-training validation report](post_training_validation.json) reports checkpoint-reload agreement, validation schema/ID checks, JSON round-trip, and support-count verification. Decode evidence: [decoder_config.json](decoder_config.json) and [decoding_candidates.json](decoding_candidates.json). Saved class weights: [loss_class_weights.json](loss_class_weights.json). The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Weights and checkpoint files are excluded from Git; checkpoint contents have not been independently verified.

[All Phase 2 results](../) · [All results](../../)
