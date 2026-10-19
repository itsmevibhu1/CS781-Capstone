# Phase 2: encoder_partial_v3

Legal named entity recognition using Partial fine-tuning. Values below come from saved artifacts; training was not rerun for documentation.

## Method and implementation

The current implementation uses the Phase 1 encoder-decoder encoder, adapting its last two encoder layers and final normalization together with the BIO classification head. Exact historical trainable names were not exported; parameter counts come from the saved metadata.

The [current notebook](../../../CS781_Phase2_Solution_Colab.ipynb) implements BPE tokenization with character offsets, overlapping windows, a BIO token-classification head, document-level prediction merging, and strict entity-span evaluation. See the [phase instructions](../../../Phase2/README.md). It initializes from Phase 1 weights. The original training code commit was not exported, so exact correspondence with the linked current implementation remains to be confirmed.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / tag | `encoder_partial` / `v3` |
| Phase 1 origin | Encoder-decoder Transformer |
| Method | Partial fine-tuning |
| Original Phase 1 parameter count | 123,762,176 |
| Task model total / trainable parameters | 67,105,817 / 14,190,361 |
| Seed / configured epochs | 42 / 3 |
| Batch size / gradient accumulation | 2 / 8 |
| Configured base learning rate | 5e-05 |
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
  "n_enc_layer": 6,
  "n_dec_layer": 6,
  "d_model": 768,
  "n_head": 8,
  "d_ff": 3072
}
```

Exact settings and split IDs: [run_config.json](run_config.json). Parameter counts, adapter targets, selected blocks, and teacher identifier: [model_metadata.json](model_metadata.json). Runtime details: [environment.json](environment.json). The original Phase 1 count may include components not used by the downstream task model.

## Held-out validation results

Strict entity matching requires the exact character span and entity label. Scores below use the 0–1 scale and average over all 12 entity labels. They are checkpoint/decoder selection scores, not private test estimates.

| Macro-F1 | Macro-precision | Macro-recall |
| ---: | ---: | ---: |
| 0.00000 | 0.00000 | 0.00000 |

At the saved checkpoint, greedy macro-F1 is 0.00000; the selected decoder achieves 0.00000. This is a decoding comparison at one checkpoint.

Selected decoder:

```json
{
  "method": "greedy",
  "entity_bias": 0.0
}
```

**Observed failure:** all 12 labels have zero true positives, and the saved counts contain zero false positives, consistent with no predicted entities on validation. This does not identify the cause; training completion is not evidence of successful entity recognition.

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
| AUTH | 0.00000 | 0.00000 | 0.00000 | 119 |
| COURT | 0.00000 | 0.00000 | 0.00000 | 81 |
| STAT | 0.00000 | 0.00000 | 0.00000 | 142 |
| PREC | 0.00000 | 0.00000 | 0.00000 | 108 |
| DATE | 0.00000 | 0.00000 | 0.00000 | 237 |
| CASENO | 0.00000 | 0.00000 | 0.00000 | 37 |

Checkpoint-selection scores: [best_validation_metrics.json](best_validation_metrics.json). Training history: [history.json](history.json).

## Training plot

![Saved training curves](training_curves.png)

## Verification and remaining records

The saved [verification.json](verification.json) reports passing validation, prediction ID/offset, and ZIP/JSON checks. The saved [post-training validation report](post_training_validation.json) reports checkpoint-reload agreement, validation schema/ID checks, JSON round-trip, and support-count verification. Decode evidence: [decoder_config.json](decoder_config.json) and [decoding_candidates.json](decoding_candidates.json). Saved class weights: [loss_class_weights.json](loss_class_weights.json). The metrics record training complete, but neither that flag nor exported verification establishes Codabench acceptance. The private test score and submission ID have not been recorded.

Expected checkpoint: `MyDrive/cs781_phase2/encoder_partial_v3/best_model.pt`, following the original folder layout. This path has not been independently verified. Weights, predictions, and submission ZIPs remain in Drive. Historical code commits, checkpoint checksums, and teacher checkpoint hashes where applicable still need recording.

[All Phase 2 results](../) · [All results](../../)
