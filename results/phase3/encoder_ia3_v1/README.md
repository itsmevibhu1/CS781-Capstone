# Phase 3: encoder_ia3_v1

Rhetorical role prediction for CL and IT documents using IA³ with the Phase 1 encoder-decoder encoder origin. This page records the exported configuration and metrics; training was not rerun for documentation.

## Method and implementation

The encoder from the Phase 1 encoder-decoder model is used for classification. The backbone weights are frozen; IA³ key/value and feed-forward scaling vectors and the classification head are trained. The Phase 1 parameter count describes the full original model, while the task model uses its encoder.

The current [Phase 3 notebook](../../../CS781_Phase3_Solution_Colab.ipynb) contains data loading, BPE tokenization, context-aware sentence inputs, classification, training, and prediction. See the [phase instructions](../../../Phase3/README.md). All downstream runs initialize from Phase 1 rather than Phase 2 weights.

The original training code commit was not recorded in these artifacts. The linked notebook is the current repository version; its exact correspondence with the historical training code still needs confirmation.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / run tag | `encoder_ia3` / `v1` |
| Phase 1 origin | Encoder-decoder encoder |
| Adaptation | IA³ |
| Original Phase 1 parameters | 123,762,176 |
| Task model total / trainable parameters | 68,326,253 / 1,239,661 |
| Trainable fraction | 1.81% |
| Seed | 42 |
| Configured maximum epochs / selected epoch | 6 / 5 |
| Batch size / gradient accumulation | 4 / 4 |
| Configured base / head learning rate | 0.0003 / 0.0015 |
| Maximum input / target tokens | 384 / 256 |
| Context sentences | 1 |
| Head dropout / weight decay | 0.15 / 0.01 |
| Warmup fraction / early-stopping patience | 0.1 / 3 |
| Class weighting | Enabled; power 0.5, maximum 4.0 |
| Gradient checkpointing | True |
| Selected transition strength / prior correction | 0.1 / 0.25 |
| IA³ target modules | 18 |

Original Phase 1 architecture configuration:

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

The source of truth is [run_config.json](run_config.json). Parameter counts, selected blocks, and exact adapter targets are in [model_metadata.json](model_metadata.json). Decode settings are in [decoder_config.json](decoder_config.json). Adapter and KD defaults in the common configuration apply only when that method uses them.

## Development results

Scores use the 0–1 scale. The combined score pools CL and IT sentences and then averages across all 13 labels, including zero-support labels. It is not the average of the two domain scores.

| Scope | Macro-F1 | Macro-precision | Macro-recall |
| --- | ---: | ---: | ---: |
| Combined dev | 0.27118 | 0.29465 | 0.27355 |
| CL dev | 0.22775 | 0.23490 | 0.24464 |
| IT dev | 0.27305 | 0.29651 | 0.27159 |

The combined development set contains 2,142 sentences. At the saved checkpoint, greedy macro-F1 is 0.26125 and selected-decoder macro-F1 is 0.27118. This compares decoding at one checkpoint, not two independently trained models. Development labels were used for checkpoint and decoder selection; these are selection scores. Codabench score, submission ID, and acceptance have not yet been recorded.

Full scores: [metrics.json](metrics.json). Per-label results: [per_label_metrics.csv](per_label_metrics.csv). Selection evidence: [best_dev_metrics.json](best_dev_metrics.json), [best_selection_metrics.json](best_selection_metrics.json), and [decoding_candidates.json](decoding_candidates.json).

## Plots

![Saved training curves](training_curves.png)

![Development confusion matrix](confusion_matrix.png)

## Verification and artifacts

The saved [verification.json](verification.json) reports training complete and passing checkpoint-reload, development JSON round-trip, label vocabulary, test ID/order, submission ZIP round-trip, and train/dev/test separation checks. These saved checks do not establish Codabench acceptance.

Recorded test output coverage: 10 documents and 2,197 sentences. Checkpoints are expected at `MyDrive/cs781_phase3/encoder_ia3_v1/best_model.pt` and `MyDrive/cs781_phase3/encoder_ia3_v1/last.pt`, following the original run-folder layout. These Drive paths have not been independently checked here and do not grant access. Predictions and submission ZIPs remain in Drive.

Training history: [history.json](history.json). Class weights: [loss_class_weights.json](loss_class_weights.json).

[All Phase 3 results](../) · [All results](../../)
