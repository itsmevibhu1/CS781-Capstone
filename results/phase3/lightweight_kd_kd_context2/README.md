# Phase 3: lightweight_kd_kd_context2

Rhetorical role prediction for CL and IT documents using Knowledge distillation with the Phase 1 lightweight origin. This page records the exported configuration and metrics; training was not rerun for documentation.

## Method and implementation

A smaller student uses the recorded subset of Phase 1 lightweight blocks. Its backbone and classifier are trainable using supervised labels and teacher-logit distillation. The recorded teacher identifier is shown below; the historical teacher checkpoint hash was not included in the exported artifacts.

The current [Phase 3 notebook](../../../Phase3/CS781_Phase3_Solution_Colab.ipynb) contains data loading, BPE tokenization, context-aware sentence inputs, classification, training, and prediction. See the [phase instructions](../../../Phase3/README.md). All downstream runs initialize from Phase 1 rather than Phase 2 weights.

The original training code commit was not recorded in these artifacts. The linked notebook is the current repository version; its exact correspondence with the historical training code still needs confirmation.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / run tag | `lightweight_kd` / `kd_context2` |
| Phase 1 origin | Lightweight |
| Adaptation | Knowledge distillation |
| Original Phase 1 parameters | 102,451,968 |
| Task model total / trainable parameters | 64,726,381 / 64,726,381 |
| Trainable fraction | 100.00% |
| Seed | 42 |
| Configured maximum epochs / selected epoch | 6 / 5 |
| Batch size / gradient accumulation | 4 / 4 |
| Configured base / head learning rate | 5e-05 / 0.00025 |
| Maximum input / target tokens | 384 / 256 |
| Context sentences | 2 |
| Head dropout / weight decay | 0.15 / 0.01 |
| Warmup fraction / early-stopping patience | 0.1 / 3 |
| Class weighting | Enabled; power 0.5, maximum 4.0 |
| Gradient checkpointing | True |
| Selected transition strength / prior correction | 0.6 / 0.25 |
| Selected Phase 1 block indices (zero based) | `[0, 2, 4, 6, 8, 10]` |
| Teacher experiment identifier | `lightweight_lorma` |
| KD weight / temperature | 0.5 / 2.0 |

Original Phase 1 architecture configuration:

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

The source of truth is [run_config.json](run_config.json). Parameter counts, selected blocks, and exact adapter targets are in [model_metadata.json](model_metadata.json). Decode settings are in [decoder_config.json](decoder_config.json). Adapter and KD defaults in the common configuration apply only when that method uses them.

## Development results

Scores use the 0–1 scale. The combined score pools CL and IT sentences and then averages across all 13 labels, including zero-support labels. It is not the average of the two domain scores.

| Scope | Macro-F1 | Macro-precision | Macro-recall |
| --- | ---: | ---: | ---: |
| Combined dev | 0.44106 | 0.52082 | 0.41772 |
| CL dev | 0.32675 | 0.36044 | 0.34062 |
| IT dev | 0.44510 | 0.52956 | 0.42702 |

The combined development set contains 2,142 sentences. At the saved checkpoint, greedy macro-F1 is 0.41280 and selected-decoder macro-F1 is 0.44106. This compares decoding at one checkpoint, not two independently trained models. Development labels were used for checkpoint and decoder selection; these are selection scores. Codabench score, submission ID, and acceptance have not yet been recorded.

Full scores: [metrics.json](metrics.json). Per-label results: [per_label_metrics.csv](per_label_metrics.csv). Selection evidence: [best_dev_metrics.json](best_dev_metrics.json), [best_selection_metrics.json](best_selection_metrics.json), and [decoding_candidates.json](decoding_candidates.json).

## Plots

![Saved training curves](training_curves.png)

![Development confusion matrix](confusion_matrix.png)

## Verification and artifacts

The saved [verification.json](verification.json) reports training complete and passing checkpoint-reload, development JSON round-trip, label vocabulary, test ID/order, submission ZIP round-trip, and train/dev/test separation checks. These saved checks do not establish Codabench acceptance.

Recorded test output coverage: 10 documents and 2,197 sentences. Checkpoints are expected at `MyDrive/cs781_phase3/lightweight_kd_kd_context2/best_model.pt` and `MyDrive/cs781_phase3/lightweight_kd_kd_context2/last.pt`, following the original run-folder layout. These Drive paths have not been independently checked here and do not grant access. Predictions and submission ZIPs remain in Drive.

Training history: [history.json](history.json). Class weights: [loss_class_weights.json](loss_class_weights.json).

[All Phase 3 results](../) · [All results](../../)
