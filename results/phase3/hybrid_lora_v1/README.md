# Phase 3: hybrid_lora_v1

Rhetorical role prediction for CL and IT documents using LoRA with the Phase 1 hybrid prefix-lm origin. This page records the exported configuration and metrics; training was not rerun for documentation.

## Method and implementation

The backbone weights are frozen; additive low-rank updates are trained on the attention projections listed in model_metadata.json, together with the classification head.

The current [Phase 3 notebook](../../../Phase3/CS781_Phase3_Solution_Colab.ipynb) contains data loading, BPE tokenization, context-aware sentence inputs, classification, training, and prediction. See the [phase instructions](../../../Phase3/README.md). All downstream runs initialize from Phase 1 rather than Phase 2 weights.

The original training code commit was not recorded in these artifacts. The linked notebook is the current repository version; its exact correspondence with the historical training code still needs confirmation.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / run tag | `hybrid_lora` / `v1` |
| Phase 1 origin | Hybrid Prefix-LM |
| Adaptation | LoRA |
| Original Phase 1 parameters | 99,139,200 |
| Task model total / trainable parameters | 100,354,029 / 1,214,829 |
| Trainable fraction | 1.21% |
| Seed | 42 |
| Configured maximum epochs / selected epoch | 6 / 2 |
| Batch size / gradient accumulation | 4 / 4 |
| Configured base / head learning rate | 0.0002 / 0.001 |
| Maximum input / target tokens | 384 / 256 |
| Context sentences | 1 |
| Head dropout / weight decay | 0.15 / 0.01 |
| Warmup fraction / early-stopping patience | 0.1 / 3 |
| Class weighting | Enabled; power 0.5, maximum 4.0 |
| Gradient checkpointing | True |
| Selected transition strength / prior correction | 0.6 / 0.25 |
| Adapter rank / alpha | 8 / 16.0 |
| Adapter target modules | 24 |

Original Phase 1 architecture configuration:

```json
{
  "vocab_size": 32000,
  "n_layer": 12,
  "d_model": 640,
  "n_head": 10,
  "d_ff": 2560,
  "max_len": 1152
}
```

The source of truth is [run_config.json](run_config.json). Parameter counts, selected blocks, and exact adapter targets are in [model_metadata.json](model_metadata.json). Decode settings are in [decoder_config.json](decoder_config.json). Adapter and KD defaults in the common configuration apply only when that method uses them.

## Development results

Scores use the 0–1 scale. The combined score pools CL and IT sentences and then averages across all 13 labels, including zero-support labels. It is not the average of the two domain scores.

| Scope | Macro-F1 | Macro-precision | Macro-recall |
| --- | ---: | ---: | ---: |
| Combined dev | 0.34177 | 0.37585 | 0.33870 |
| CL dev | 0.29273 | 0.33088 | 0.30247 |
| IT dev | 0.34163 | 0.39294 | 0.33516 |

The combined development set contains 2,142 sentences. At the saved checkpoint, greedy macro-F1 is 0.31485 and selected-decoder macro-F1 is 0.34177. This compares decoding at one checkpoint, not two independently trained models. Development labels were used for checkpoint and decoder selection; these are selection scores. Codabench score, submission ID, and acceptance have not yet been recorded.

Full scores: [metrics.json](metrics.json). Per-label results: [per_label_metrics.csv](per_label_metrics.csv). Selection evidence: [best_dev_metrics.json](best_dev_metrics.json), [best_selection_metrics.json](best_selection_metrics.json), and [decoding_candidates.json](decoding_candidates.json).

## Plots

![Saved training curves](training_curves.png)

![Development confusion matrix](confusion_matrix.png)

## Verification and artifacts

The saved [verification.json](verification.json) reports training complete and passing checkpoint-reload, development JSON round-trip, label vocabulary, test ID/order, submission ZIP round-trip, and train/dev/test separation checks. These saved checks do not establish Codabench acceptance.

Recorded test output coverage: 10 documents and 2,197 sentences. Checkpoint files are excluded from Git and have not been independently checked here. Predictions and submission ZIPs are excluded from Git.

Training history: [history.json](history.json). Class weights: [loss_class_weights.json](loss_class_weights.json).

[All Phase 3 results](../) · [All results](../../)
