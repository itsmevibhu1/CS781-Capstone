# Phase 4: lightweight_kd_v1

Court judgment prediction with one binary outcome per case. Saved artifacts supply the settings and scores below; training was not rerun for this documentation.

## Method and implementation

The run uses **Knowledge distillation** from the Phase 1 `lightweight` origin. The [current solution notebook](../../../Phase4/CS781_Phase4_Solution_Colab.ipynb) implements chunked document representation, `attention` pooling, and a task head; see the [phase instructions](../../../Phase4/README.md) for method details. The historical training code commit was not recorded, so exact correspondence with the current notebook remains unconfirmed.

The smaller student is initialized from the recorded Phase 1 blocks and trained with supervised targets plus teacher distillation. The teacher run and signature are preserved in [model_metadata.json](model_metadata.json); the referenced checkpoint has not been loaded or independently verified here.

## Saved configuration

| Setting | Value |
| --- | --- |
| Experiment / run tag | `lightweight_kd` / `v1` |
| Phase 1 origin | lightweight |
| Method | Knowledge distillation |
| Phase 1 parameter count | 102,451,968 |
| Task model total / initially trainable parameters | 63,782,027 / 63,782,027 |
| Finally trainable parameters | 63,782,027 |
| Seed | 42 |
| Maximum epochs / best epoch | 6 / 3 |
| Case batch / accumulation | 2 / 4 |
| Base learning rate / head multiplier | 5e-05 / 5.0 |
| Chunk length / overlap / maximum chunks | 384 / 64 / 4 |
| Training / evaluation chunk policy | spread / spread |
| Chunk microbatch size | 4 |
| Document pooling / hidden dimension | attention / 128 |
| Dropout / weight decay | 0.15 / 0.01 |
| Warmup fraction / patience | 0.1 / 3 |
| Phase 1 block indices | `[0, 2, 4, 6, 8, 10]` |
| Recorded teacher run | `/content/drive/MyDrive/cs781_phase4/decoder_full_v1` |
| KD weight / temperature | 0.5 / 2.0 |

Exact settings: [run_config.json](run_config.json). Original architecture, adapter targets, parameter counts, selected blocks, and teacher metadata: [model_metadata.json](model_metadata.json). Class/label weights: [class_weights.json](class_weights.json). Common adapter/KD defaults apply only to methods that use them.

## Development results

Macro-F1 averages outcome classes 0 and 1. Secondary precision/recall below are for class 1.

| Macro-F1 | Class-1 precision | Class-1 recall | Accuracy |
| ---: | ---: | ---: | ---: |
| 0.64027 | 0.66357 | 0.57545 | 0.64185 |

Selected probability threshold: **0.331661**. Prediction rule: class 1 when its probability is at least the threshold.

Development cases: 1,988. At the same saved checkpoint, the 0.5 baseline macro-F1 is 0.59059, versus 0.64027 after threshold selection. This is a within-checkpoint threshold comparison, not a comparison of separately trained models. Dev labels selected checkpoints and thresholds, so these are selection scores. Official test score, Codabench ID, and acceptance remain unrecorded.

Full scores: [metrics.json](metrics.json). Checkpoint-selection evidence: [best_dev_metrics.json](best_dev_metrics.json). Thresholds: [threshold.json](threshold.json).

## Document coverage

Only a sampled portion of long documents may be represented. Saved coverage is not a claim of reading every token.

| Split | Cases | Mean token coverage | Mean chunks |
| --- | ---: | ---: | ---: |
| train | 37,387 | 56.84% | 3.58 |
| dev | 1,988 | 41.70% | 3.94 |
| test | 1,517 | 41.09% | 3.94 |

[Full chunk coverage report](chunk_coverage.json).

## Plots

![Training curves](training_curves.png)

![Development confusion matrix](confusion_matrix.png)

## Verification and artifact locations

[verification.json](verification.json) reports training complete and passing checkpoint-reload, dev JSON round-trip, original test ID/type, output-format, ZIP, and separation checks. These are saved reports, not a fresh official-data evaluation or proof of competition acceptance.

Expected checkpoint: `MyDrive/cs781_phase4/lightweight_kd_v1/best_model.pt`. This path follows the original folder layout and has not been independently checked. Weights, predictions, probability arrays, caches, and submission ZIPs remain in Drive. [history.json](history.json) preserves the numeric training history. No standalone environment report was exported.

[All Phase 4 results](../) · [All results](../../)
