# Phase 1: run_lightweight

## Available evidence

This folder contains the saved loss plot exported from `run_lightweight/`. The folder name identifies the lightweight transformer run. No run configuration, numeric history, model metadata, verification report, or evaluation score was included in this export. See the [export manifest](../export_manifest.json).

![Saved loss curve](loss_curve.png)

The image is evidence of a saved plot; it does not independently establish training completion, exact hyperparameters, model selection, or Codabench acceptance. No numeric loss or task score has been inferred from the image.

## Implementation and reproduction

The current [Phase 1 notebook](../../../Phase1/CS781_Phase1_Solution_Colab.ipynb) defines this architecture under `ARCHITECTURE = "lightweight"`, along with the data/tokenizer/training/generation pipeline. See the [phase instructions](../../../Phase1/README.md). This is implementation documentation, not proof that current notebook defaults match the historical run.

Preserve the notebook version, configuration cell output, or training logs actually used for this run before recording exact dimensions, token budgets, optimization settings, stopping criteria, and generation settings. The original code commit and checkpoint checksum are not available in the exported evidence.

## Artifact location and remaining records

Weights and prediction/submission files are excluded from Git. The tokenizer pairing with this historical checkpoint still needs confirmation.

## Matched official best submission

Codabench submission **962785** (user-provided results) records **BERTScore F1 0.83094329**, **ROUGE-L F1 0.19069432**, **BLEU-4 0.03978785**, **1,000 examples**, and reported rank **75**. The author confirmed **4 distinct valid submissions**; the rank-recording date remains unconfirmed.

Both this run’s `prediction.json` and `submission.zip` have exactly the same prediction content as the downloaded official submission. The [match report](../best_submission_match.json) found only this run folder among eight compared artifacts, with no errors. These two file matches identify one saved run. No model weights were loaded for this comparison. The [official result record](../official_best.json) preserves fingerprints and the scope of this attribution; historical code/version and checkpoint-weight checksums remain unverified.

[All Phase 1 results](../) · [All results](../../)
