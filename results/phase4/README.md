# Phase 4 results

Court judgment prediction. The [implementation and instructions](../../Phase4/README.md) are prepared, but saved run artifacts have not yet been imported. No result rows or completed runs are claimed here.

Import each run’s `run_config.json`, `model_metadata.json`, `metrics.json`, `best_dev_metrics.json`, `threshold.json`, `verification.json`, `chunk_coverage.json`, `class_weights.json`, `history.json`, and available plots. Phase 5 also produces `per_label_metrics.csv` and `missed_statutes.png`. An ensemble should retain its separate folder and `ensemble_recipe.json` rather than be counted as another adaptation method.

Preserve original experiment names and run tags. Add a run README and an [experiment ledger](../../experiments/runs.csv) entry from the actual saved configuration and metrics. Do not infer past run settings from current notebook defaults. Model weights, datasets, predictions, probabilities, caches, and submission archives remain in Drive.

Phase 4 uses macro-F1 over outcome classes 0 and 1, with the selected probability threshold documented. Development labels select checkpoints and thresholds. Codabench scores and acceptance must be recorded independently.

[All results](../)
