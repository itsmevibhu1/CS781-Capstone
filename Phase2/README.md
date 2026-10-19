# Phase 2 notebook

Open [CS781_Phase2_Solution_Colab.ipynb](CS781_Phase2_Solution_Colab.ipynb) in Google Colab, select a GPU runtime, and add a Colab secret named **HF** with access to the course dataset.

The notebook mounts Drive. It reads the existing tokenizer and model weights from `MyDrive/cs781_phase1/` and writes all outputs to `MyDrive/cs781_phase2/`.

Run the notebook once per experiment, changing `EXPERIMENT` in section 3:

| Order | EXPERIMENT | Phase 1 origin | Adaptation |
|---|---|---|---|
| 1 | `decoder_full` | Decoder-only | Full fine-tuning |
| 2 | `encoder_partial` | Encoder–decoder encoder | Last two encoder layers and final norm |
| 3 | `hybrid_lora` | Hybrid Prefix-LM | LoRA attention projections |
| 4 | `lightweight_lorma` | Lightweight | LoRMA+ attention projections |
| 5 | `decoder_bitfit` | Decoder-only | Backbone biases and classification head |
| 6 | `lightweight_kd` | Lightweight, half its blocks | Supervised loss plus trained teacher distillation |

Alternatively, `RUN_ALL=True` runs them sequentially. The KD run automatically selects a completed larger Phase 2 teacher by held-out strict macro-F1. Each run saves `prediction.json` and `submission.zip` in its own folder. Upload each ZIP to Codabench yourself.

After a runtime interruption, rerun with the same experiment, `RUN_TAG`, and settings to restore the latest saved training state. Use a new `RUN_TAG` when changing settings. Save intervals refer to optimizer steps; unsaved work after the latest interval may repeat.

Section **12, Post-training validation and submission verification**, displays the document's required scores after the best checkpoint is reloaded:

- Strict Macro-F1 (primary).
- Strict Macro-Precision and Strict Macro-Recall (secondary).
- Per-label precision, recall, F1, and TP/FP/FN/support counts.

The verification checks checkpoint score reproducibility, all validation/test IDs, character offset bounds, entity labels, duplicate spans, JSON round-trips, and the ZIP root format. Reports are saved as `post_training_validation.json` and `verification.json`.

The revised defaults use `RUN_TAG='v2'`, preserving any earlier `v1` results. Training uses bounded class weights computed exclusively from training entities and a higher learning rate for the new classification head. After training, a small search compares greedy decoding with BIO-constrained Viterbi decoding and entity score biases. `decoding_candidates.json` records every candidate, and `decoder_config.json` records the selection used on the official test set. It retains greedy decoding when no candidate improves local F1.

Validation is used for checkpoint and decoder selection, so this report is a selection score rather than an unbiased estimate of private test performance. Actual private test metrics come from Codabench; no private test score is invented here.

The notebook preserves original document strings, uses the Phase 1 BPE character offsets, and merges overlapping windows before decoding. Its BIO head models flat entities. Nested spans or separate entities sharing a BPE token raise an explicit alignment error. Gold boundaries inside a BPE token are reported in `alignment_report.json`; this tokenizer limits their exact recoverability. It does not silently claim complete character-boundary coverage.

Dimensions are inferred from your actual checkpoints, which were validated locally. `PHASE1_HEADS` must match the original run because attention head counts are not always recoverable from state dictionaries. The defaults match the supplied Phase 1 notebook.

## Validation

`validate_phase2_notebook.py` checks notebook syntax/format, all six adaptations with real local Phase 1 checkpoints, exact offset decoding, LoRMA's multiplicative formula, adapter gradients, strict metrics, and small synthetic training/resume/KD/submission pipelines. It does not authenticate or download the official dataset, or measure actual task performance.

Run it in a Python environment with PyTorch, tokenizers, numpy, tqdm, matplotlib, nbformat, and huggingface_hub:

```bash
python Phase2/validate_phase2_notebook.py
```

Sources are linked in the notebook, including the [LoRMA paper](https://aclanthology.org/2025.findings-acl.527/) (LoRMA+, Eq. 8). Keep the course-required LLM declaration with your deliverables.

## Saved run results

[Browse all 14 Phase 2 runs](../results/phase2/) for saved configurations, strict held-out validation scores, per-label tables, plots, and verification availability. Older runs lack some newer reports. Both partial-encoder runs record zero F1; all run pages preserve measured results, including unsuccessful experiments.

The `v1` and `v3` groups each contain six experiment types, while `v2` and `v4` contain decoder-only variants. Local coverage reports do not establish accepted submissions. Historical code commits and Codabench records remain to be supplied.
