# Phase 3 validation

The generated notebook passed offline checks on 14 October 2026. Checks use the real tokenizer and saved Phase 1 checkpoints in this workspace, plus small synthetic CL/IT train/dev/test files. Temporary artifacts are written under `/tmp`; original Phase 1 and Phase 2 artifacts remain unchanged.

Passed:

- Notebook schema validation and compilation of every Python cell.
- All seven-file download/schema/split-audit paths, using an offline provider with the official filename layout.
- Label vocabulary list, label-to-ID, ID-to-label, and nested mapping formats; invalid sentence-label counts are rejected.
- Context budgeting, target masks, long-sentence head/tail truncation, and empty-sentence handling.
- Exact macro-F1/precision/recall accounting from sentence confusion matrices, including per-domain reports.
- LoRA, rsLoRA and LoRMA+ identity initialization, gradients and explicit merged-weight formulas.
- IA³ identity initialization and nonzero scaling-vector gradients.
- Train-fitted sequence decoding agrees with an independent exhaustive path search on a small sequence.
- Strict loading and finite forward/backward passes for all six experiment options with actual local Phase 1 weights.
- Right padding does not change valid sentence classification outputs in those tests.
- Synthetic supervised training, class weights, dev checkpoint/decoder selection, reports, confusion/training plots, JSON and ZIP verification.
- Knowledge distillation uses a completed larger teacher and a smaller Phase 1-initialized student.
- Completed runs reuse training state; early stopping remains completed after resumption.
- A mid-epoch interruption resumes correctly, including a short final accumulation group, and matches uninterrupted final weights exactly.
- Sentence predictions are reconstructed in document order even when inference examples are deliberately reversed.
- The coverage report detects missing methods/origins and counts distinct prediction contents rather than assuming six files are distinct.

One resume issue was found and fixed: inference DataLoader creation initially consumed global Torch RNG. Prediction now uses a dedicated generator so validation does not change subsequent training dropout after an interruption.

Limits: the official gated dataset was not authenticated/downloaded here. Actual official file-schema compatibility, Colab GPU execution, task scores, and Codabench acceptance require your runtime. Synthetic metrics are correctness checks, not measured capstone performance. Aggregation is sentence-level macro over all thirteen labels, with domain reports also saved; parity with a separately supplied official evaluation script is not claimed.

Repeat with `python Phase3/validate_phase3_notebook.py` in an environment containing PyTorch, tokenizers, numpy, tqdm, matplotlib, nbformat and huggingface_hub.
