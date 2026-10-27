# Phase 5 validation — 25 October 2026

Validation uses the actual local Phase 1 BPE and all four saved Phase 1 architectures. Complete fitting/calibration/prediction pipelines use synthetic train/dev/test documents; artifacts stay under `/tmp`. Official gated data and private ground truth are not available to these tests.

Checked:

- Standalone notebook schema, compilation of every non-magic Python cell and official Phase 5 split filenames.
- Real three-file loader with JSON train, CSV dev and JSONL test; safe Python literals and UTF-16 are also exercised. Invalid/truncated/HTML/blank data are rejected.
- Multi-label integer set parsing, deduplication, sparse statute IDs, paragraph text normalization, original integer/string case IDs, repeated train/dev instances and explicit isolation of test labels.
- Exact-text overlaps are exercised in the complete synthetic loader: two training copies of dev/test cases are excluded, preserving all held-out IDs and recomputing train counts. Separate cases cover repeated/conflicting-label training copies, documents shared by all three splits, blank-text exclusion from matching, nonmutating filtering, report-only/error policies and an empty-training-set guard.
- Disk-backed token caches, empty test document representation, deterministic chunk policies and full sliding-window coverage arithmetic.
- Independent set-based calculations of per-statute Macro-F1, aggregate Micro-F1 and exact-set Subset Accuracy across random synthetic examples. Known/reference-present/reference-range conventions and ID gaps are checked explicitly.
- Exact single-statute positive-class F1 threshold selection against brute force, bounded global calibration and safe baseline retention.
- Strict loading of real Phase 1 checkpoints, finite multi-label forward/backward passes, classifier gradients and case-batch padding invariance for all six core profiles.
- LoRMA identity initialization and adapter gradients; optional adaptation profiles, all five hierarchical pooling paths, train-only label graph and BCE/focal/asymmetric finite losses/gradients.
- Complete synthetic core training with class/positive weighting, label-prior initialization, short final gradient-accumulation groups and automatic dev checkpoint selection.
- Reloaded best-checkpoint metric equality, independent dev JSON score round trips, rare-statute/error reports, probability shape/ordering validation and submission JSON/ZIP verification.
- Smaller student's larger completed teacher, different student/teacher tags and chunk budgets, train-only teacher cache, and exactly matching student weights for cold/warm teacher caches.
- Completed-run resumption, exact interrupted/resumed final weights, gradual unfreezing and the optimizer's future-layer parameters.
- Saved-probability ensembles, output label-order mismatch rejection, malformed statute output rejection, experiment/method coverage and best-run selection per Phase 1 architecture.

Limits: no claim is made about actual official dev/test scores, Colab GPU execution, competition acceptance, or parity with unseen evaluator code. The notebook reports multiple label-dimension conventions because the PDF's scorer description is incomplete. Synthetic scores are implementation checks, not capstone performance.

Reproduce with `python Phase5/validate_phase5_notebook.py` using torch, numpy, tokenizers, tqdm, matplotlib, nbformat and huggingface_hub.
