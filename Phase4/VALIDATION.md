# Phase 4 validation — 21 October 2026

Offline validation passed using the real Phase 1 tokenizer and all four saved Phase 1 checkpoints in this workspace. Synthetic cases exercise the full train/dev/test workflow without authenticating to the course dataset. Temporary artifacts are written under `/tmp`; supplied Phase 1/2/3 results are preserved.

Verified:

- Notebook schema and Python compilation for every non-magic code cell.
- Actual three-file download/schema/split-audit code with an offline provider using the official paths.
- File parsing checks cover standard JSON, UTF-8 BOM, UTF-16 BOM, JSON Lines, CSV with quoted multiline Unicode text and Python literal containers. The complete synthetic workflow uses JSON train, CSV dev and JSONL test files. Blank, HTML, LFS-pointer, truncated, partially malformed and executable-looking inputs are rejected. A simulated bad cached file triggers exactly one forced download; repeated failure reports the split/source filename and persists its diagnosis.
- Binary label handling, paragraph-list text normalization, original integer/string IDs, and rejection of invalid labels/predictions.
- Repeated train/dev case IDs preserve all instances, texts and conflicting labels without key collisions. The three-file synthetic pipeline includes repeated IDs in both labeled splits. Instance audits, probability-cache loading, JSON prediction round trips, post-training metrics and ensembles retain each row. Ambiguous duplicate test IDs remain rejected.
- Persistent disk-backed token-cache creation, empty-case handling and case-level data preparation.
- Overlapping-window full coverage, deterministic spread/head/tail/head+tail/random selection, and per-case ownership of chunks.
- Exact threshold optimization agrees with independent brute force across random small datasets, repeated probabilities and single-class edge cases.
- Confusion-matrix precision/recall/macro-F1 formulas agree with independent known counts.
- LoRMA+ identity initialization, gradients, frozen base and explicit nonzero multiplicative weight formula.
- Strict real checkpoint loading, finite forward/backward passes and case-batch padding invariance for all six core methods.
- Optional full/adapter/gradual/linear-probe paths and gradients for all five pooling choices on tiny compatible models.
- Synthetic class-weighted supervised training, short accumulation groups, dev threshold/checkpoint selection, score reports, error reports, plots and JSON/ZIP verification.
- Larger trained teacher selected for a smaller Phase 1-initialized student, including reuse across run tags and differing chunk budgets.
- Teacher-cache reuse preserves the cache file and produces exactly the same student weights as a cold-cache run.
- Mid-epoch interrupted/resumed training matches uninterrupted final model weights exactly.
- Completed-run reuse, gradual-unfreezing parameter growth and early-stopping completion survive resumption.
- Saved-probability ensemble selection, recipe output and final submission validation.
- Coverage reports track distinct prediction contents; synthetic repeated predictions are correctly reported as not satisfying six distinct submissions.

During development, student random state was explicitly preserved around cold teacher reconstruction, and accumulation was weighted by the number of cases rather than just microbatch count. Class weighting uses per-case loss before averaging, so a one-case minibatch does not cancel its class weight.

Limits: the official gated Phase 4 data was not downloaded here. Official file-schema compatibility, Colab CUDA execution, real dev/leaderboard scores and competition acceptance remain runtime checks. Primary local aggregation is macro-F1 over labels 0 and 1; secondary binary and macro precision/recall are both supplied. Parity with an unseen official evaluation implementation is not claimed. Synthetic test scores are not capstone performance.

Reproduce with `python Phase4/validate_phase4_notebook.py` in an environment containing PyTorch, tokenizers, numpy, tqdm, matplotlib, nbformat and huggingface_hub.
