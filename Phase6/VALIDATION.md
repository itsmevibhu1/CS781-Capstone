# Phase 6 validation — 30 October 2026

All non-magic notebook cells compile and the notebook validates against nbformat. CPU tests use PyTorch 2.15.0+cpu, tokenizers 0.22.2, transformers 4.57.6, bert-score 0.3.13 and rouge-score 0.1.2.

Original Phase 1 weights are no longer stored locally following the user's cleanup. Tests create small checkpoints from the original four architecture definitions and load them through the same state-shape inference/strict-loading code used in Colab. This validates checkpoint structure and computations but does not claim to revalidate unavailable real weight files.

Passed checks:

- Exactly two official dataset paths: train and test; no invented public dev request.
- Input schema, original integer/string IDs, test-reference isolation, duplicate-source grouping, fixed held-out partition and exclusion of exact public/test source copies.
- Regression for ID 299 with a missing training summary in the complete loader. Missing, blank, empty-list and unsupported-field cases are audited/excluded without targets being invented; valid ordered paragraph lists are joined. Strict-policy and excessive-invalid-rate guards stop; malformed test documents are never silently skipped and valid empty test text retains its original ID.
- BPE special IDs, persistent source/reference token stores and empty test input handling.
- All six core and ten optional profiles load compatible Phase 1 states, produce finite target losses/gradients and preserve case-batch padding invariance.
- Wrapper logits match the four original Phase 1 forward computations before adaptation updates. The native encoder–decoder retains and uses its decoder/cross-attention/head.
- Self-attention and native cross-attention cached decoding agree with full-sequence decoding at each checked target position, including soft prompts and hybrid source-prefix masking.
- LoRA/LoRMA identity initialization, frozen projection bases and nonzero update gradients; adapter and prompt gradients.
- Native independently chunked encoder backward passes and all source selection policies.
- KD ignores padded targets; a larger frozen trained teacher transfers distributions to a smaller Phase 1-initialized student on training references only.
- Real BERTScore API and ROUGE calculations, using a tiny local random BERT metric model offline. Identical summaries score 1 on ROUGE and approximately 1 on BERTScore. Metric construction preserves training RNG state.
- Six complete synthetic train/select/reload/evaluate/test/export runs, token-weighted accumulation, best-checkpoint summary equality, metric equality and JSON metric round trips.
- Reusing a verified completed run, interrupted/resumed exact final weights, greedy/beam generation, map–reduce inference and bounded decoding search.
- Test summary JSON schema, nonempty strings, original ID preservation, document counts and root-only JSON ZIP validation.
- Compatible-run summaries, all-four-origin coverage, compulsory LoRMA/KD coverage and best run per architecture.

Limits: these are implementation checks, not measured competition performance. Official gated data were not downloaded here; real Phase 1 weights, full RoBERTa-large evaluation and CUDA execution must be checked in Colab. Exact parity with an unpublished competition scorer is not claimed. Synthetic generated summaries/scores do not establish summarization quality or factuality.

Reproduce with `python Phase6/validate_phase6_notebook.py`. Temporary test checkpoints, caches and artifacts are stored under `/tmp`.
