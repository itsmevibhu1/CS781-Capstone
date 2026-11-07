# Phase 7 validation — 4 November 2026

CPU verification passed using PyTorch 2.15.0+cpu, tokenizers 0.22.2, numpy 2.5.4, scipy 1.18.1 and scikit-learn 1.9.1. Synthetic compatible Phase 1 checkpoints and all generated artifacts are stored under `/tmp`. Original local weights were removed during user cleanup; unavailable real weights were not claimed to be retested.

Verified:

- nbformat schema and Python compilation of every non-magic notebook cell; all six documented dataset paths.
- Public repository metadata independently lists those six Phase 7 files. Content requests without authentication returned HTTP 401; actual file contents remain a Colab/HF-token runtime check.
- Actual six-file loading path with query lists, candidate sentence lists, wrapped dev exports and JSONL test candidates.
- Original query/candidate strings, leading zeros, numeric-gold-to-pool conversion only when unambiguous, split-specific pools and test relevance isolation.
- Token caches and source-length statistics; no-positive train queries audited while empty-gold dev queries remain in the metric denominator.
- Independent set counting for Micro-F1@7, precision and recall, including gold sets larger than seven and empty gold sets.
- Sparse BM25 scores agree with the explicit per-term formula. Cached lexical ranks remain aligned with pool IDs.
- LoRMA, IA³ output/input/fused-QKV identity initialization and trainable gradients.
- All six core and eight optional profiles strictly load compatible original architecture states, produce finite gradients and preserve case-batch padding invariance and normalized embeddings.
- Prompt gradients through a frozen checkpointed backbone; all pooling choices and asymmetric projection heads.
- Multi-positive batch masks mark all known positives; multi-positive, pairwise and combined losses are finite and masked scores receive zero gradient. KD masks produce finite KL.
- Dense ranking/tie order, pure-neural policy ordering, chunk late interaction against independent per-candidate computations and original-ID citation matching.
- Complete synthetic core training, dev policy selection, larger frozen teacher to smaller student, checkpoint reload rank equality and full metric equality.
- Completed-run reuse; exact interrupted/resumed final weights; train-only dense negative mining persisted across epochs; citation-enabled ranking policy path.
- Export JSON round trips and strict rejection of missing queries, duplicate/short lists, foreign-pool IDs and integer-valued prediction IDs; ZIP root/layout and JSON contents.
- Experiment comparison, all-four-origin/method coverage and best run per architecture.

Limits: no actual competition score, gated-data schema guarantee, Colab CUDA result or Codabench acceptance is asserted. Synthetic test scores are implementation checks, not retrieval quality. Official tests must use the real Drive Phase 1 weights and authorized dataset download. The notebook does not fabricate a private test score.

Reproduce with `python Phase7/validate_phase7_notebook.py` in the dependencies listed in README.
