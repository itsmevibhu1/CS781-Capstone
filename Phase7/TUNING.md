All editable settings are in **Section 3**. Start with `EXPERIMENT='encoder_full'`, `RUN_ALL=False`.
Run all cells in a new Colab runtime. In a prepared runtime, changing only the experiment means rerunning
Section 3 and Sections 10–11. Rebuild Sections 4–5 if changing preprocessing, self exclusion, lexical
settings, negative depth or the data files. Fresh tags preserve comparisons; identical tags/settings resume.

| Goal | Section 3 control |
|---|---|
| One profile | `EXPERIMENT`, `RUN_ALL=False` |
| Six core profiles | `RUN_ALL=True`; completed matching runs are reused |
| Eight extra profiles | `RUN_EXTRA_APPROACHES=True` |
| Automatic training experiments | `RUN_TUNING=True`, edit `TUNING_TRIALS`, choose `TUNE_BASE_EXPERIMENT` |
| Manual learning rate | `LEARNING_RATE`; None uses each profile's LR |
| Effective query batch | `QUERY_BATCH_SIZE * GRAD_ACCUM_STEPS` |
| Positives/negatives | `POSITIVES_PER_QUERY`, `NEGATIVES_PER_QUERY` |
| BM25 hard negative fraction | `HARD_NEGATIVE_FRACTION`, `HARD_NEGATIVE_DEPTH` |
| In-batch negatives | `USE_IN_BATCH_NEGATIVES`; other queries' positives are checked against each query's complete gold set |
| Dynamic neural hard negatives | `MINE_DENSE_NEGATIVES=True`; additional train-only encoding each epoch |
| Training objective | `LOSS_NAME`: multi_positive/pairwise/combined |
| Score sharpness | `TEMPERATURE`, `PAIRWISE_MARGIN` |
| Document context | `CHUNK_LENGTH`, `CHUNK_OVERLAP`, `MAX_CHUNKS` |
| Chunk selection | `CHUNK_POLICY`: spread/head/tail/head_tail |
| Pooling | `POOLING`: attention/mean/max/mean_max |
| Query/candidate representation | `SHARE_QUERY_CANDIDATE_PROJECTION` |
| Embedding/head | `EMBEDDING_DIM`, `HEAD_DROPOUT`, `HEAD_LR_MULTIPLIER` |
| GPU chunk microbatch | `CHUNK_BATCH_SIZE` |
| Inference document batch | `ENCODE_DOCUMENT_BATCH_SIZE` |
| Adaptation capacity | `RANK`, `ALPHA`, `PROMPT_TOKENS`, `PARTIAL_LAYERS` |
| Distillation | `KD_WEIGHT`, `KD_TEMPERATURE`, `TEACHER_RUN_DIR` |
| BM25 | `BM25_K1`, `BM25_B`, `BM25_MAX_FEATURES`, `BM25_STOP_WORDS` |
| Neural/lexical fusion | `DENSE_WEIGHTS`, `RRF_K`, `RRF_DEPTH`; chosen on dev |
| Chunk late interaction | `SEARCH_LATE_INTERACTION=True`; compares max-per-query-chunk candidate similarity |
| Provided-text citation feature | `ENABLE_CITATION_FEATURE=True`, `CITATION_WEIGHTS`; no external query/API |
| Self candidates | `EXCLUDE_SELF_MATCH`; default false follows the specified candidate pool without adding a rule |
| Duration | `EPOCHS`, `EARLY_STOPPING_PATIENCE` |

**Core profiles:** `encoder_full`, `decoder_partial`, `hybrid_lorma`, `lightweight_prompt`,
`lightweight_ia3`, `lightweight_kd`. Full, partial, LoRMA, prompt and IA³ are genuinely applied to the
backbone; every run also trains its retrieval projection/pooling head. KD removes alternating lightweight
layers from the Phase 1 student and trains it using supervised relevance plus a larger frozen teacher's
similarity distribution on the same train query/candidate batches. It does not use a classifier teacher
from an earlier phase or an externally pretrained embedding model.

**Extra profiles:** `decoder_full`, `encoder_partial`, `encoder_lorma`, `encoder_ia3`, `decoder_ia3`,
`hybrid_full`, `lightweight_full`, `lightweight_lorma`. Hyperparameter variants, pooling changes and
fusion changes do not count as new adaptation techniques. Six core profiles also cover the PDF's
stricter general six-submission instruction, despite §6.5/§13 specifying at least four. Distinct accepted
submissions must still be checked on Codabench; identical prediction contents do not establish six submissions.

For parallel Colab sessions, use separate notebook copies and distinct experiments, set `RUN_ALL=False`,
`RUN_EXTRA_APPROACHES=False`, `RUN_TUNING=False`, and prepare shared token caches first. Do not run the
same experiment/tag concurrently. Run KD after a completed larger Phase 7 teacher exists. The teacher
is chosen by dense-only dev Micro-F1 among eligible larger models, not by how strongly BM25 rescues
its final fusion score. Pin `TEACHER_RUN_DIR` if desired; resumed students retain their saved teacher.

The twelve automatic trials really train their overrides in separate `<base>__<trial>_<tag>` folders.
They are independent one-factor changes, not cumulative. Shorten the list according to measured runtime.
Keep official split identities fixed. No training queries use dev/test relevance labels; test labels are
ignored. All ranking components search the complete corresponding candidate pool. Chunk selection
is sampled context, not complete document reading; inspect `chunk_coverage.json`.

The multi-positive loss distributes target probability uniformly across every known positive present
in that sampled candidate batch. Hard negatives exclude the complete gold set; in-batch negatives are
also checked against it. This avoids teaching a known positive as negative merely because it belongs to
another query's sampled set. Pairwise loss compares positives to the hardest sampled negative. Queries
with no labeled positives remain in preprocessing/audits but cannot form positive-supervised train batches.

Default validation compares dense cosine retrieval, optional chunk late interaction, and a bounded
nonzero-neural/BM25 reciprocal-rank fusion search. The standalone BM25 baseline is separately saved;
pure BM25 is not selected as an allegedly adapted neural experiment. If it beats every learned run,
report that result and investigate training/representations. Ties in checkpoint selection use dense-only
dev F1, so a flat hybrid score does not hide improvements in the learned representation.

**Metrics:** Micro-F1@7 = `2*total_hits/(7*query_count + complete_gold_count)`.
Precision = `total_hits/(7*query_count)`; recall = `total_hits/complete_gold_count`.
Gold sets of fewer/more than seven are neither padded nor truncated. Empty dev gold sets still contribute
seven predictions to the precision/F1 denominator. The oracle score reports the maximum possible F1
under the exact-seven rule. Keep `TOP_K=7`; changing it violates the submission contract.

Numeric-only gold IDs resolve to zero-padded candidate IDs only when the exact ID is missing and the
numeric match is unambiguous. Original pool strings are always written. Test predictions are objects,
not the Phase 6 summary list format. Every test query must have seven unique strings from test_candidates.
Duplicate IDs with conflicting case contents fail instead of being overwritten. Empty text is audited and
encoded with UNK. A pool with no usable text stops for schema inspection.

Outputs: `MyDrive/cs781_phase7/<experiment>_<RUN_TAG>/`. Keep best/last checkpoints for inference/resume,
dev/test embeddings for ranking analysis, full configs/metrics, error reports and submission ZIPs.
The root CSV and `best_per_architecture.json` compare complete compatible runs. `bm25_baseline/` is an
extra diagnostic submission, not a required adaptation method. No private test score is fabricated and
the code does not upload to Codabench. Inspect missed relevant cases, query length/coverage and lexical
versus neural rankings before drawing legal-relevance conclusions. Repeated dev tuning may overfit.
Keep the required LLM declaration and references in your report.
