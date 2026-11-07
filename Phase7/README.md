# Phase 7 — Prior Case Retrieval

Open [CS781_Phase7_Solution_Colab.ipynb](CS781_Phase7_Solution_Colab.ipynb) in Colab, select a GPU, create an **HF** secret containing your plain `hf_...` token, and run all cells. The notebook is standalone; uploading the supporting Python files is unnecessary.

This follows §13 of the capstone PDF and the professor's starter sequence: authentication/configuration → official data → model/adaptation → training/dev scores → saved weights/predictions → verified submission ZIP. It uses your four Phase 1 backbones and BPE. No external pretrained embedding/retrieval model, external LLM or API supplies embeddings, targets or rankings. The encoder–decoder origin uses its encoder for case representation.

## Required inputs and all six official files

Phase 1 inputs are the four `MyDrive/cs781_phase1/run_<architecture>/final_model.pt` files and `bpe_tokenizer/vocab.json` / `merges.txt`. Local copies of these weights are not required by Colab.

The six competition paths are:

```text
Phase_7_PCR/Train/train.json
Phase_7_PCR/Train/train_candidates.json
Phase_7_PCR/Dev/dev.json
Phase_7_PCR/Dev/dev_candidates.json
Phase_7_PCR/Test/test.json
Phase_7_PCR/Test/test_candidates.json
```

Each query searches **only its own split's candidate pool**. Train relevance supervises optimization; dev relevance selects checkpoints/ranking policies; any public test relevance field is ignored. Every candidate is available to full-pool ranking; gold candidates are not secretly inserted into an evaluation shortlist.

Case text can be a string, sentence strings, ordered text objects or a sentence mapping. Original ID strings and leading zeros are preserved. When gold IDs are integers but pool IDs are zero-padded strings, only an unambiguous numeric match is accepted. Missing/ambiguous gold IDs and conflicting duplicate IDs stop with an error instead of being overwritten. Empty texts are audited and receive UNK; all-empty query or candidate sets stop for schema inspection. A candidate pool must have at least seven usable IDs.

The token caches cover all queries/candidates in every split. They are memory-mapped int32 files persisted in Drive and staged to Colab local disk. `dataset_manifest.json` records counts, empty-text counts, gold-ID conversions and train queries without positive supervision. Such train queries cannot form positive-supervised batches, but remain in preprocessing/baseline output. Empty-gold dev queries remain in evaluation. `document_lengths.json` records token-length percentiles.

## First run

In **Section 3**:

```python
EXPERIMENT = 'encoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
RUN_TAG = 'v1'
```

After the first verified run, set `RUN_ALL=True` to train all six core profiles. The matching completed first run is checked and reused without retraining or repeated encoding. In the same prepared runtime, rerun Section 3 and Sections 10–11; in a new runtime, run all cells.

| Exact profile | Phase 1 origin | Adaptation |
|---|---|---|
| `encoder_full` | Encoder–decoder encoder | Full encoder fine-tuning |
| `decoder_partial` | Decoder-only | Last layers and final normalization |
| `hybrid_lorma` | Hybrid Prefix-LM | Compulsory LoRMA+ attention updates |
| `lightweight_prompt` | Lightweight | Continuous input prompt, frozen backbone |
| `lightweight_ia3` | Lightweight | Key/value and FF-intermediate rescaling |
| `lightweight_kd` | Smaller lightweight student | Relevance supervision plus larger retrieval teacher's distributions |

These are actual training/adaptation procedures, not just differently named decoders. Every run also trains its projection/pooling head. KD runs last so a larger trained teacher is available. The student keeps alternating Phase 1 lightweight layers, rather than importing another model.

The core list covers all four Phase 1 origins, at least three non-KD adaptation methods, at least four listed techniques and compulsory LoRMA/KD. The PDF's opening instructions require six submissions while §6.5/§13 specify at least four; six core runs cover the stricter count. Accepted and distinct submissions must still be verified on Codabench. Identical prediction contents are flagged and do not establish six submissions.

For parallel Colab sessions choose distinct profiles in separate notebook copies, with sweep flags false. Prepare shared token caches first and never run the same experiment/tag concurrently. Keep the same tag for unchanged profiles; change the tag for manual settings changes. Run KD after a larger completed Phase 7 teacher exists. Phase 6 summarizers and Phase 5 classifiers are not compatible retrieval teachers.

## Retrieval and score optimization

The neural retriever encodes overlapping source chunks, uses mean+final token features, projects them into a shared normalized space and combines chunks by attention, mean, max or mean+max. Query/candidate projections can be shared or asymmetric. The default reads at most three spread windows; it is sampled context, not a claim of complete document coverage. `chunk_coverage.json` reports actual dev/test coverage.

The default loss distributes target probability uniformly across known positives present in each candidate batch. Positives can be sampled from the full relevance set. BM25 hard negatives and random negatives exclude **every** known positive, not only the currently chosen positive. Other queries' candidate positives are checked again for each query before becoming in-batch negatives. Optional hardest-negative pairwise/combined losses and train-only dense mining broaden comparisons. Mining state is saved for deterministic resume.

BM25 uses sparse count matrices, nonnegative log1p IDF, document length normalization and presence-only query terms. Its vocabulary and weights are constructed independently for each split from that candidate pool. No relevance labels enter BM25 fitting. A standalone baseline is saved under `bm25_baseline/`; it does not count as a Phase 1 adaptation method. Cached top lexical ranks avoid recomputing BM25 at every epoch; a larger fusion depth rebuilds the necessary rank tail.

Dev selection compares dense cosine ranks with optional chunk late interaction and bounded reciprocal-rank fusion against BM25. Late interaction averages each query chunk's maximum candidate-chunk similarity. Full-pool embeddings are searched; fusion uses an explicit rank depth. A trained-model submission always retains a nonzero neural contribution. Pure BM25 is reported separately so lexical results are not mislabeled as neural adaptation. If the baseline wins, report it and investigate the learned representation rather than assuming training helped.

Optional exact IndianKanoon URL signals inspect only the provided query text and are disabled by default. No case website is queried. Their coefficient is dev-selected with a zero-signal fallback. The default does not exclude self IDs because the PDF does not add that rule; `EXCLUDE_SELF_MATCH` is available and validates gold/pool feasibility.

Eight optional profiles: `decoder_full`, `encoder_partial`, `encoder_lorma`, `encoder_ia3`, `decoder_ia3`, `hybrid_full`, `lightweight_full`, `lightweight_lorma`. Select one or use `RUN_EXTRA_APPROACHES=True`.

Twelve bounded training trials compare LR, chunk budgets/lengths, pooling, source selection, ranking losses, negatives, asymmetric heads, dense mining and temperature. Set `RUN_TUNING=True`, choose your strongest measured `TUNE_BASE_EXPERIMENT`, and shorten `TUNING_TRIALS` to fit the available compute. Each override really trains under a separate `<base>__<trial>_<tag>` folder. Trials vary one factor independently; promising combinations require a new trial/tag. Methodological/hyperparameter changes are not additional adaptation techniques.

All controls are in **Section 3** and notebook **Section 12** / [TUNING.md](TUNING.md) explains each. When changing data/self-exclusion or lexical settings, rebuild Sections 4–5. Other manual training changes use a fresh tag. Config fingerprints prevent accidentally reusing incompatible runs. Bigger context, more negatives or more trials need not improve hidden-test relevance.

## Mandatory knowledge distillation

`lightweight_kd` starts from your Phase 1 lightweight checkpoint and keeps alternating layers, reducing its parameter count. Its teacher must be a larger completed compatible **Phase 7 retrieval model**. Auto-selection uses dense-only dev Micro-F1@7, so a teacher is selected by learned retrieval ability rather than its BM25 contribution. `TEACHER_RUN_DIR` pins a source folder. Resumed students retain the teacher from their saved metadata even if another teacher finishes later.

The frozen teacher scores the same train query/candidate IDs as the student, with its own representation settings. Temperature-scaled distribution KL is combined with supervised retrieval loss. Candidate masks and ID order are aligned; masked positions do not produce infinities/NaNs. Neither dev nor test relevance becomes a teacher training target. Online KD adds inference compute and memory for the frozen teacher; reduce query batch/microbatch sizes as needed.

## Exact metrics and verification

Primary **Micro-F1@7**, secondary **Micro-Precision@7 and Micro-Recall@7**:

```text
hits       = sum over queries |predicted_top_7 ∩ complete_gold_set|
predicted  = 7 × number_of_queries
gold       = sum over queries |complete_gold_set|
precision  = hits / predicted
recall     = hits / gold
F1         = 2 × hits / (predicted + gold)
```

Gold sets are deduplicated but **never truncated or padded** to seven, whether smaller, larger or empty. Zero denominators score zero. The reported oracle F1 describes the limit imposed by the exact-seven contract. Scores are deterministic set arithmetic; no external metric model is downloaded.

Every evaluated epoch searches the full corresponding dev pool and selects its ranking policy. The checkpoint key is hybrid Micro-F1@7 with dense-only Micro-F1 as a tie-breaker. After fitting, the best checkpoint is strictly reloaded, dev embeddings/ranks are recomputed and both predictions and full metrics must match the saved selection result. Exported dev JSON is reread and rescored. Test predictions are independently checked for query coverage, exact seven-way uniqueness, original candidate-string values, pool membership and root-only JSON ZIP contents.

Validation scores are selection scores, not an untouched independent estimate. Inspect `dev_errors.json` for missed gold IDs and irrelevant retrieved cases. Compare dense-only, lexical and selected hybrid scores, representation coverage, rare/long queries, parameter counts and runtime. Code cannot prove legal relevance or private-test performance; actual Codabench scores must come from your submitted predictions.

## Outputs and submission

All results save automatically to:

```text
MyDrive/cs781_phase7/<experiment>_<RUN_TAG>/
```

Per run: full configuration, model/teacher metadata, `last.pt`, `best_model.pt`, history, training curves, selected dev policy/predictions/metrics, dense dev metrics, ranking trials, error reports, dev/test embeddings, context coverage, verification JSON and submission files. Root files include the data/loading/token-length manifests, token caches, `experiment_summary.csv`, `coverage.json` and `best_per_architecture.json`.

Submission JSON is an **object**, unlike Phase 6:

```json
{
  "0000021652": ["0000590455", "0000916840", "0001481813", "0001907199", "0000000987", "0000001566", "0000002886"]
}
```

Each query has exactly seven unique original strings from its own test pool, ordered by predicted relevance. `submission.zip` contains only `prediction.json` at its root. Upload manually to Codabench. Keep required architecture/method submissions even if another run has the best dev score; a lexical baseline or a repeated identical ZIP does not replace compulsory adaptation experiments.

## Local verification and limits

Build: `python Phase7/build_phase7_notebook.py`.

Test: `python Phase7/validate_phase7_notebook.py` with torch, numpy, tokenizers, tqdm, matplotlib, nbformat, huggingface_hub, scipy and scikit-learn.

The original local Phase 1 weights were deleted during cleanup. Offline tests use small checkpoints created from the original architecture definitions and synthetic six-file datasets; they verify strict loading and complete training/ranking/export behavior. They do not establish actual task scores or replace loading the real Drive checkpoints. The Hugging Face repository metadata confirms all six documented file paths; file contents require authentication and could not be inspected without your token. Actual dataset compatibility and CUDA execution are runtime checks. See [VALIDATION.md](VALIDATION.md).

## Saved run results

[Browse Phase 7 results](../results/phase7/) for six trained runs and a separate BM25 baseline, saved settings, metric comparisons, coverage limits, plots and verification. Historical training commits and Codabench scores/IDs remain unavailable. See the [experiment ledger](../experiments/runs.csv).

The imported results came from the friend’s Drive account. Checkpoint paths refer to that account until a separate copy is made and verified. Selected ranking scores may include BM25 fusion; dense-only metrics and the lexical baseline are reported separately.
