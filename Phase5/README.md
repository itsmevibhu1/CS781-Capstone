# Phase 5 — Legal Statute Identification

Open [CS781_Phase5_Solution_Colab.ipynb](CS781_Phase5_Solution_Colab.ipynb) in Colab, select a GPU, add a secret named **HF** containing your plain `hf_...` token, and run the cells in order. No external pretrained model or external LLM API is used. The notebook is standalone; uploading the local Python source files is unnecessary.

Inputs: `MyDrive/cs781_phase1/bpe_tokenizer/` and the four `run_<architecture>/final_model.pt` Phase 1 checkpoints. This follows the professor's starter progression while replacing its generation objective with the multi-label task in §11 of the capstone document. The encoder–decoder origin contributes its pretrained encoder.

Official supplied data:

- `Phase_5_LSI/Train/train.json`: model fitting, positive weights, priors, co-occurrence graph and teacher training logits.
- `Phase_5_LSI/Train/dev.json`: checkpoint selection, threshold calibration, hyperparameter comparison and ensemble selection.
- `Phase_5_LSI/Test/test.json`: prediction only. Any supplied test label field is ignored.

Drive outputs: **`MyDrive/cs781_phase5/<experiment>_<RUN_TAG>/`**. Shared label catalogs, manifests and disk-backed token caches live directly in `cs781_phase5/`. Phase 1–4 outputs are not overwritten.

## First run and required approaches

Section 3:

```python
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
BUILD_ENSEMBLE = False
RUN_TAG = 'v1'
```

The six core profiles are:

| Exact experiment string | Phase 1 origin | Actual adaptation |
|---|---|---|
| `decoder_full` | Decoder-only | Full backbone fine-tuning |
| `encoder_partial` | Encoder–decoder | Last encoder layers and final norm |
| `hybrid_lorma` | Hybrid Prefix-LM | LoRMA+ on frozen attention projections |
| `hybrid_adapters` | Hybrid Prefix-LM | Residual bottleneck adapters |
| `lightweight_lorma` | Lightweight | LoRMA+ |
| `lightweight_kd` | Lightweight, alternating pretrained layers retained | Smaller student with hard labels and larger teacher's Bernoulli targets |

After the first successful run, `RUN_ALL=True` runs the entire core list in order, reusing a completed matching first run. KD runs last so a larger trained teacher is available. All new document heads train during their respective methods; no separate manual adaptation step is needed.

The PDF's general guidelines say six submissions, while §6.5 and §11 give a minimum of four. Six core runs cover the stricter count plus all four Phase 1 origins, compulsory LoRMA/KD, at least three adaptation methods and four listed techniques. The report tracks both counts. Only distinct accepted Codabench submissions establish submission compliance. Do not drop a compulsory method's submission solely because another profile scores higher.

For independent Colab runtimes, select a different experiment string in each notebook copy and leave sweep/ensemble flags false. Prepare Sections 1–5 in one runtime before starting others. Do not run the same experiment/tag twice concurrently. Train KD after a larger teacher finishes. After all runs, use one runtime to regenerate the summary and optional ensemble. Colab's runtime availability is external to this code.

## Score optimization and optional experiments

The default hierarchy processes spread windows, represents each using the Phase 1 backbone, and combines chunk features/positions into a document head with **one independent logit per statute**. Multiple statutes can be predicted. The initial default reads four sampled windows rather than claiming complete coverage; saved coverage measures the fraction actually used.

Implemented comparisons include:

- Full, partial, LoRMA+, residual adapters, gradual unfreezing, frozen-backbone probing and smaller-student distillation.
- Mean, max, mean+max, learned attention and chunk-Transformer pooling.
- Spread, head, tail, head+tail, random training windows and complete-window document representations.
- Positive-weighted/unweighted BCE, focal and asymmetric losses; label smoothing; train-derived head biases.
- Global thresholds and support-dependent per-label thresholds shrunk toward a global threshold.
- Optional train-label co-occurrence propagation, tested by the `label_graph` tuning trial.
- Bounded weighted-probability ensembles aligned by original case instances and statute ordering.

Seven additional profiles are `lightweight_full`, `encoder_full`, `decoder_lorma`, `encoder_adapters`, `decoder_gradual`, `decoder_linear_probe` and `hybrid_full`. Select one individually or set `RUN_EXTRA_APPROACHES=True`. A linear probe is a baseline, not a replacement for the required adaptation techniques.

`RUN_TUNING=True` really trains the thirteen provided one-factor trials, with separate output folders. Choose `TUNE_BASE_EXPERIMENT` from measured Phase 5 dev results and shorten `TUNING_TRIALS` to fit your GPU budget. Change manual settings under a new tag. Hyperparameter variants do not count as new adaptation methods. More training or trials do not guarantee improved hidden-test scores.

The complete controls table and examples are in notebook **Section 14** and [TUNING.md](TUNING.md).

## Mandatory knowledge distillation

`lightweight_kd` begins with your Phase 1 lightweight weights and keeps alternating backbone layers. A teacher must be a **larger completed compatible Phase 5 non-KD model**. The default chooses the highest dev Macro-F1 among eligible models; `TEACHER_RUN_DIR` pins a particular source. Source tags and chunk budgets may differ from the student.

The teacher is evaluated only on train for its cached training targets. Distillation uses independent sigmoid/Bernoulli targets, since softmax would incorrectly make statutes mutually exclusive. `KD_WEIGHT` and `KD_TEMPERATURE` control its contribution. The teacher cache is tied to source checkpoint metadata, label/data fingerprint and tokenizer; cache construction preserves student RNG state so cold/warm cache runs agree.

## Metrics and verification

Primary: **Macro-F1 over statute decisions**, not Macro-F1 over binary outcome classes. Secondary: **Micro-F1 and Subset Accuracy**. Zero-denominator F1 is zero.

The PDF says the evaluator derives the label dimension from reference data, but does not supply exact scorer code. Default `METRIC_LABEL_SCOPE='reference_range'` assumes integer columns `0..max(gold statute ID)`; it reports Macro-F1 for observed reference labels and for the full known catalog as well. Secondary metrics use the chosen scope, with full-catalog alternatives also saved. If an official scorer clarifies a different convention, change this setting. Do not choose a convention merely to inflate scores.

Every evaluated epoch compares the 0.5 baseline with bounded global/per-label threshold calibration on dev. Rare-label thresholds are regularized; a dev-worse per-label policy does not replace the global/0.5 baseline. After training, the best checkpoint is strictly reloaded and every recorded dev metric must match. Exported dev JSON is decoded into multi-hot labels and independently rescored. Case ID values/types, valid integer statute IDs, uniqueness, document counts and ZIP contents are verified.

Strict official-schema and format checks include repeated labeled IDs, paragraph-list text, JSON/JSONL/CSV/safe literal exports, empty/bad download detection and one fresh-download retry. Original test IDs must remain unique. Original ID overlap across splits is audited because IDs can be local identifiers. Default `SPLIT_OVERLAP_POLICY='exclude_train'` removes only training instances whose exact nonempty text also occurs in dev/test; every dev/test document, original ID and label remains untouched. Training counts, loss weights, head priors and label co-occurrence are recomputed using the retained training rows. `split_audit.json` reports before/after counts and examples; `excluded_train_instances.json` lists every excluded training instance by ID/hash, without copying full text. Dev/test duplicates are reported and retained. The `error` policy restores strict stopping; `report` retains all official training rows while explicitly reporting remaining overlap. No training labels are transferred to held-out predictions.

If an earlier notebook stopped with `Identical nonempty texts across splits`, upload the updated notebook and run from the beginning with the default overlap policy. Training had not started, so your existing run tag is usable. A completed run made with another overlap policy needs a new tag because the dataset fingerprint changes.

The statute catalog is inferred from train+dev; `STATUTE_IDS` accepts an official valid-ID list if provided. No nonexistent intervening IDs are predicted merely because IDs have gaps. Labels with no train positives are explicitly reported. Only actual official runtime/Codabench evaluation can establish task performance; synthetic test metrics are not competition results.

## Results and submission

Each run saves:

- Configuration, model/adaptation metadata, class/label weights and chunk coverage.
- `last.pt` for exact resume, `best_model.pt`, `history.json` and train/dev curves.
- `metrics.json`, `best_dev_metrics.json`, `threshold.json` and `per_label_metrics.csv`.
- `dev_prediction.json`, dev/test probabilities, `dev_errors.json`, rare-label analysis and missed-statute plot.
- `verification.json`, `prediction.json` and **`submission.zip`**.

`prediction.json` is a list such as:

```json
[{"id": "case_001", "pred": [1, 7, 23]}, {"id": "case_002", "pred": [5, 81]}]
```

The ZIP contains only `prediction.json` at its root. Upload each appropriate ZIP to Codabench manually.

Root `experiment_summary.csv`, `coverage.json` and `best_per_architecture.json` compare compatible saved tags. The best-per-origin list helps retain the best four different Phase 1 architectures. `BUILD_ENSEMBLE=True` optionally creates an additional `ensemble_<tag>/submission.zip`, using dev-selected blends and retaining a single-model baseline when no pair improves it. Ensembles do not count as additional adaptation methods or replace required architecture-specific experiments.

## Local verification

Build: `python Phase5/build_phase5_notebook.py`.

Test: `python Phase5/validate_phase5_notebook.py` in an environment with torch, numpy, tokenizers, tqdm, matplotlib, nbformat and huggingface_hub. Tests use the four actual local Phase 1 checkpoints plus synthetic complete train/dev/test pipelines; they do not authenticate to the gated dataset. See [VALIDATION.md](VALIDATION.md).

## Saved run results

[Browse Phase 5 results](../results/phase5/) for six documented runs, configurations, score comparisons, coverage, plots, and saved verification. Training-complete flags pass in the imported artifacts. These are development selection results; historical code commits and Codabench IDs/scores/acceptance remain unrecorded. See the [experiment ledger](../experiments/runs.csv).
