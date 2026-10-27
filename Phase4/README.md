# Phase 4 — Court Judgment Prediction

Open [CS781_Phase4_Solution_Colab.ipynb](CS781_Phase4_Solution_Colab.ipynb) in Colab. Select a GPU, create a secret named **HF** containing only your `hf_...` access token, and run all cells.

Inputs: `MyDrive/cs781_phase1/` (your four trained weights and shared BPE tokenizer).
Outputs: **`MyDrive/cs781_phase4/<run_name>_<RUN_TAG>/`**.

The notebook follows the professor's starter workflow and Phase 4 §10. It downloads all three official files:

- `Phase_4_CJPE/Train/train.json` — fitting only.
- `Phase_4_CJPE/Train/dev.json` — checkpoint, hyperparameter, threshold and ensemble selection.
- `Phase_4_CJPE/Test/test.json` — unlabeled prediction only.

Output format differs from Phase 3: `prediction.json` is a **list** of `{"id": original_id, "pred": 0}` objects, with an integer outcome of 0 or 1 for every test case. The ZIP contains only this JSON at its root. Upload the ZIP to Codabench manually.

Section 4 retains every train/dev instance when several rows share a case ID. Internal row keys prevent overwriting texts, labels, probabilities or validation predictions; original IDs remain in exported JSON. `MyDrive/cs781_phase4/case_instance_audit.json` records row counts and repeated-ID examples. Cross-split checks still compare original case IDs and texts. Duplicate test IDs stop loading for inspection because assigning new test IDs would change the submission contract.

Section 4 detects the contents of each downloaded file instead of assuming its `.json` suffix describes the format. Supported inputs are JSON, JSON Lines, CSV with explicit `id`/`text` columns, and safely parsed Python literal lists/mappings. UTF-8 BOM and UTF-16 BOM are supported. Blank files, LFS pointers, HTML/XML responses and malformed records are rejected; no invalid lines are skipped. Parsing failures trigger one fresh Hugging Face download. `dataset_loading_report.json` in the Phase 4 Drive folder records the split, byte size, format and any failed attempts. If parsing still fails, the exception names the specific split and source filename.

If you encountered `Duplicate/ambiguous case ID 2019_890` before training, upload the updated notebook to Colab, run Sections 1–4 and continue through the remaining sections. The failed run has no trained results to resume. Keep your chosen experiment and hyperparameters. Existing completed runs using an older loader require a new `RUN_TAG` because instance keys change the dataset fingerprint.

## First run and six core methods

In **Section 3**:

```python
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_TAG = 'v1'
```

After checking the first run's actual dev scores, coverage and runtime, keep the settings/tag unchanged and set `RUN_ALL=True`. The completed decoder training is reused; the remaining core experiments run sequentially.

| Profile | Phase 1 origin | Adaptation |
|---|---|---|
| `decoder_full` | Decoder-only | Full fine-tuning |
| `encoder_partial` | Encoder–decoder encoder | Last encoder layers and final normalization |
| `hybrid_adapters` | Hybrid Prefix-LM | Post-block residual bottleneck adapters |
| `lightweight_lorma` | Lightweight | LoRMA+ |
| `decoder_bitfit` | Decoder-only | Backbone biases plus new document head |
| `lightweight_kd` | Smaller lightweight student | Hard labels and cached teacher-logit distillation |

These are actual training/adaptation procedures, performed inside the training function. There is no further manual PEFT step. Core methods cover all four Phase 1 origins, compulsory LORMA/KD, and more than four listed approaches. The notebook checks the general six-distinct-submission minimum as well as method/origin coverage; it cannot verify Codabench acceptance.

## More approaches and automatic tuning

Set `RUN_EXTRA_APPROACHES=True` to additionally run:

- `lightweight_full`
- `encoder_full`
- `hybrid_lorma`
- `lightweight_adapters`
- `decoder_gradual` — progressively unfreeze more blocks.
- `decoder_linear_probe` — frozen backbone, learned document head.

Any profile can also be selected individually with `RUN_ALL=False`. The linear probe is a comparison baseline, not a substitute for a required adaptation method.

Set **`RUN_TUNING=True`** to execute the bounded `TUNING_TRIALS` list using `TUNE_BASE_EXPERIMENT`. The eight provided trials compare lower LR, more chunks, longer chunks, mean+max pooling, chunk-Transformer pooling, head/tail selection, random training windows and unweighted loss. They really train separate models. Edit or shorten the list according to measured runtime. Hyperparameter variations do not count as new adaptation methods.

For example, after completing a teacher:

```python
EXPERIMENT = 'lightweight_kd'
RUN_ALL = False
RUN_TAG = 'kd_chunks8'
MAX_CHUNKS = 8
```

This starts a new Phase 1-initialized student. Unlike the Phase 3 workflow, a compatible completed teacher may come from **another run tag** and use different chunk settings. Set `TEACHER_RUN_DIR` to an explicit source folder or leave it `None` to select the strongest eligible larger teacher by dev macro-F1. Its train-only logits are cached once and checked against the source checkpoint.

## Where to change what

**All controls are in Section 3. Section 14 contains the detailed tuning table.**

- Training: `EPOCHS`, the selected `EXPERIMENTS[name]['lr']`, `HEAD_LR_MULTIPLIER`, dropout, class weighting, patience.
- Effective case batch: `BATCH_SIZE * GRAD_ACCUM_STEPS`.
- Long-document representation: `MAX_CHUNKS`, `CHUNK_LENGTH`, overlap, head/tail/spread policies and `DOC_POOLING`.
- Memory: `CHUNK_BATCH_SIZE` controls chunk microbatches independently of cases per batch.
- Adaptation: LORMA rank/alpha, adapter bottleneck dimension, partially trainable layers, gradual-unfreezing rate.
- Distillation: `KD_WEIGHT`, `KD_TEMPERATURE`, optional explicit teacher directory.
- Automation: `RUN_ALL`, `RUN_EXTRA_APPROACHES`, `RUN_TUNING`, `TUNING_TRIALS`, `BUILD_ENSEMBLE`.

Use a **new `RUN_TAG`** when changing a manual setting; rerun Section 3 onward. Keep the original split and seed for controlled comparisons. For exact resume, keep the same tag and settings. Increasing epochs under a new tag starts from Phase 1 with a new schedule; it does not extend a completed schedule.

The default reads up to four spread-out overlapping windows, including beginning/end. It is a sampled representation. `chunk_coverage.json` reports actual coverage. Increase the budget after inspecting coverage/runtime; `MAX_CHUNKS=0` or policy `all` reads all windows and can be expensive. Token IDs use memory-mapped int32 cache files, avoiding a full in-memory token corpus; Colab stages canonical Drive caches onto local disk for faster reads.

## Scores, verification and outputs

Primary: **macro-F1 across classes 0 and 1**. Secondary Precision/Recall are reported for class 1, with macro and both class metrics also saved because the PDF does not specify the secondary averaging convention.

Each epoch selects a probability threshold on dev and uses the resulting macro-F1 for checkpoint selection. After training the best checkpoint is reloaded and every reported dev metric is recomputed. The final report includes per-class counts, confusion matrix, dev errors, chunk coverage, train/dev curves and prediction checks.

`BUILD_ENSEMBLE=True` compares a few probability blends of the best compatible saved runs against the best single model, using dev only. It does not retrain backbones and retains the single-model baseline if no blend improves dev F1. An extra `ensemble_<tag>/submission.zip` and recipe are saved when at least two compatible completed runs exist. Ensembles do not count as another PEFT technique.

Per run: configuration, metadata, coverage, class weights, last/best checkpoints, history, baseline/selected dev scores, threshold, dev predictions/probabilities/errors, plots, test probabilities, verification JSON and submission files. Root-level `experiment_summary.csv` and `coverage.json` summarize compatible runs across tags and check distinct prediction contents.

Actual task performance must be measured in Colab. Dev-selected scores are selection scores; repeated tuning may overestimate private-test generalization. Keep the required LLM declaration and cite the references linked in the notebook. Do not treat synthetic validation-test scores as capstone results.

## Offline validation

Run `python Phase4/validate_phase4_notebook.py` in an environment containing torch, tokenizers, numpy, tqdm, matplotlib, nbformat and huggingface_hub. The script uses actual local Phase 1 checkpoints plus synthetic train/dev/test files and writes temporary artifacts under `/tmp`. See [VALIDATION.md](VALIDATION.md) for the verification scope.

## Repository results status

The code and existing offline validation notes are included here. Saved official run artifacts have not yet been imported into this repository. Current notebook defaults do not establish settings for past submissions; keep the original run configurations and code versions. See the [results status](../results/phase4/README.md). Codabench scores, IDs, and acceptance must come from actual competition records.
