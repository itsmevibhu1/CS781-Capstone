All editable controls are in **Section 3**. Start with one complete `decoder_full` run, inspect runtime,
coverage, label frequencies and dev errors, then compare methods. Phase 4 scores do not select Phase 5 winners.

| Goal | Section 3 control |
|---|---|
| Run one profile | `EXPERIMENT='hybrid_lorma'`, `RUN_ALL=False` |
| Six core runs | `RUN_ALL=True` (KD runs last) |
| Seven optional profiles | `RUN_EXTRA_APPROACHES=True` |
| Bounded automatic tuning | `RUN_TUNING=True`, `TUNE_BASE_EXPERIMENT`, edit/shorten `TUNING_TRIALS` |
| Fresh manual comparison | New `RUN_TAG`; every run starts from Phase 1 |
| Exact resumption | Same settings, experiment and tag; `RESUME=True` |
| Training duration | `EPOCHS`, `EARLY_STOPPING_PATIENCE` |
| Learning rate | Change the profile's `lr` in `CORE_EXPERIMENTS`/`EXTRA_EXPERIMENTS` after their definitions |
| Head learning rate | `HEAD_LR_MULTIPLIER` times the profile LR |
| Effective case batch | `BATCH_SIZE * GRAD_ACCUM_STEPS` |
| GPU chunk microbatch | `CHUNK_BATCH_SIZE` (reduce if out of memory) |
| More document coverage | `MAX_CHUNKS=8`; `0` or policy `all` uses every window and costs more |
| Longer windows | `CHUNK_LENGTH`, `CHUNK_OVERLAP`; must fit the Phase 1 positional limit |
| Context selection | `TRAIN_CHUNK_POLICY`, `EVAL_CHUNK_POLICY`: spread/head/tail/head_tail/all; random_mix is train-only |
| Hierarchical pooling | `DOC_POOLING`: attention/mean/max/mean_max/transformer |
| Rare label imbalance | `USE_CLASS_WEIGHTS`, `CLASS_WEIGHT_POWER`, `POS_WEIGHT_CAP` |
| Loss | `LOSS_NAME='bce'`, `'focal'` or `'asymmetric'`; gamma/clip controls nearby |
| Label priors | `INITIALIZE_LABEL_PRIORS`; estimates use train labels only |
| Label relationships | Tuning override `label_graph_weight=0.2`; graph is computed from train co-occurrence only |
| Thresholds | `THRESHOLD_MODE`, `THRESHOLD_GRID`, `MIN_DEV_POSITIVES`, `THRESHOLD_SHRINKAGE` |
| Empty predictions | `ALLOW_EMPTY_PREDICTIONS=False` forces top one only when no statute passes its threshold |
| Official scorer convention | `METRIC_LABEL_SCOPE`; all three Macro-F1 variants are saved |
| Exact-text split overlap | `SPLIT_OVERLAP_POLICY='exclude_train'` removes training copies of dev/test; `error` stops; `report` retains all and audits |
| LoRMA | `ADAPTER_RANK`, `ADAPTER_ALPHA` |
| Residual adapters | `BOTTLENECK_DIM` |
| Partial/gradual tuning | `PARTIAL_LAYERS`, `UNFREEZE_LAYERS_PER_EPOCH` |
| Distillation | `KD_WEIGHT`, `KD_TEMPERATURE`, `TEACHER_RUN_DIR` |
| Ensemble | `BUILD_ENSEMBLE=True`, `ENSEMBLE_TOP_K`, `ENSEMBLE_PAIR_WEIGHTS` |

**Individual parallel sessions:** make separate Colab notebook copies and choose distinct experiments.
Keep `RUN_ALL=False`, `RUN_EXTRA_APPROACHES=False`, `RUN_TUNING=False`, `BUILD_ENSEMBLE=False`.
Prepare Sections 1–5 once before starting other sessions, and do not train the same experiment/tag
in two runtimes. Distinct experiments may share a tag; each writes its own folder. The final summary
and ensemble can be regenerated in one runtime after all training completes.

**KD:** finish a larger Phase 5 teacher first. Leave `TEACHER_RUN_DIR=None` to select the best compatible
completed larger non-KD model by dev Macro-F1, or supply an explicit source folder. The teacher may use
another tag/window budget. Its train logits are cached and checked against checkpoint metadata. The
student starts from alternating Phase 1 lightweight layers, not random or external pretrained weights.

**Tuning really trains each trial.** Overrides are saved in distinct `<base>__<trial>_<tag>` folders.
Trials vary one factor relative to the base, not cumulatively. To combine promising settings, create a
new trial with several overrides or edit the controls under a fresh tag. Per-label thresholds are fitted
inside each run on dev. They do not create additional fine-tuning techniques for requirement counts.

**Compare:** use the same official dev split and primary metric convention. Inspect `per_label_metrics.csv`,
`rare_label_analysis.json`, `dev_errors.json`, `chunk_coverage.json` and train/dev curves. The best model
per origin is listed in `best_per_architecture.json`; retain architecture-specific submissions even if a
cross-architecture ensemble scores higher. `experiment_summary.csv` compares all compatible tags.

The PDF has conflicting submission counts: its general guidelines say six for Phases 2–7, while §6.5 and
§11 specify four. Six core profiles are supplied to cover the stricter count while meeting all four origins,
at least three adaptation methods, four listed approaches, compulsory LoRMA and KD. Actual accepted
submissions and uniqueness must be checked on Codabench; saved ZIPs alone do not prove acceptance.

The scorer's exact handling of absent IDs/ID gaps is not supplied. `reference_range` includes IDs from
zero through the largest gold ID; `reference_present` averages only IDs occurring in gold; `known`
averages the full known catalog. All use zero F1 for zero-denominator labels. Match any published
official scorer rather than selecting whichever convention makes the local number largest.

Train/dev calibrations are selection scores. Private test scores cannot be verified locally without its
labels. Do not tune on test feedback. More trials or epochs need not improve unseen performance.
