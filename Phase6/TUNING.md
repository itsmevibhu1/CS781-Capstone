All controls are in **Section 3**. Start with one `encoder_full` run. Set up a Colab GPU, an `HF` secret
containing your plain `hf_...` token, and `` with all four final Phase 1 checkpoints
and the BPE tokenizer. This notebook is standalone; no Phase 2–5 training is rerun.

In an already prepared runtime, changing only the experiment requires rerunning **Section 3, then
Sections 10–11**. A new runtime must run all cells. Changing `SEED`, `VALIDATION_FRACTION` or overlap
filtering requires rebuilding Section 4 as well; rerun all cells for those dataset changes. Use fresh
tags for manual changes. Changing BERTScore model/layers/rescaling rebuilds its cached metric model.

| Goal | Control |
|---|---|
| One experiment | `EXPERIMENT='encoder_full'`, `RUN_ALL=False` |
| Six core experiments | `RUN_ALL=True`; larger teachers finish before KD |
| Ten optional profiles | `RUN_EXTRA_APPROACHES=True` |
| Automatic training sweep | `RUN_TUNING=True`, edit `TUNING_TRIALS`, set `TUNE_BASE_EXPERIMENT` |
| Dev-only generation sweep | `RUN_DECODING_SEARCH=True` after the selected model has finished |
| Fresh manual trial | New `RUN_TAG`; `LEARNING_RATE` overrides the profile LR |
| Resume interrupted training | Identical tag/settings; `RESUME=True` |
| Epochs/overfitting | `EPOCHS`, `EARLY_STOPPING_PATIENCE`; select by held-out BERTScore, not training loss |
| Case batch and accumulation | `BATCH_SIZE`, `GRAD_ACCUM_STEPS`; loss weighting counts actual supervised tokens |
| Input and target lengths | `MAX_SOURCE_TOKENS`, `MAX_TARGET_TOKENS` |
| Source selection | `SOURCE_POLICY`: head/tail/head_tail/spread/sentence_mmr |
| Native encoder hierarchy | `ENCODER_CHUNK_SIZE=384` independently encodes source windows and combines their memories |
| LoRA/LoRMA capacity | `RANK`, `ALPHA` |
| Adapter capacity | `BOTTLENECK_DIM` |
| Soft prompt capacity | `PROMPT_TOKENS`; frozen backbone, learned continuous prefix |
| Partial tuning depth | `PARTIAL_LAYERS` |
| Teacher/student transfer | `KD_WEIGHT`, `KD_TEMPERATURE`, `TEACHER_RUN_DIR` |
| Generated length | `GENERATION['max_new_tokens']`, `['min_new_tokens']` — tokens, not words |
| Greedy vs beam | `GENERATION['num_beams']=1` vs 2/3 |
| Length/repetition | `length_penalty`, `repetition_penalty`, `no_repeat_ngram_size` |
| Redundant sentences | `GENERATION['deduplicate_sentences']=True` |
| Two-stage generation | `GENERATION['long_document_mode']='map_reduce'`, `map_reduce_max_chunks` |
| Validation fraction | `VALIDATION_FRACTION`, `SEED`; keep fixed for comparable experiments |
| Missing/unusable training fields | `INVALID_TRAIN_ROW_POLICY='exclude'` audits/excludes them; `error` stops |
| Unexpected invalid-record rate | `MAX_INVALID_TRAIN_FRACTION=0.05`; inspect data_quality_report.json if exceeded |
| Epoch selection runtime | `EPOCH_VALIDATION_LIMIT=128`; 0 uses all held-out documents each epoch |
| Final metric verification | Keep `FINAL_VALIDATION_LIMIT=0` to score every held-out document |
| Validation loss runtime | `EVAL_LOSS_LIMIT`; diagnostic only, not a replacement for generation metrics |
| BERTScore settings | `BERTSCORE_MODEL`, `BERTSCORE_LAYERS`, `BERTSCORE_RESCALE`; match any published official scorer |
| Metric runtime/memory | `BERTSCORE_BATCH_SIZE`, `BERTSCORE_DEVICE='cpu'` or `'cuda'` |

Core profiles: `encoder_full`, `decoder_lora`, `hybrid_lorma`, `encoder_adapters`,
`lightweight_prompt`, `lightweight_kd`. They cover all four origins, compulsory LoRMA/KD and at least
four techniques explicitly listed in Phase 6. The PDF has conflicting global submission minimums
(six in general guidelines, four in §6.5/§12); six substantially different core runs cover the stricter count.
Check accepted and distinct submissions on Codabench yourself; local ZIPs do not establish acceptance.

Optional profiles: `decoder_full`, `encoder_lora`, `encoder_lorma`, `encoder_partial`, `hybrid_lora`,
`hybrid_adapters`, `lightweight_lorma`, `lightweight_lora`, `lightweight_pfeiffer`, `lightweight_full`.
Full/partial fine-tuning are useful baselines in addition to the techniques explicitly listed in §12.2.1.
The adapter implementations use the described residual sites in your existing pre-norm architecture;
the Pfeiffer-style profile has one output bottleneck per layer.

Run individual experiments in separate Colab notebook copies with `RUN_ALL=False`,
`RUN_EXTRA_APPROACHES=False`, `RUN_TUNING=False`, `RUN_DECODING_SEARCH=False`. Prepare Section 4 once
before other sessions use shared caches. Do not run the same experiment/tag concurrently. Run KD after a
larger Phase 6 teacher completes. Changing a manual training/decoding setting requires a fresh tag;
the decoding search uses a separate `<experiment>__decoding_<tag>` directory without retraining.

The training sweep really trains each listed trial and saves overrides under separate names. Trials
are one-factor changes, not cumulative. Choose the strongest measured base and shorten the list to fit
your GPU budget. Source/target budgets share the decoder-only context; longer summaries reduce its
available source context. The native encoder–decoder has separate source and target context budgets.
Generation never silently discards the beginning of a generated summary through a rolling window.

`head_tail`/`spread`/MMR are selective representations, not full document reading. `sentence_mmr` ranks
source sentences by document-centrality and redundancy without consulting reference summaries.
Map–reduce reads the selected chunk budget and performs several extra generation calls; compare on
the same validation set before submitting. Native encoder chunking changes attention scope while
retaining selected source tokens in cross-attention memory. Increasing budgets, beams or epochs is not
guaranteed to improve quality and costs more runtime.

**Distillation:** the student retains alternating Phase 1 lightweight layers. A larger trained Phase 6
teacher supplies token-level distributions on **train references only**. Both models use the same BPE
and target sequence; their document representations can differ. The teacher is frozen, not updated by
student gradients. No validation summaries become training labels. Expect extra training compute and
memory for two models; reduce `BATCH_SIZE` and increase accumulation if necessary. This is online
token-KL distillation; no sequence-cache or external teacher is silently substituted.

**Scores:** checkpoint selection scores are from a fixed held-out subset by default. Final metrics are
from the full held-out split, which includes those selection cases and is not an untouched independent
test. The best subset epoch may differ from the best hypothetical full-validation epoch. Use full epoch
validation (`EPOCH_VALIDATION_LIMIT=0`) for closer model selection when compute permits.
All runs must use the same split and metric settings. BERTScore/ROUGE do not prove factual accuracy:
read summaries and inspect entities, dates, provisions, holdings, coverage and repetition yourself.

The official PDF does not specify scorer version, rescaling, layers or ROUGE stemming. Defaults are
explicit and saved with BERTScore's hash, not claimed to reproduce an unseen scorer. RoBERTa-large is
downloaded **only as a metric**; the summarizer and KD teacher use Phase 1 weights exclusively. Do not
change the metric model just to raise numbers. Long summaries may be truncated by the metric model's
tokenizer; keep evaluation settings consistent and compare realistic output lengths.

Run outputs: `<experiment>_<RUN_TAG>/`. Keep `best_model.pt` for inference/teachers,
`last.pt` for resume, and all configurations/metrics/predictions/verification files for your report.
`prediction.json` is a list of `{id, summary}` objects; `submission.zip` contains only that JSON at its root.
`best_per_architecture.json` and `experiment_summary.csv` compare compatible complete runs. Retain the
required architecture-specific and compulsory-method submissions even if another run scores better.
Actual private test scores are available only from Codabench. Keep the required LLM declaration.
