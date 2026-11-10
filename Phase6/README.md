# Phase 6 — Abstractive Legal Summarization

Open [CS781_Phase6_Solution_Colab.ipynb](CS781_Phase6_Solution_Colab.ipynb) in Colab, select a GPU, add a secret named **HF** containing your plain `hf_...` token, and run cells in order. The notebook embeds all runtime code; no local Python modules need uploading.

The implementation follows **§12 of the capstone PDF**, the professor's Phase 1 starter progression and the earlier phases' authentication and checkpoint workflow. Unlike Phase 5 classification, it retains and adapts the full pretrained generation head. The encoder–decoder model retains both its pretrained encoder and decoder. No external summarization model or LLM API supplies outputs or training targets.

## Inputs and data split

Required inputs:

```text
bpe_tokenizer/vocab.json
bpe_tokenizer/merges.txt
run_decoder_only/final_model.pt
run_encoder_decoder/final_model.pt
run_hybrid_prefixlm/final_model.pt
run_lightweight/final_model.pt
```

The Phase 6 dataset has **two official files**, not three:

- `Phase_6_Summ/Train/train.json`: `id`, `document`, reference `summary`.
- `Phase_6_Summ/Test/test.json`: `id`, `document`; references are private and any test summary field is ignored.

As specified by §12.3.2, a public dev file does not exist. Section 4 derives a reproducible grouped held-out split from training data. Identical source documents remain together, preventing copies from appearing in both fitting and validation sets. Default exact train/test overlap filtering excludes training copies without changing any test ID or test document. Every exclusion and split membership is recorded. All remaining labeled documents are used for fitting or validation.

The default validation fraction is 5%. Set `VALIDATION_FRACTION` and `SEED` before initial preparation and keep them fixed across comparisons. `split_manifest.json` and data/tokenizer fingerprints protect reuse/resumption. Format checks support JSON/JSONL, appropriate-column CSV and safe literal exports; failed parsing triggers one fresh download. String-list document/summary paragraphs are joined in order. Default `INVALID_TRAIN_ROW_POLICY='exclude'` records and excludes training rows with missing, blank or unsupported document/summary fields; it never invents reference text. The default `MAX_INVALID_TRAIN_FRACTION=0.05` stops if over 5% are unusable, so a widespread schema mismatch cannot silently discard most of the dataset. Empty string input documents remain supported with UNK; unusable test document types stop with detailed errors, and no test row is skipped. Original test IDs and types remain unchanged.

`data_quality_report.json` contains per-split counts, reason counts and ID/type examples; `excluded_training_rows.json` records every excluded training row without full document text. An earlier error such as `299: expected document text and a nonempty training summary` did not distinguish the failing field. The updated loader identifies the exact cause and can retain valid paragraph-list summaries. Upload the updated notebook and run from the beginning; a run that stopped at data loading can retain its tag.

## Start in Colab

Section 3:

```python
EXPERIMENT = 'encoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
RUN_DECODING_SEARCH = False
RUN_TAG = 'v1'
```

The native encoder–decoder is the first baseline because it already has source–target cross-attention. Its actual quality must be measured. After verifying the first run, use `RUN_ALL=True` for the six core profiles. The completed first run is reused after config and output checks; it does not retrain or regenerate summaries.

| Exact profile | Phase 1 origin | Adaptation |
|---|---|---|
| `encoder_full` | Encoder–decoder | Full source–target fine-tuning |
| `decoder_lora` | Decoder-only | Additive low-rank attention projection updates |
| `hybrid_lorma` | Hybrid Prefix-LM | Compulsory LoRMA+ multiplicative updates |
| `encoder_adapters` | Encoder–decoder | Houlsby-style bottlenecks after attention/FF residual sites |
| `lightweight_prompt` | Lightweight | Learned continuous prompt, frozen backbone |
| `lightweight_kd` | Lightweight with alternating Phase 1 layers retained | Smaller student's token-level distillation from a trained larger teacher |

Core profiles cover all four Phase 1 origins and at least four techniques explicitly listed in §12.2.1, including compulsory LoRMA and KD. Full/partial fine-tuning are additional useful baselines. General submission counts conflict in the PDF: six in its opening guidelines, four in §6.5/§12. Six substantially different core runs cover the stricter count. Saved outputs alone do not establish distinct, accepted Codabench submissions.

For parallel Colab notebooks, choose one profile per copy, leave all sweep flags false and prepare shared tokens in one session first. Never train the same experiment/tag in two sessions. Run KD after a larger Phase 6 teacher completes. Phase 4/5 classification teachers are not compatible summarization teachers.

## Long documents and efficient generation

Decoder-only/lightweight models train on instruction + selected source + summary prefix + reference target. Hybrid training uses bidirectional source-prefix attention and causal target attention. Native encoder–decoder training encodes source tokens separately and predicts shifted reference summaries. **Only target tokens contribute to loss**; source and padding positions are excluded.

Default source policy is head+tail with a 640-token source budget and up to 256 target tokens. These are editable starting points, not an optimized claim. Decoder families share a fixed context, so longer targets/prompts can reduce available source context. Native encoder–decoder source and target budgets are separate.

Source comparisons include head, tail, head+tail, spread windows and reference-free sentence MMR. Native encoder chunking independently encodes selected windows and combines their states for decoder cross-attention. Optional map–reduce generates intermediate selected-chunk summaries followed by a final generated summary. It is an inference comparison, not extra chunk/reference training with invented labels.

Generation reuses **self-attention KV caches and native decoder cross-attention KV caches**. Source encoder states are computed once. This avoids the Phase 1 starter's full-prefix recomputation at every token. Greedy/beam decoding, generation length, length penalty, repetition penalty, n-gram blocking and sentence deduplication can be compared on validation. No rolling window silently discards the start of a generated target.

`context_coverage.json` reports source length percentiles, source-budget coverage bounds and reference truncation fractions. Inspect this file before increasing lengths. MMR may select fewer tokens than the budget, so its reported budget coverage is an upper bound. More context or beams can improve or harm scores and always costs more compute.

## Additional approaches and optimization

Ten optional profiles:

```text
decoder_full       encoder_lora         encoder_lorma
encoder_partial    hybrid_lora          hybrid_adapters
lightweight_lorma  lightweight_lora     lightweight_pfeiffer
lightweight_full
```

Use `RUN_EXTRA_APPROACHES=True` or select any profile individually. The adapter implementations preserve the existing pre-norm blocks, with explicitly stated residual sites; they are not claims of exact paper reproduction. LoRMA+ applies `(I + alpha/r BA)W`; LoRA applies additive low-rank updates. Both start with zero update and freeze the base projections. Soft prompt tuning learns only prompt embeddings.

Eight bounded training trials compare LR, longer source/target budgets, spread/MMR/head context, label smoothing and independent encoder chunks. Set `RUN_TUNING=True`, choose the strongest measured `TUNE_BASE_EXPERIMENT`, and shorten `TUNING_TRIALS` to fit your budget. Each trial really trains and saves separately. These are not additional adaptation techniques.

Six bounded generation comparisons reuse a trained model without retraining: greedy, beam 3, longer, shorter, sentence deduplication and map–reduce. `RUN_DECODING_SEARCH=True` selects a policy using the fixed validation subset, then evaluates the selected policy on the full held-out split and exports a separate submission. The original model's submission remains intact. It retains its baseline policy if no candidate improves validation BERTScore.

All controls are in **Section 3**; [TUNING.md](TUNING.md) and notebook **Section 12** explain what to change. Change manual settings under a **new `RUN_TAG`**. Identical settings/tag resume optimizer/scheduler/scaler/RNG and the exact next batch. A fresh tag starts from Phase 1, not the last completed phase's classifier. Do not extend a completed schedule just by changing its epoch count under the same tag.

## Mandatory KD

`lightweight_kd` retains alternating Phase 1 lightweight layers and trains the resulting smaller generator. The teacher is a **larger, completed Phase 6 model evaluated on the full held-out split**. The default selects the highest eligible held-out BERTScore; `TEACHER_RUN_DIR` can specify one. Its identity/signature is pinned for student resumption, even if better teachers finish later.

Online token-KL distillation compares teacher/student next-token distributions under identical gold reference prefixes, masking padding, and combines that loss with supervised summary loss. Both use the same Phase 1 BPE. Source representation can differ between teacher and student; no validation/test references become training targets. The teacher is frozen. This mode does not generate a hidden pseudo-summary cache or invoke an external model. Expect additional compute/GPU memory; reduce batch size and increase accumulation as needed.

## Metrics and result verification

Required metrics are **BERTScore F1**, **ROUGE-1 F1** and **ROUGE-L F1**. The PDF mentions RoBERTa-large in scorer troubleshooting, but does not publish all evaluator settings. Explicit defaults are RoBERTa-large layer 17, no IDF, no baseline rescaling, and stemmed ROUGE. Settings and the [BERTScore reproducibility hash](https://github.com/Tiiiger/bert_score) are saved. Match a subsequently published official scorer; local/private scores are not guaranteed to match. ROUGE uses mean per-document F1 from `rouge-score`.

**RoBERTa is evaluation only.** Its pretrained weights never initialize the summarizer, teacher or adaptation layers and never supply training targets. Its download/cache is separate from Phase 1. BERTScore failures stop scoring rather than silently selecting checkpoints with ROUGE. Default scoring runs on CPU to preserve training GPU memory; change `BERTSCORE_DEVICE='cuda'` if capacity permits. Long generated/reference summaries may be truncated by the metric tokenizer; record consistent metric and length settings.

Default epoch selection generates a fixed 128-document held-out subset. After training, the selected checkpoint is strictly reloaded, its selected summaries must match, and BERTScore/ROUGE are recomputed. Final evaluation uses **every held-out document** when `FINAL_VALIDATION_LIMIT=0`; subset and full scores are distinct artifacts. JSON is reread and rescored. Every test document ID/type, nonempty summary, JSON count and ZIP entry is checked.

The full held-out set includes selection cases and is not an untouched independent test. Repeated tuning can overestimate generalization. Generation metrics cannot establish factual accuracy: manually inspect entities, dates, statutes, legal reasoning and holdings. No private test score is fabricated. Actual Codabench results must come from uploading your ZIP.

## Saved outputs

```text
<experiment>_<RUN_TAG>/
```

Each run saves configuration, model/teacher metadata, source/target coverage, `last.pt`, `best_model.pt`, history, fixed-selection manifest, selected validation predictions/scores, final `metrics.json`, full `validation_prediction.json`, generation settings, curves, verification JSON and submission files. Token caches, loading/split manifests, `experiment_summary.csv`, `coverage.json` and `best_per_architecture.json` live at the project root.

Submission format:

```json
[{"id": "doc_001", "summary": "Generated legal summary ..."}]
```

`submission.zip` contains only `prediction.json` at its root. The optional decoding-search output is `<experiment>__decoding_<tag>/`. Retain required architecture-specific and compulsory-method submissions, even if another model has the highest score. There is no classification-style probability ensemble of generated summaries in this notebook.

## Offline validation

Build with `python Phase6/build_phase6_notebook.py`. Test with `python Phase6/validate_phase6_notebook.py` in an environment containing torch, tokenizers, numpy, nbformat, matplotlib, tqdm, huggingface_hub, transformers, bert-score and rouge-score.

The original local Phase 1 checkpoints were removed during disk cleanup. Tests create small compatible checkpoints for all four original classes, verify strict loading and computation equivalence, then exercise every profile and complete synthetic pipelines. The real BERTScore API is tested with a tiny randomly initialized local metric model; its scores are **not task performance**. Actual trained checkpoints, RoBERTa-large scoring, official file contents and Colab CUDA execution remain runtime checks. See [VALIDATION.md](VALIDATION.md).

## Saved run results

[Browse Phase 6 results](../results/phase6/) for six trained runs, saved settings, metric comparisons, coverage limits, plots and verification. Historical training commits and Codabench scores/IDs remain unavailable. See the [experiment ledger](../experiments/runs.csv).

Phase 6 checkpoints were selected on 128 held-out documents; final scores cover all 351. The saved scorer configuration and exact split are recorded with the results.

Saved result files are included in `results/phase6/`. the exported configurations, metrics, histories, and plots are committed in the repository.

## Official result identification

The [Phase 6 result index](../results/phase6/) records submission 1022146, displayed official BERTScore F1 0.8361, reported rank 76, and 6 author-confirmed distinct valid submissions. Exact matching identifies `decoder_lora_v1`: decoder-only architecture, LoRA, tag `v1`. Both saved prediction artifacts match all 100 submitted summaries among 12 compared files without errors. The configuration checksum matches locally; see the [match evidence](../results/phase6/best_submission_match.json).
