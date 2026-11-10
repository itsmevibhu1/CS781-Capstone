# Phase 3 results

Rhetorical role prediction for CL and IT documents. Eight saved runs cover six experiment types and two context variants. Open a run for its method, configuration, detailed scores, plots, and saved verification.

| Run | Origin | Method | Context | Trainable parameters | Combined dev F1 | CL dev F1 | IT dev F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [decoder_full_v1](decoder_full_v1/) | Decoder-only | Full fine-tuning | 1 | 111,630,445 | 0.37446 | 0.32049 | 0.37896 |
| [encoder_ia3_v1](encoder_ia3_v1/) | Encoder-decoder encoder | IA³ | 1 | 1,239,661 | 0.27118 | 0.22775 | 0.27305 |
| [hybrid_lora_v1](hybrid_lora_v1/) | Hybrid Prefix-LM | LoRA | 1 | 1,214,829 | 0.34177 | 0.29273 | 0.34163 |
| [hybrid_rslora_v1](hybrid_rslora_v1/) | Hybrid Prefix-LM | Rank-stabilized LoRA | 1 | 1,214,829 | 0.34906 | 0.29052 | 0.36071 |
| [lightweight_kd_kd_context2](lightweight_kd_kd_context2/) | Lightweight | Knowledge distillation | 2 | 64,726,381 | 0.44106 | 0.32675 | 0.44510 |
| [lightweight_kd_v1](lightweight_kd_v1/) | Lightweight | Knowledge distillation | 1 | 64,726,381 | 0.37283 | 0.27489 | 0.38953 |
| [lightweight_lorma_kd_context2](lightweight_lorma_kd_context2/) | Lightweight | LoRMA+ | 2 | 1,580,653 | 0.41568 | 0.32894 | 0.39317 |
| [lightweight_lorma_v1](lightweight_lorma_v1/) | Lightweight | LoRMA+ | 1 | 1,580,653 | 0.39126 | 0.30634 | 0.39373 |

The highest recorded combined development macro-F1 is **0.44106**, from `lightweight_kd_kd_context2`. This identifies the best development selection score among the imported runs; private test performance is not recorded.

## Reading the comparisons

- Scores use the 0–1 scale. Combined macro-F1 pools both domains across 13 labels; it is not the mean of CL and IT macro-F1.
- Development labels selected checkpoints and decoders. These are selection scores; original training commits remain unrecorded. The official best submission is linked below, and the author confirmed eight distinct valid submissions; those confirmations are separate from saved local-dev metrics.
- The `v1` runs use context setting 1. The two `kd_context2` runs use setting 2. The LoRMA+ run with that tag is an adapted model; only the `lightweight_kd` experiment applies distillation.
- Architecture and adaptation both vary across several rows. Treat these as descriptive comparisons, not isolated estimates of adaptation effects. KD context variants also involve separately trained teachers. No repeated-seed uncertainty estimate is available.
- All saved verification files report training complete and passing artifact checks. Those reports do not replace official evaluator or Codabench acceptance evidence.

[Phase 3 implementation and instructions](../../Phase3/README.md) · [Experiment ledger](../../experiments/runs.csv) · [All results](../)

## Matched official best submission

Submission **992321**, reported rank **110**, has displayed **Macro-F1 0.4290**, **Macro-Precision 0.4887**, and **Macro-Recall 0.4309**. These are rounded four-decimal leaderboard values. The downloaded ZIP contains role lists for **10 documents and 2,197 sentences**.

[official_best.json](official_best.json) preserves the supplied metrics and reference fingerprints. **Matched run: `lightweight_kd_kd_context2`**, a lightweight student trained with knowledge distillation and context setting 2. The [match report](best_submission_match.json) compares 16 artifacts with no errors; both this run’s prediction JSON and ZIP match all 2,197 labels across 10 documents. The reported configuration checksum matches locally. The author confirmed **8 distinct valid submissions**; the rank-recording date remains unconfirmed. Local-dev scores in the comparison table are separate from the official test score.
