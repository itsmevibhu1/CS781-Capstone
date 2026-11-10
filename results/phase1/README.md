# Phase 1 results

Four architecture folders were exported. Each currently contains a loss plot only; historical run settings, numeric histories, evaluation metrics, and submission records are unavailable in this export.

| Run | Architecture indicated by folder name | Available artifact | Evaluation score |
| --- | --- | --- | --- |
| [run_decoder_only](run_decoder_only/) | Decoder-only Transformer | Loss curve | Not recorded |
| [run_encoder_decoder](run_encoder_decoder/) | Encoder-decoder Transformer | Loss curve | Not recorded |
| [run_hybrid_prefixlm](run_hybrid_prefixlm/) | Hybrid Prefix-LM | Loss curve | Not recorded |
| [run_lightweight](run_lightweight/) | Lightweight Transformer | Loss curve; exact official prediction match | BERTScore F1 **0.83094329** |

The [export manifest](export_manifest.json) lists missing artifacts. The [current Phase 1 implementation](../../Phase1/README.md) describes how to run the code; its defaults must not be presented as historical settings without confirmation. Models, predictions, and ZIPs are excluded from Git. Training completion and Codabench acceptance cannot be established from these plots alone.

[Experiment ledger](../../experiments/runs.csv) · [All results](../)

## Official best submission

The user supplied Codabench submission **962785** with **BERTScore F1 0.83094329**, **ROUGE-L F1 0.19069432**, **BLEU-4 0.03978785**, and **1,000 evaluated examples**. The corresponding reported leaderboard rank is **75**. The author confirmed **4 distinct valid submissions**; the rank-recording date remains unconfirmed.

[official_best.json](official_best.json) records these screenshot-transcribed metrics and the downloaded submission’s archive/prediction fingerprints. The ZIP itself stays outside Git. The prediction file contains 1,000 ID-to-text records and passes ZIP/JSON parsing checks.

**Matched model: lightweight Transformer, `run_lightweight`.** The [match report](best_submission_match.json) compares eight saved prediction artifacts without errors. Both `run_lightweight/prediction.json` and `run_lightweight/submission.zip` match all 1,000 official predictions exactly after ignoring JSON key order/serialization whitespace. These are two files from one run, not two competing model matches. No other compared run matched. The expected checkpoint is `run_lightweight/final_model.pt`; its weight checksum and the original training Git commit have not been independently verified.
