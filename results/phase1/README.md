# Phase 1 results

Four architecture folders were exported. Each currently contains a loss plot only; historical run settings, numeric histories, evaluation metrics, and submission records are unavailable in this export.

| Run | Architecture indicated by folder name | Available artifact | Evaluation score |
| --- | --- | --- | --- |
| [run_decoder_only](run_decoder_only/) | Decoder-only Transformer | Loss curve | Not recorded |
| [run_encoder_decoder](run_encoder_decoder/) | Encoder-decoder Transformer | Loss curve | Not recorded |
| [run_hybrid_prefixlm](run_hybrid_prefixlm/) | Hybrid Prefix-LM | Loss curve | Not recorded |
| [run_lightweight](run_lightweight/) | Lightweight Transformer | Loss curve | Not recorded |

The [export manifest](export_manifest.json) lists missing artifacts. The [current Phase 1 implementation](../../Phase1/README.md) describes how to run the code; its defaults must not be presented as historical settings without confirmation. Models, predictions, and ZIPs remain in Drive. Training completion and Codabench acceptance cannot be established from these plots alone.

[Experiment ledger](../../experiments/runs.csv) · [All results](../)
