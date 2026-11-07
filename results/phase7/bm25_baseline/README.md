# Phase 7: BM25 baseline

This is a lexical retrieval baseline, not a trained Phase 1 adaptation or another completed neural experiment. Its export contains [metrics.json](metrics.json) only; model configuration, training history and neural verification reports are not expected here.

| Micro-F1@7 | Micro-precision@7 | Micro-recall@7 |
| ---: | ---: | ---: |
| 0.11895 | 0.12954 | 0.10997 |

Evaluated queries: 118; retrieved IDs: 826; gold relevance entries: 973; hits: 107. The metrics JSON also preserves per-query hit/error counts. These dev scores can be compared with the neural/fused policies, but are not private test scores.

The [current retrieval implementation](../../../Phase7/README.md) describes BM25 and rank fusion. A standalone historical BM25 config was not exported; current defaults should not be assumed to establish the historical baseline configuration. The baseline source folder belongs to the friend’s Drive account. No checkpoint or training-complete status is claimed and no Codabench submission is recorded.

[All Phase 7 results](../)
