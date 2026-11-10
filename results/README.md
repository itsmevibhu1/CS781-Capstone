# Results

Browse by phase, then open a run for its saved evidence and documentation.

| Phase | Task | Recorded folders | Results |
| --- | --- | ---: | --- |
| 1 | Legal language model pretraining | 4 | [Phase 1 official result, plots, and evidence gaps](phase1/) |
| 2 | Legal named entity recognition | 14 | [Phase 2 settings and validation results](phase2/) |
| 3 | Rhetorical role prediction | 8 | [Phase 3 settings and development results](phase3/) |
| 4 | Court judgment prediction | 6 | [Phase 4 development results](phase4/) |
| 5 | Legal statute identification | 6 | [Phase 5 development results](phase5/) |
| 6 | Abstractive legal summarization | 6 | [Phase 6 held-out results](phase6/) |
| 7 | Prior case retrieval | 6 trained + 1 baseline | [Phase 7 development results](phase7/) |

Phase 1 has four loss-plot folders and an official best result attributed to `run_lightweight` by exact prediction matching. Achieved pretraining token totals and numeric histories remain unavailable. Phase 2 and Phase 3 include saved configurations and selection metrics. Scores from different tasks and split definitions are not directly comparable.

The [experiment ledger](../experiments/runs.csv) links folders to saved configurations where available and recorded run settings. Codabench scores, IDs, and acceptance are not yet recorded. Folder counts do not prove distinct accepted submissions.

Weights, prediction files, datasets, caches, and large archives are excluded from Git. Each phase index explains the completeness of its exported evidence.
