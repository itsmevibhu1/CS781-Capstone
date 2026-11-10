# Phase 1: Legal language model pretraining

The primary entry point is [CS781_Phase1_Solution_Colab.ipynb](CS781_Phase1_Solution_Colab.ipynb). It contains data loading, BPE tokenization, four model definitions, training, generation, and submission packaging.

## Architecture options

Set `ARCHITECTURE` in the notebook's master configuration to one of:

| Value | Architecture |
| --- | --- |
| `decoder_only` | Causal decoder Transformer |
| `encoder_decoder` | Encoder-decoder Transformer |
| `hybrid_prefixlm` | Prefix-LM with bidirectional prefix attention |
| `lightweight` | Lightweight Transformer with grouped-query attention |

These models are defined for training from scratch. Inspect the configuration cell for dimensions, token budgets, source mixture, optimization, and generation settings. Current notebook defaults must not be assumed to describe every historical run.

## Run and storage

1. Upload the main notebook to Colab and select a GPU runtime.
2. Follow its Hugging Face authentication and storage setup cells.
3. Set the architecture and training configuration before running the remaining cells.
4. Preserve the trained tokenizer and model checkpoints for downstream phases.
5. Validate and upload each submission archive to Codabench; record acceptance and scores separately.

Use the configured project directory with `bpe_tokenizer/` and architecture-specific `run_<architecture>/` folders. The notebook writes `prediction.json` and `submission.zip` into each run folder. Large artifacts remain outside Git.

The local [bpe_tokenizer/](../bpe_tokenizer/) contains the saved vocabulary and merge files. Preserve the exact tokenizer paired with a checkpoint when reproducing a run.

## Results to document

Populate [the experiment ledger](../experiments/runs.csv) from actual artifacts: training settings, code version if available, checkpoint path, evaluation results, and Codabench submission IDs. Add selected loss plots and measured comparisons. Local model files alone do not establish leaderboard scores or submission acceptance.

`CS781_Phase1_FourArchitectures.ipynb` is an additional notebook retained in this folder. Before attributing a submitted result to either notebook, verify which version and settings produced that run.

## Saved run results

[Browse the four Phase 1 run folders](../results/phase1/). Only saved loss plots were exported. The run pages explain the missing historical configurations, numeric metrics, verification, and Codabench records; no historical settings were inferred from current notebook defaults.

## Official result identification

The [Phase 1 result index](../results/phase1/) now records official submission 962785 and its reported rank and metrics. An exact comparison of all 1,000 predictions identifies `run_lightweight` as the matching saved run: its JSON and ZIP both match the official submission among eight compared artifacts, with no errors. See the [matching evidence](../results/phase1/best_submission_match.json). Weight checksums and the original training commit remain unverified.
