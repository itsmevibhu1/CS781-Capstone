# CS781: Legal NLP Capstone

Individual capstone work by Vibhu Yadav (231145). The capstone entry points are listed below.

## Phase entry points

| Phase | Task | Main notebook | Documentation |
| --- | --- | --- | --- |
| 1 | Legal language model pretraining | [Phase 1 notebook](Phase1/CS781_Phase1_Solution_Colab.ipynb) | [Phase 1 README](Phase1/README.md) |
| 2 | Legal named entity recognition | [Phase 2 notebook](Phase2/CS781_Phase2_Solution_Colab.ipynb) | [Phase 2 README](Phase2/README.md) |
| 3 | Rhetorical role prediction | [Phase 3 notebook](Phase3/CS781_Phase3_Solution_Colab.ipynb) | [Phase 3 README](Phase3/README.md) |
| 4 | Court judgment prediction | [Phase 4 notebook](Phase4/CS781_Phase4_Solution_Colab.ipynb) | [Phase 4 README](Phase4/README.md) |
| 5 | Legal statute identification | [Phase 5 notebook](Phase5/CS781_Phase5_Solution_Colab.ipynb) | [Phase 5 README](Phase5/README.md) |
| 6 | Abstractive legal summarization | [Phase 6 notebook](Phase6/CS781_Phase6_Solution_Colab.ipynb) | [Phase 6 README](Phase6/README.md) |
| 7 | Prior case retrieval | [Phase 7 notebook](Phase7/CS781_Phase7_Solution_Colab.ipynb) | [Phase 7 README](Phase7/README.md) |

Phase 1 implements decoder-only, encoder-decoder, hybrid Prefix-LM, and lightweight architectures trained from scratch. Phases 2–7 adapt Phase 1 checkpoints using the methods documented in their READMEs. The available code is not evidence that every experiment has finished or been accepted on Codabench.

## Running the notebooks

1. Open the relevant main notebook in Google Colab and choose a GPU runtime.
2. Supply a Hugging Face token with access to the official course dataset through the authentication mechanism in that notebook. Never commit tokens.
3. Mount Google Drive and confirm the configured input/output paths.
4. Select the experiment and run tag, then execute the notebook in order. Follow the phase README for experiment-specific settings.
5. Preserve each run's configuration, metrics, checkpoint, and submission. Record actual Codabench submission IDs and scores separately from development scores.

Install dependencies using each notebook's installation cell. A locked environment for the completed experiments has not yet been recorded.

## Artifact storage

Large artifacts remain in Google Drive:

| Drive path relative to MyDrive | Purpose |
| --- | --- |
| `cs781_phase1/` | Phase 1 tokenizer, architecture run folders, and pretrained checkpoints |
| `cs781_phase2/` | Phase 2 experiment outputs |
| `cs781_phase3/` | Phase 3 experiment outputs |
| `cs781_phase4/` | Phase 4 experiment outputs |
| `cs781_phase5/` | Phase 5 experiment outputs |
| `cs781_phase6/` | Phase 6 outputs in the user’s Drive |
| `cs781_phase7/` | Phase 7 outputs in the friend’s Drive; not yet copied into the user’s account |

The shared local tokenizer files are in [bpe_tokenizer/](bpe_tokenizer/). Phases 2–7 expect the Phase 1 tokenizer and checkpoints in Drive; cloning this repository alone does not provide the weights. Expected checkpoint paths for the documented Phase 1–7 folders are recorded in the experiment ledger. Drive access instructions and checkpoint hashes still need to be supplied.

Checkpoints, datasets, prediction archives, and executed notebook copies are excluded from new Git additions. Small metric/configuration files and selected plots can be copied into `results/phaseN/<experiment>_<run_tag>/` when available. Review files for credentials and dataset text before adding them.

## Experiment records and results

[experiments/runs.csv](experiments/runs.csv) is the ledger for actual runs and submissions. Fill it from saved run artifacts and Codabench records, rather than reconstructing scores from memory. Use repository-relative paths for configurations and exact Drive paths for checkpoints. Record the code commit used when known; leave unavailable information blank.

Keep development and Codabench scores separate. Development scores used for checkpoint or decoder selection are selection scores. Synthetic validation metrics do not measure capstone performance.

## Supporting code and validation

`Phase2/` through `Phase7/` contain notebook builders and offline validation scripts. Builders reuse earlier phase notebook definitions and expect the phase folder layout in this repository. Phase 5 also includes source modules embedded into its standalone notebook. Validation scripts also require local Phase 1 weights and the dependencies listed in the phase READMEs.

Existing validation notes: [Phase 2](Phase2/VALIDATION.md), [Phase 3](Phase3/VALIDATION.md), [Phase 4](Phase4/VALIDATION.md), [Phase 5](Phase5/VALIDATION.md), [Phase 6](Phase6/VALIDATION.md), [Phase 7](Phase7/VALIDATION.md). These describe prior offline checks and their limits; they do not establish current Colab training results or Codabench acceptance.

## Remaining documentation

- Inventory completed experiments and accepted submissions for each phase.
- Recover missing Phase 1 historical configurations and metrics; expand analysis of the documented experiments and add later phases.
- Record exact checkpoint locations and reproducible environment versions.
- Preserve the code/configuration associated with each submission, including earlier versions when available.
- Add later phases and the LaTeX report as work progresses.
- Cite papers and reused code, and complete the course-required LLM declaration.

Keep competition solutions private and provide instructor access according to course instructions.

## Results

[Browse results by phase](results/). Results currently cover four Phase 1 plot folders, 14 Phase 2 runs, and eight Phase 3 runs. Phase 1 historical settings and scores remain unrecorded; Phase 2 and Phase 3 include saved configurations and selection metrics. Each phase has its own experiment index; model weights remain in Drive.

Phase 4 and Phase 5 each contain six documented runs with saved configurations, development scores, coverage, plots, and verification. See [Phase 4 results](results/phase4/) and [Phase 5 results](results/phase5/).

Phase 6 includes six trained summarization runs with final evaluation on all 351 held-out documents. Phase 7 includes six trained retrieval runs and a separate BM25 baseline; its artifacts were exported from the friend’s Drive. See [Phase 6 results](results/phase6/) and [Phase 7 results](results/phase7/).
