# CS781: Legal NLP Capstone

Individual capstone work by Vibhu Yadav (231145). The capstone entry points are listed below.

## Phase entry points

| Phase | Task | Main notebook | Documentation |
| --- | --- | --- | --- |
| 1 | Legal language model pretraining | [Phase 1 notebook](CS781_Phase1_Solution_Colab.ipynb) | [Phase 1 README](Phase1/README.md) |
| 2 | Legal named entity recognition | [Phase 2 notebook](CS781_Phase2_Solution_Colab.ipynb) | [Phase 2 README](Phase2/README.md) |
| 3 | Rhetorical role prediction | [Phase 3 notebook](CS781_Phase3_Solution_Colab.ipynb) | [Phase 3 README](Phase3/README.md) |

Phase 1 implements decoder-only, encoder-decoder, hybrid Prefix-LM, and lightweight architectures trained from scratch. Phases 2 and 3 adapt Phase 1 checkpoints using the methods documented in their READMEs. The available code is not evidence that every experiment has finished or been accepted on Codabench.

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

The shared local tokenizer files are in [bpe_tokenizer/](bpe_tokenizer/). Phase 2 and Phase 3 expect the Phase 1 tokenizer and checkpoints in Drive; cloning this repository alone does not provide the weights. Exact artifact paths and access instructions still need to be recorded for each completed run.

Checkpoints, datasets, prediction archives, and executed notebook copies are excluded from new Git additions. Small metric/configuration files and selected plots can be copied into `results/phaseN/<experiment>_<run_tag>/` when available. Review files for credentials and dataset text before adding them.

## Experiment records and results

[experiments/runs.csv](experiments/runs.csv) is the ledger for actual runs and submissions. Fill it from saved run artifacts and Codabench records, rather than reconstructing scores from memory. Use repository-relative paths for configurations and exact Drive paths for checkpoints. Record the code commit used when known; leave unavailable information blank.

Keep development and Codabench scores separate. Development scores used for checkpoint or decoder selection are selection scores. Synthetic validation metrics do not measure capstone performance.

## Supporting code and validation

`Phase2/` and `Phase3/` contain notebook builders and offline validation scripts. Run builders from this repository layout because they reuse the root Phase 1 notebook. Validation scripts also require local Phase 1 weights and the dependencies listed in the phase READMEs.

Existing validation notes: [Phase 2](Phase2/VALIDATION.md), [Phase 3](Phase3/VALIDATION.md). These describe prior offline checks and their limits; they do not establish current Colab training results or Codabench acceptance.

## Remaining documentation

- Inventory completed experiments and accepted submissions for each phase.
- Add actual run configurations, result tables, plots, and analysis.
- Record exact checkpoint locations and reproducible environment versions.
- Preserve the code/configuration associated with each submission, including earlier versions when available.
- Add later phases and the LaTeX report as work progresses.
- Cite papers and reused code, and complete the course-required LLM declaration.

Keep competition solutions private and provide instructor access according to course instructions.

## Results

[Browse results by phase](results/). Each phase contains an index of its documented experiments.
