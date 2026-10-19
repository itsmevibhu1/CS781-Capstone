# Result export tools

Upload [Export_Phases1_2_Results_Colab.ipynb](Export_Phases1_2_Results_Colab.ipynb) to Google Colab and run all cells in a CPU runtime. It reads `MyDrive/cs781_phase1/` and `MyDrive/cs781_phase2/`, and downloads one ZIP with small configuration, metric, history, environment, verification, and plot artifacts. Original run folder names are preserved under `results/phase1/` and `results/phase2/`.

No training is rerun and Drive files are not modified. Weights, predictions, datasets, caches, and submission ZIPs are excluded. Each phase has an export manifest. Missing files and source folders are reported; no settings or scores are invented. A Phase 1 export may contain only loss plots because the original notebook did not save independent configurations or metric files.

Extract the ZIP into the repository root. Replace the filename below with your actual download:

```bash
cd /home/vibhu/Documents/CS781-Capstone-clean
unzip -n ~/Downloads/phases1_2_results_YYYYMMDD_HHMMSS.zip -d .
git status --short
```

The `-n` option preserves existing local files. Review the inventories, create phase/run READMEs, and populate the experiment ledger before committing imported results. Codabench IDs and scores must be entered from actual competition records. Preserve the notebooks/configurations actually used for past runs when available.

For a mounted Drive or local copy, the standalone script uses only the Python standard library:

```bash
python tools/export_phases1_2_results.py --drive-root /path/to/MyDrive --output /tmp/phases1_2_results.zip
```

Pass `--phases 1` or `--phases 2` to export a single phase. Existing ZIPs are never replaced.
