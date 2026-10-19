# Run Phase 3 in Colab

Upload [CS781_Phase3_Solution_Colab.ipynb](../CS781_Phase3_Solution_Colab.ipynb) to Colab, select a GPU runtime, add a secret named **HF** containing only your `hf_...` access token, and run all cells. The token is never printed.

The notebook mounts Drive, reads **your Phase 1** tokenizer/weights from `MyDrive/cs781_phase1/`, and saves to `MyDrive/cs781_phase3/`. It does not use Phase 2 weights: the Phase 3 document requires every experiment to start from Phase 1.

## Data usage

All seven official files are downloaded: CL/IT train, CL/IT dev, CL/IT test, and `label_vocab.json`.

- **Train:** fit the model, loss weights, and per-domain transition/class priors.
- **Dev:** choose checkpoints, tune postprocessing, show combined and CL/IT macro-F1, precision, and recall.
- **Test:** generate one valid label for every sentence in original order, with the official IDs. Test gold labels are private; no test score is fabricated.

## Select experiments in Section 3

| EXPERIMENT | Phase 1 origin | Method |
|---|---|---|
| `decoder_full` | Decoder-only | Full fine-tuning |
| `encoder_ia3` | Encoder–decoder encoder | IA³ key/value and FF scaling |
| `hybrid_lora` | Hybrid Prefix-LM | LoRA |
| `lightweight_lorma` | Lightweight | LoRMA+ |
| `hybrid_rslora` | Hybrid Prefix-LM | Rank-stabilized LoRA |
| `lightweight_kd` | Lightweight, half its blocks | Supervised loss plus teacher-logit distillation |

For a first run:

```python
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_TAG = 'v1'
```

After confirming actual dev scores, keep the same settings/tag and use `RUN_ALL=True` to run the remaining experiments. The completed decoder training is reused. KD runs last and selects the strongest completed compatible larger teacher by dev macro-F1.

The PDF's general instructions require six submissions per phase. These methods also cover the four Phase 1 origins and compulsory LORMA/distillation requirements. Download and upload each distinct `submission.zip` to Codabench manually; local coverage does not prove submission acceptance.

## Tune in Section 3; change the run tag

Section **13** of the notebook provides a tuning table. Start with controlled changes:

- Compare `CONTEXT_SENTENCES=0`, `1`, `2` for sentence-only vs context-aware input.
- Compare `EPOCHS=6` vs `10` while retaining dev checkpoint selection and early stopping.
- Adjust `EXPERIMENTS[EXPERIMENT]['lr']` for the selected backbone/adapter, and `HEAD_LR_MULTIPLIER` for the new head.
- Compare class weighting on/off; inspect per-label support and errors, especially rare roles.
- Adjust sequence lengths only after reviewing the reported target truncation counts.

Use a fresh `RUN_TAG` for any changed training/decoding settings, e.g. `RUN_TAG='decoder_context2'`. Rerun Section 3 and every later cell. Do not change multiple variables in a comparison unless you deliberately test a combined configuration.

Resume uses the same settings and restores model, optimizer, schedule, AMP scaler, RNG and epoch/batch position. A changed epoch count starts a new schedule under a new tag, rather than continuing an exhausted one. KD teacher selection requires compatible run settings, so complete the architecture runs under a matching tag before that tag's KD run.

## Outputs and checks

Each run folder contains configuration, last/best checkpoints, history, greedy and selected dev metrics, per-label CSV, dev predictions/errors, training/confusion plots, decoder candidates/settings, test predictions, verification results, and a ZIP containing only `prediction.json` at its root.

The notebook verifies label spelling, IDs, one label per sentence, ordering, no duplicate/missing sentence predictions, finite logits/loss, train/dev/test separation, reloaded checkpoint scores, JSON round-trips and ZIP contents. It reports all thirteen official classes with zero F1 when a class has no true positives; domain scores are also saved to make aggregation explicit.

All score optimization uses dev labels. Repeated selection can overestimate private test performance. Actual performance requires your official dataset and Colab run; architecture/unit tests cannot establish a target F1.

Offline checks can be run with:

```bash
python Phase3/validate_phase3_notebook.py
```

The script requires PyTorch, tokenizers, numpy, tqdm, matplotlib, nbformat, and huggingface_hub. It uses your local Phase 1 checkpoints and temporary synthetic train/dev/test data; it does not authenticate or download gated course data. Paper references for LoRA, rsLoRA, LoRMA, IA³ and distillation are linked in the notebook. Keep the required LLM declaration and cite the reused code/references.
