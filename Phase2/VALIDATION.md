# Validation performed

Passed local checks using CPU PyTorch and the actual Phase 1 tokenizer and `final_model.pt` files in this workspace:

- Notebook JSON schema and every Python code cell compile.
- Strict loading of all four real Phase 1 checkpoints.
- Forward/backward passes for all six experiment options, with finite loss and classifier gradients.
- LoRA and LoRMA attention adapters start as identity updates, receive gradients, and match their merged-weight formulas.
- Right padding does not change predictions on the valid tokens of a document.
- Character-offset decoding round-trips a known COURT and DATE example, with Unicode present in the original text.
- Document-level BIO decoding preserves an entity across a window boundary.
- Strict metrics use exact spans/labels and include all twelve labels in macro averages.
- Constrained BIO Viterbi decoding agrees with an independent exhaustive search on a small sequence.
- A constructed validation example exercises selecting a decoder with better exact-span F1; real-data gains are not claimed.
- Reloaded best-checkpoint metrics match recorded scores before decoder selection.
- Post-training score tables, JSON reports, and the standalone ZIP verification section execute successfully.
- Small synthetic training, held-out evaluation, checkpoint creation/reload, knowledge distillation, prediction generation, and ZIP validation.
- A short final gradient-accumulation group trains correctly.
- Interrupted/resumed tiny-model training produces exactly the same final weights as uninterrupted training.

The official gated Phase 2 dataset was not downloaded in this environment. Authentication, actual dataset annotation compatibility, Colab GPU training, private leaderboard scores, and submission acceptance remain runtime checks. No full Phase 2 training or task-performance claim is made.

The repeatable validation script is `validate_phase2_notebook.py`. All temporary training artifacts from validation were written under `/tmp`, leaving the supplied Phase 1 weights unchanged.
