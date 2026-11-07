SEED = 42
EXPERIMENT = 'encoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
RUN_DECODING_SEARCH = False
RUN_TAG = 'v1'
RESUME = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
VALIDATION_FRACTION = 0.05
EXCLUDE_TRAIN_TEST_OVERLAP = True
INVALID_TRAIN_ROW_POLICY = 'exclude'  # exclude unusable labeled rows with an audit, or error
MAX_INVALID_TRAIN_FRACTION = 0.05  # Stop if >5% are unusable; inspect schema before proceeding.
EPOCHS = 4
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SOURCE_TOKENS = 640
MAX_TARGET_TOKENS = 256  # Reference budget includes its terminal EOS.
SOURCE_POLICY = 'head_tail'  # head, tail, head_tail, spread, sentence_mmr
ENCODER_CHUNK_SIZE = 0  # Native encoder: 0=joint source; e.g. 384=encode windows independently.
INSTRUCTION = 'Summarize the legal facts, arguments, reasoning and decision.\nDocument:\n'
SUMMARY_PREFIX = '\nLegal summary:\n'
LEARNING_RATE = None  # None uses the profile LR. Set a number for manual tuning under a NEW tag.
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
LABEL_SMOOTHING = 0.1
GRADIENT_CHECKPOINTING = True
EARLY_STOPPING_PATIENCE = 2
SAVE_EVERY_STEPS = 200
MAX_OPTIMIZER_STEPS = None  # Smoke test only; capped runs are marked incomplete.
RANK = 8
ALPHA = 16.0
BOTTLENECK_DIM = 64
PROMPT_TOKENS = 16
PARTIAL_LAYERS = 2
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0
TEACHER_RUN_DIR = None  # A completed, larger Phase 6 teacher; None selects best eligible saved run.
GENERATION = {'num_beams': 2, 'max_new_tokens': 256, 'min_new_tokens': 24,
              'length_penalty': 1.0, 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 3,
              'deduplicate_sentences': False, 'long_document_mode': 'select', 'map_reduce_max_chunks': 4}
# Fixed deterministic subset for epoch selection, not a substitute for final full validation.
EPOCH_VALIDATION_LIMIT = 128  # 0 evaluates the entire held-out split at every epoch.
FINAL_VALIDATION_LIMIT = 0  # Keep 0 for real runs: score ALL held-out examples after training.
EVAL_LOSS_LIMIT = 256  # 0=all held-out examples for teacher-forced loss diagnostics.
BERTSCORE_MODEL = 'roberta-large'  # EVALUATION ONLY. Never used by the summarizer or teacher.
BERTSCORE_LAYERS = 17
BERTSCORE_RESCALE = False
BERTSCORE_BATCH_SIZE = 8
BERTSCORE_DEVICE = 'cpu'  # 'cuda' is faster but shares GPU memory with the summarizer.
ROUGE_USE_STEMMER = True
PHASE1_HEADS = {'decoder_only': 12, 'encoder_decoder': 8, 'hybrid_prefixlm': 10, 'lightweight': 12}
CORE_EXPERIMENTS = {
    'encoder_full': {'architecture': 'encoder_decoder', 'method': 'full', 'lr': 2e-5},
    'decoder_lora': {'architecture': 'decoder_only', 'method': 'lora', 'lr': 2e-4},
    'hybrid_lorma': {'architecture': 'hybrid_prefixlm', 'method': 'lorma', 'lr': 2e-4},
    'encoder_adapters': {'architecture': 'encoder_decoder', 'method': 'houlsby', 'lr': 2e-4},
    'lightweight_prompt': {'architecture': 'lightweight', 'method': 'prompt', 'lr': 5e-3},
    'lightweight_kd': {'architecture': 'lightweight', 'method': 'kd', 'lr': 2e-5},
}
EXTRA_EXPERIMENTS = {
    'decoder_full': {'architecture': 'decoder_only', 'method': 'full', 'lr': 2e-5},
    'encoder_lora': {'architecture': 'encoder_decoder', 'method': 'lora', 'lr': 2e-4},
    'encoder_lorma': {'architecture': 'encoder_decoder', 'method': 'lorma', 'lr': 2e-4},
    'encoder_partial': {'architecture': 'encoder_decoder', 'method': 'partial', 'lr': 5e-5},
    'hybrid_lora': {'architecture': 'hybrid_prefixlm', 'method': 'lora', 'lr': 2e-4},
    'hybrid_adapters': {'architecture': 'hybrid_prefixlm', 'method': 'houlsby', 'lr': 2e-4},
    'lightweight_lorma': {'architecture': 'lightweight', 'method': 'lorma', 'lr': 2e-4},
    'lightweight_lora': {'architecture': 'lightweight', 'method': 'lora', 'lr': 2e-4},
    'lightweight_pfeiffer': {'architecture': 'lightweight', 'method': 'pfeiffer', 'lr': 2e-4},
    'lightweight_full': {'architecture': 'lightweight', 'method': 'full', 'lr': 2e-5},
}
EXPERIMENTS = {**CORE_EXPERIMENTS, **EXTRA_EXPERIMENTS}
TUNE_BASE_EXPERIMENT = 'encoder_full'
TUNING_TRIALS = [
    {'name': 'lr_low', 'overrides': {'lr': 1e-5}},
    {'name': 'long_source', 'overrides': {'max_source_tokens': 896}},
    {'name': 'long_target', 'overrides': {'max_target_tokens': 384}},
    {'name': 'spread', 'overrides': {'source_policy': 'spread'}},
    {'name': 'mmr_context', 'overrides': {'source_policy': 'sentence_mmr'}},
    {'name': 'head_only', 'overrides': {'source_policy': 'head'}},
    {'name': 'no_smoothing', 'overrides': {'label_smoothing': 0.0}},
    {'name': 'independent_encoder_chunks', 'overrides': {'encoder_chunk_size': 384}},
]
DECODING_CANDIDATES = [
    {'name': 'greedy', 'overrides': {'num_beams': 1}},
    {'name': 'beam3', 'overrides': {'num_beams': 3}},
    {'name': 'longer', 'overrides': {'max_new_tokens': 384}},
    {'name': 'shorter', 'overrides': {'max_new_tokens': 160}},
    {'name': 'deduplicate', 'overrides': {'deduplicate_sentences': True}},
    {'name': 'map_reduce', 'overrides': {'long_document_mode': 'map_reduce'}},
]
assert EXPERIMENT in EXPERIMENTS and TUNE_BASE_EXPERIMENT in EXPERIMENTS
assert 0 < VALIDATION_FRACTION < 1 and EPOCHS > 0 and BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0
assert MAX_OPTIMIZER_STEPS is None or MAX_OPTIMIZER_STEPS > 0
assert 0 <= LABEL_SMOOTHING < 1 and 0 <= KD_WEIGHT <= 1 and KD_TEMPERATURE > 0
seed_all(SEED)
