SEED = 42
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
BUILD_ENSEMBLE = False  # Enable after two or more runs have finished.
RUN_TAG = 'v1'
RESUME = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
EPOCHS = 6
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
CHUNK_LENGTH = 384
CHUNK_OVERLAP = 64
MAX_CHUNKS = 4  # 0 = every overlapping window; substantially more compute.
CHUNK_BATCH_SIZE = 4
TRAIN_CHUNK_POLICY = 'spread'
EVAL_CHUNK_POLICY = 'spread'
DOC_POOLING = 'attention'  # attention, mean, max, mean_max, transformer
DOC_HIDDEN = 128
HEAD_DROPOUT = 0.15
HEAD_LR_MULTIPLIER = 5.0
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 0.5
POS_WEIGHT_CAP = 20.0
LOSS_NAME = 'bce'  # bce, focal, asymmetric
FOCAL_GAMMA = 2.0
ASL_GAMMA_POS = 0.0
ASL_GAMMA_NEG = 4.0
ASL_CLIP = 0.05
LABEL_SMOOTHING = 0.0
INITIALIZE_LABEL_PRIORS = True
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CHECKPOINTING = True
ADAPTER_RANK = 8
ADAPTER_ALPHA = 16.0
BOTTLENECK_DIM = 64
PARTIAL_LAYERS = 2
UNFREEZE_LAYERS_PER_EPOCH = 2
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0
TEACHER_RUN_DIR = None  # Optional absolute Drive folder of a completed Phase 5 teacher.
SAVE_EVERY_STEPS = 100
MAX_OPTIMIZER_STEPS = None  # Use a new tag and a small cap only for smoke checks.
NUM_WORKERS = 0
TUNE_THRESHOLD = True
THRESHOLD_MODE = 'auto'  # global, per_label, auto (dev-select global vs shrunk per-label)
THRESHOLD_GRID = [round(float(x), 4) for x in np.linspace(0.02, 0.98, 49)] + [0.5]
MIN_DEV_POSITIVES = 3
THRESHOLD_SHRINKAGE = 10.0
ALLOW_EMPTY_PREDICTIONS = True
# PDF says the evaluator derives its dimension from reference data, but supplies no evaluator code.
# reference_range = IDs 0..max(gold ID); reference_present = observed gold IDs; known = full catalog.
# All three Macro-F1 variants are reported. Change only if the official evaluator clarifies its rule.
METRIC_LABEL_SCOPE = 'reference_range'
STATUTE_IDS = None  # Optional official valid-ID list; otherwise infer from TRAIN + DEV, never test labels.
SPLIT_OVERLAP_POLICY = 'exclude_train'  # exclude_train (default), error, report
PHASE1_HEADS = {'decoder_only': 12, 'encoder_decoder': 8, 'hybrid_prefixlm': 10, 'lightweight': 12}

# Four architectures, three adaptation methods plus compulsory KD. LoRMA is compulsory.
CORE_EXPERIMENTS = {
    'decoder_full': {'architecture': 'decoder_only', 'method': 'full', 'lr': 2e-5},
    'encoder_partial': {'architecture': 'encoder_decoder', 'method': 'partial', 'lr': 5e-5},
    'hybrid_lorma': {'architecture': 'hybrid_prefixlm', 'method': 'lorma', 'lr': 2e-4},
    'hybrid_adapters': {'architecture': 'hybrid_prefixlm', 'method': 'adapters', 'lr': 2e-4},
    'lightweight_lorma': {'architecture': 'lightweight', 'method': 'lorma', 'lr': 2e-4},
    'lightweight_kd': {'architecture': 'lightweight', 'method': 'kd', 'lr': 5e-5},
}
EXTRA_EXPERIMENTS = {
    'lightweight_full': {'architecture': 'lightweight', 'method': 'full', 'lr': 2e-5},
    'encoder_full': {'architecture': 'encoder_decoder', 'method': 'full', 'lr': 2e-5},
    'decoder_lorma': {'architecture': 'decoder_only', 'method': 'lorma', 'lr': 2e-4},
    'encoder_adapters': {'architecture': 'encoder_decoder', 'method': 'adapters', 'lr': 2e-4},
    'decoder_gradual': {'architecture': 'decoder_only', 'method': 'gradual', 'lr': 2e-5},
    'decoder_linear_probe': {'architecture': 'decoder_only', 'method': 'linear_probe', 'lr': 2e-4},
    'hybrid_full': {'architecture': 'hybrid_prefixlm', 'method': 'full', 'lr': 2e-5},
}
EXPERIMENTS = {**CORE_EXPERIMENTS, **EXTRA_EXPERIMENTS}
TUNE_BASE_EXPERIMENT = 'hybrid_lorma'  # Change to your strongest measured dev profile.
TUNING_TRIALS = [
    {'name': 'lr_low', 'overrides': {'lr': 1e-4}},
    {'name': 'more_chunks', 'overrides': {'max_chunks': 8}},
    {'name': 'longer_chunks', 'overrides': {'chunk_length': 512}},
    {'name': 'mean_max', 'overrides': {'pooling': 'mean_max'}},
    {'name': 'doc_transformer', 'overrides': {'pooling': 'transformer'}},
    {'name': 'head_tail', 'overrides': {'train_chunk_policy': 'head_tail', 'eval_chunk_policy': 'head_tail'}},
    {'name': 'random_chunks', 'overrides': {'train_chunk_policy': 'random_mix'}},
    {'name': 'no_pos_weights', 'overrides': {'use_class_weights': False}},
    {'name': 'focal', 'overrides': {'loss_name': 'focal'}},
    {'name': 'asymmetric', 'overrides': {'loss_name': 'asymmetric'}},
    {'name': 'global_threshold', 'overrides': {'threshold_mode': 'global'}},
    {'name': 'label_smoothing', 'overrides': {'label_smoothing': 0.02}},
    {'name': 'label_graph', 'overrides': {'label_graph_weight': 0.2}},
]
ENSEMBLE_TOP_K = 3
ENSEMBLE_PAIR_WEIGHTS = [0.25, 0.5, 0.75]
assert EXPERIMENT in EXPERIMENTS and TUNE_BASE_EXPERIMENT in EXPERIMENTS
assert EPOCHS > 0 and BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0 and SAVE_EVERY_STEPS > 0
assert 0 <= KD_WEIGHT <= 1 and KD_TEMPERATURE > 0
assert METRIC_LABEL_SCOPE in ('known', 'reference_present', 'reference_range')
assert MAX_OPTIMIZER_STEPS is None or MAX_OPTIMIZER_STEPS > 0
seed_all(SEED)
