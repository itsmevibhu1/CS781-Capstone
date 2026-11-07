SEED = 42
EXPERIMENT = 'encoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
RUN_TAG = 'v1'
RESUME = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
EPOCHS = 6
QUERY_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 4
HARD_NEGATIVE_FRACTION = 0.75
MINE_DENSE_NEGATIVES = False  # Optional train-only mining after each completed epoch.
USE_IN_BATCH_NEGATIVES = True
CHUNK_LENGTH = 256
CHUNK_OVERLAP = 48
MAX_CHUNKS = 3  # 0 = all windows; expensive for long candidate pools.
CHUNK_BATCH_SIZE = 8
ENCODE_DOCUMENT_BATCH_SIZE = 8
CHUNK_POLICY = 'spread'  # spread, head, tail, head_tail
POOLING = 'attention'  # attention, mean, max, mean_max
EMBEDDING_DIM = 256
HEAD_DROPOUT = 0.1
SHARE_QUERY_CANDIDATE_PROJECTION = True
LOSS_NAME = 'multi_positive'  # multi_positive (uniform positives), pairwise, combined
TEMPERATURE = 0.05
PAIRWISE_MARGIN = 0.1
LEARNING_RATE = None
HEAD_LR_MULTIPLIER = 3.0
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CHECKPOINTING = True
RANK = 8
ALPHA = 16.0
PROMPT_TOKENS = 16
PARTIAL_LAYERS = 2
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0
TEACHER_RUN_DIR = None  # Completed, compatible, larger Phase 7 retrieval teacher.
SAVE_EVERY_STEPS = 50
MAX_OPTIMIZER_STEPS = None
BM25_K1 = 1.5
BM25_B = 0.75
BM25_MAX_FEATURES = 60000
BM25_STOP_WORDS = 'english'  # None preserves stop words.
HARD_NEGATIVE_DEPTH = 128
RRF_K = 60
RRF_DEPTH = 200
# All selected trained-model outputs retain a nonzero neural contribution.
DENSE_WEIGHTS = [1.0, 0.75, 0.5, 0.25]
SEARCH_LATE_INTERACTION = True
ENABLE_CITATION_FEATURE = False  # Exact provided-document IndianKanoon URL matches; no external lookup.
CITATION_WEIGHTS = [0.0, 0.01, 0.03]
EXCLUDE_SELF_MATCH = False  # PDF does not prohibit self IDs; enable only after inspecting gold/pools.
PHASE1_HEADS = {'decoder_only': 12, 'encoder_decoder': 8, 'hybrid_prefixlm': 10, 'lightweight': 12}
TOP_K = 7  # Submission contract; not a tunable hyperparameter.
CORE_EXPERIMENTS = {
    'encoder_full': {'architecture': 'encoder_decoder', 'method': 'full', 'lr': 2e-5},
    'decoder_partial': {'architecture': 'decoder_only', 'method': 'partial', 'lr': 5e-5},
    'hybrid_lorma': {'architecture': 'hybrid_prefixlm', 'method': 'lorma', 'lr': 2e-4},
    'lightweight_prompt': {'architecture': 'lightweight', 'method': 'prompt', 'lr': 2e-3},
    'lightweight_ia3': {'architecture': 'lightweight', 'method': 'ia3', 'lr': 2e-4},
    'lightweight_kd': {'architecture': 'lightweight', 'method': 'kd', 'lr': 2e-5},
}
EXTRA_EXPERIMENTS = {
    'decoder_full': {'architecture': 'decoder_only', 'method': 'full', 'lr': 2e-5},
    'encoder_partial': {'architecture': 'encoder_decoder', 'method': 'partial', 'lr': 5e-5},
    'encoder_lorma': {'architecture': 'encoder_decoder', 'method': 'lorma', 'lr': 2e-4},
    'encoder_ia3': {'architecture': 'encoder_decoder', 'method': 'ia3', 'lr': 2e-4},
    'decoder_ia3': {'architecture': 'decoder_only', 'method': 'ia3', 'lr': 2e-4},
    'hybrid_full': {'architecture': 'hybrid_prefixlm', 'method': 'full', 'lr': 2e-5},
    'lightweight_full': {'architecture': 'lightweight', 'method': 'full', 'lr': 2e-5},
    'lightweight_lorma': {'architecture': 'lightweight', 'method': 'lorma', 'lr': 2e-4},
}
EXPERIMENTS = {**CORE_EXPERIMENTS, **EXTRA_EXPERIMENTS}
TUNE_BASE_EXPERIMENT = 'hybrid_lorma'
TUNING_TRIALS = [
    {'name': 'lr_low', 'overrides': {'lr': 1e-4}},
    {'name': 'more_chunks', 'overrides': {'max_chunks': 6}},
    {'name': 'longer_chunks', 'overrides': {'chunk_length': 384}},
    {'name': 'mean_pooling', 'overrides': {'pooling': 'mean'}},
    {'name': 'mean_max', 'overrides': {'pooling': 'mean_max'}},
    {'name': 'head_tail', 'overrides': {'chunk_policy': 'head_tail'}},
    {'name': 'pairwise', 'overrides': {'loss_name': 'pairwise'}},
    {'name': 'combined_loss', 'overrides': {'loss_name': 'combined'}},
    {'name': 'more_negatives', 'overrides': {'negatives': 8}},
    {'name': 'asymmetric_heads', 'overrides': {'share_projection': False}},
    {'name': 'dense_mining', 'overrides': {'mine_dense_negatives': True}},
    {'name': 'warmer_scores', 'overrides': {'temperature': 0.1}},
]
assert EXPERIMENT in EXPERIMENTS and TUNE_BASE_EXPERIMENT in EXPERIMENTS
assert TOP_K == 7 and EPOCHS > 0 and QUERY_BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0
assert MAX_OPTIMIZER_STEPS is None or MAX_OPTIMIZER_STEPS > 0
assert all(0 < value <= 1 for value in DENSE_WEIGHTS)
assert 0 <= HARD_NEGATIVE_FRACTION <= 1 and 0 <= KD_WEIGHT <= 1 and KD_TEMPERATURE > 0
seed_all(SEED)
