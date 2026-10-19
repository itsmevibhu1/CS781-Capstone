"""Build a self-contained Colab notebook using the existing Phase 1 model definitions."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
phase1 = json.loads((ROOT / 'Phase1' / 'CS781_Phase1_Solution_Colab.ipynb').read_text())
cells = []
def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(s).strip().splitlines(True)})
def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': textwrap.dedent(s).strip().splitlines(True)})

md('''
# CS781 Phase 2 — Legal Named Entity Recognition

This notebook adapts **your saved Phase 1 weights** for legal NER. It follows the starter
notebook's authentication → data → model → training → evaluation → prediction → ZIP workflow.
It is self-contained: the compatible model definitions from your Phase 1 solution are included below.

**Colab:** select a GPU runtime, add a secret named `HF`, and run all cells. Drive is mounted
and all results go to `MyDrive/cs781_phase2`. Your existing `MyDrive/cs781_phase1` is the input.
Run one experiment per session (choose `EXPERIMENT` below), or opt into sequential runs.
The six options cover all four Phase 1 origins, five adaptation methods, and distillation.
Train the four architecture experiments before the KD run, which chooses a teacher using held-out F1.

Phase 2 (§8) uses strict character-span macro-F1. The general instructions (§1.4) require **six
non-identical submissions**, despite §8.2 describing four methodological experiments. Creating ZIPs
here does not submit them to Codabench. Keep your course-required LLM declaration and cite sources.
The defaults use class-balanced training and a faster-learning new classification head. After training,
a small validation-only search compares greedy decoding with legal BIO Viterbi paths and entity-score
calibration. This keeps the baseline when none improves measured validation F1. It cannot guarantee
a private leaderboard improvement.
''')
md('## 1. Install runtime dependencies')
code('''
%pip -q install "tokenizers>=0.20,<0.24" "huggingface_hub>=0.26,<3" "tqdm>=4.66" "numpy>=1.26" "matplotlib>=3.8"
''')
md('## 2. Mount Drive, authenticate, and select the device')
code('''
import os, json, math, random, time, zipfile, hashlib, gc, copy, getpass, platform
from pathlib import Path
from contextlib import nullcontext
from collections import Counter
from bisect import bisect_left, bisect_right
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

try:
    from google.colab import drive, userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    BASE_DIR = Path('/content/drive/MyDrive')
else:
    BASE_DIR = Path.cwd()

PHASE1_DIR = BASE_DIR / 'cs781_phase1' if IN_COLAB else BASE_DIR
PROJECT_DIR = BASE_DIR / 'cs781_phase2'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(PROJECT_DIR / 'hf_cache')
from huggingface_hub import hf_hub_download, login

HF_TOKEN = os.environ.get('HF_TOKEN')
if IN_COLAB and not HF_TOKEN:
    try:
        HF_TOKEN = userdata.get('HF')
    except Exception:
        pass
def validate_hf_token(value):
    # Never print or include the credential in an error message.
    if not isinstance(value, str):
        raise TypeError('HF token must be a plain text string, not ' + type(value).__name__ + '.')
    value = value.strip()
    if not value.startswith('hf_'):
        raise ValueError('Paste only the Hugging Face personal access token starting with hf_.')
    return value

if HF_TOKEN:
    try:
        HF_TOKEN = validate_hf_token(HF_TOKEN)
    except (TypeError, ValueError) as error:
        print(str(error))
        HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass('Paste only your hf_ token (hidden): ')
HF_TOKEN = validate_hf_token(HF_TOKEN)
login(token=HF_TOKEN, add_to_git_credential=False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_DTYPE = torch.bfloat16 if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
print('Device:', DEVICE, '| PyTorch:', torch.__version__)
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
else:
    print('CPU is suitable for checks; select a Colab GPU before full training.')
print('Phase 1 input:', PHASE1_DIR)
print('Phase 2 output:', PROJECT_DIR)

# Atomic writes avoid leaving partially written JSON/checkpoints after interruption.
def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def save_torch(data, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)

def amp_context():
    return torch.autocast('cuda', dtype=AMP_DTYPE) if DEVICE.type == 'cuda' else nullcontext()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
''')
md('''
## 3. Experiment configuration

`EXPERIMENT` selects one run. Leave `RUN_ALL=False` for independent sessions; set it to `True`
to run the six options in order. Rerunning the training cell resumes `last.pt`, including optimizer,
scheduler, AMP scaler, and random generator states. Completed runs reuse `best_model.pt`.
Use a new `RUN_TAG` for a new experiment with changed settings. A config mismatch stops resumption.
`MAX_OPTIMIZER_STEPS` is an optional smoke-test cap; capped runs are explicitly marked incomplete.

The encoder–decoder run uses its pretrained **encoder** for token classification; unused decoder
and language-model heads are removed after strict Phase 1 loading. The hybrid uses its existing
bidirectional prefix mode for the whole input. Decoder-only and lightweight keep causal attention.
The KD student starts from Phase 1 lightweight weights and keeps every second block (six of twelve
with your defaults); the trained teacher is larger than the student.
''')
code('''
SEED = 42
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_TAG = 'v2'
RESUME = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
VALIDATION_FRACTION = 0.15
MAX_LENGTH = 384
WINDOW_OVERLAP = 96
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
WEIGHT_DECAY = 0.01
USE_CLASS_WEIGHTS = True
O_LOSS_WEIGHT = 0.35
MAX_ENTITY_CLASS_WEIGHT = 3.0
HEAD_LR_MULTIPLIER = 5.0
OPTIMIZE_DECODING = True
# Small validation-only search: keep the uncalibrated greedy baseline as a candidate.
DECODER_ENTITY_BIASES = [-0.3, 0.0, 0.3]
WARMUP_FRACTION = 0.1
MAX_OPTIMIZER_STEPS = None
SAVE_EVERY_STEPS = 100
GRADIENT_CHECKPOINTING = True
ADAPTER_RANK = 8
ADAPTER_ALPHA = 8.0
HEAD_DROPOUT = 0.1
KD_TEMPERATURE = 2.0
KD_WEIGHT = 0.5
NUM_WORKERS = 0  # Avoid Drive multiprocessing problems in Colab.

# Head counts for rotary/causal models are not stored in state_dict.
# These MUST match the settings actually used in your Phase 1 runs.
PHASE1_HEADS = {'decoder_only': 12, 'encoder_decoder': 8,
                'hybrid_prefixlm': 10, 'lightweight': 12}

EXPERIMENTS = {
    'decoder_full': {'architecture': 'decoder_only', 'method': 'full', 'lr': 2e-5},
    'encoder_partial': {'architecture': 'encoder_decoder', 'method': 'partial', 'lr': 5e-5},
    'hybrid_lora': {'architecture': 'hybrid_prefixlm', 'method': 'lora', 'lr': 2e-4},
    'lightweight_lorma': {'architecture': 'lightweight', 'method': 'lorma', 'lr': 2e-4},
    'decoder_bitfit': {'architecture': 'decoder_only', 'method': 'bitfit', 'lr': 2e-4},
    'lightweight_kd': {'architecture': 'lightweight', 'method': 'kd', 'lr': 5e-5},
}
TEACHER_CANDIDATES = ['decoder_full', 'encoder_partial', 'hybrid_lora', 'lightweight_lorma']
# Optional explicit teacher; None chooses a completed candidate by held-out strict macro-F1.
TEACHER_EXPERIMENT = None
assert EXPERIMENT in EXPERIMENTS
assert 0 < WINDOW_OVERLAP < MAX_LENGTH
assert BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0 and EPOCHS > 0
assert ADAPTER_RANK > 0 and SAVE_EVERY_STEPS > 0
assert O_LOSS_WEIGHT > 0 and MAX_ENTITY_CLASS_WEIGHT >= 1 and HEAD_LR_MULTIPLIER > 0
assert MAX_OPTIMIZER_STEPS is None or MAX_OPTIMIZER_STEPS > 0
assert 0 < VALIDATION_FRACTION < 1
assert 0 <= KD_WEIGHT <= 1 and KD_TEMPERATURE > 0
seed_all(SEED)
''')
md('## 4. Download and validate the official documents and character annotations')
code('''
FILES = {
    'fold1_text': 'Phase_2_LNER/Train/fold1-text.json',
    'fold1_gold': 'Phase_2_LNER/Train/fold1-gold.json',
    'fold2_text': 'Phase_2_LNER/Train/fold2-text.json',
    'fold2_gold': 'Phase_2_LNER/Train/fold2-gold.json',
    'labels': 'Phase_2_LNER/Train/labels.txt',
    'test_text': 'Phase_2_LNER/Test/fold3-text.json',
}
local_files = {}
for key, filename in tqdm(FILES.items(), desc='Downloading dataset', unit='file'):
    local_files[key] = Path(hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename,
                                         repo_type='dataset', token=HF_TOKEN))

EXPECTED_LABELS = ['APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT',
                   'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO']
ENTITY_LABELS = local_files['labels'].read_text(encoding='utf-8-sig').split()
if set(ENTITY_LABELS) != set(EXPECTED_LABELS) or len(ENTITY_LABELS) != 12:
    raise ValueError(f'Unexpected official labels.txt: {ENTITY_LABELS}. Inspect this file before training.')
BIO_LABELS = ['O'] + [prefix + '-' + label for label in ENTITY_LABELS for prefix in ['B', 'I']]
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(BIO_LABELS)

def read_documents(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a document-ID mapping.')
    result = {}
    for doc_id, record in data.items():
        text = record if isinstance(record, str) else record.get('text') if isinstance(record, dict) else None
        if not isinstance(text, str):
            raise ValueError(f'{path}, document {doc_id}: expected original text string.')
        result[str(doc_id)] = text  # No whitespace normalization: offsets must remain unchanged.
    return result

def normalize_gold(data, documents):
    if not isinstance(data, dict) or set(map(str, data)) != set(documents):
        raise ValueError('Gold/text IDs differ, or gold is not an ID mapping. Inspect the dataset schema.')
    result = {}
    for doc_id, records in data.items():
        doc_id = str(doc_id)
        text = documents[doc_id]
        if isinstance(records, dict):
            records = records.get('entities', records.get('annotations'))
        if not isinstance(records, list):
            raise ValueError(f'Unsupported annotations for {doc_id}: expected a list.')
        spans = []
        for entity in records:
            if isinstance(entity, (list, tuple)) and len(entity) in (3, 4):
                start, end, label = entity[:3]
                supplied_text = entity[3] if len(entity) == 4 else None
            elif isinstance(entity, dict):
                start = entity.get('start', entity.get('start_offset'))
                end = entity.get('end', entity.get('end_offset'))
                label = entity.get('label', entity.get('type'))
                supplied_text = entity.get('text', entity.get('entity_text'))
            else:
                raise ValueError(f'Unsupported entity schema in {doc_id}: {entity!r}')
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
                raise ValueError(f'Invalid character span: {doc_id}, {entity!r}')
            if label not in ENTITY_LABELS:
                raise ValueError(f'Unknown entity label: {label!r}')
            if supplied_text is not None and supplied_text != text[start:end]:
                raise ValueError(f'Entity text disagrees with end-exclusive offsets: {doc_id}, {entity!r}')
            spans.append((start, end, label))
        result[doc_id] = sorted(set(spans))
    return result

all_text, all_gold = {}, {}
for fold in [1, 2]:
    docs = read_documents(local_files[f'fold{fold}_text'])
    annotations = json.loads(local_files[f'fold{fold}_gold'].read_text(encoding='utf-8-sig'))
    gold = normalize_gold(annotations, docs)
    for doc_id in docs:
        if doc_id in all_text and (all_text[doc_id] != docs[doc_id] or all_gold[doc_id] != gold[doc_id]):
            raise ValueError(f'Conflicting duplicate document {doc_id} across folds.')
    all_text.update(docs)
    all_gold.update(gold)
test_text = read_documents(local_files['test_text'])
if set(all_text) & set(test_text):
    raise ValueError('Train/test document IDs overlap; investigate before training.')
if len(all_text) < 2:
    raise ValueError('Need at least two labeled documents for a held-out split.')

# Split by document text hash: duplicate texts cannot leak across train and validation.
groups = {}
for doc_id, text in all_text.items():
    groups.setdefault(hashlib.sha256(text.encode()).hexdigest(), []).append(doc_id)
group_keys = sorted(groups)
random.Random(SEED).shuffle(group_keys)
if len(group_keys) < 2:
    raise ValueError('Need at least two unique document texts.')
n_val = min(len(group_keys)-1, max(1, round(len(group_keys) * VALIDATION_FRACTION)))
val_ids = sorted(doc_id for key in group_keys[:n_val] for doc_id in groups[key])
train_ids = sorted(doc_id for key in group_keys[n_val:] for doc_id in groups[key])
DATA_FINGERPRINT = hashlib.sha256(json.dumps({'text': all_text, 'gold': all_gold,
    'test': test_text}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
SPLIT_INFO = {'seed': SEED, 'data_fingerprint': DATA_FINGERPRINT,
              'train_ids': train_ids, 'validation_ids': val_ids}
save_json(SPLIT_INFO, PROJECT_DIR / f'split_seed_{SEED}.json')
save_json({'entity_labels': ENTITY_LABELS, 'bio_labels': BIO_LABELS}, PROJECT_DIR / 'labels.json')
print('Training documents:', len(train_ids), '| Validation:', len(val_ids), '| Test:', len(test_text))
print('Training entity counts:', dict(Counter(s[2] for d in train_ids for s in all_gold[d])))
print('Validation entity counts:', dict(Counter(s[2] for d in val_ids for s in all_gold[d])))
''')
md('''
## 5. Reuse the Phase 1 BPE tokenizer and audit exact offsets

The tokenizer is loaded from Drive, never retrained. Its offset mappings reference the original
Unicode character string. Leading/trailing whitespace in byte-level tokens is trimmed only in the
**offset view**, not in the original text or token IDs. Entity spans must be end-exclusive.
BIO predicts flat entities; nested/overlapping gold entities cannot be represented and cause a clear
error rather than silently changing the labels. Unrepresentable subtoken boundaries are counted and
saved; their closest covering tokens are used for training, and their exact-span ceiling is reported.
''')
code('''
TOKENIZER_DIR = PHASE1_DIR / 'bpe_tokenizer'
for name in ['vocab.json', 'merges.txt']:
    if not (TOKENIZER_DIR / name).is_file():
        raise FileNotFoundError(f'Missing Phase 1 tokenizer: {TOKENIZER_DIR / name}')
tok = ByteLevelBPETokenizer(str(TOKENIZER_DIR / 'vocab.json'), str(TOKENIZER_DIR / 'merges.txt'))
PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID = [tok.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']]
if any(i is None for i in [PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID]):
    raise ValueError('Tokenizer special tokens do not match your Phase 1 solution.')
VOCAB_SIZE_ACTUAL = tok.get_vocab_size()
TOKENIZER_HASH = hashlib.sha256(b''.join((TOKENIZER_DIR / f).read_bytes() for f in ['vocab.json', 'merges.txt'])).hexdigest()

# Copy the exact tokenizer into Phase 2 for reproducible reloads.
import shutil
output_tokenizer_dir = PROJECT_DIR / 'bpe_tokenizer'
output_tokenizer_dir.mkdir(exist_ok=True)
for name in ['vocab.json', 'merges.txt']:
    shutil.copy2(TOKENIZER_DIR / name, output_tokenizer_dir / name)

def trim_offset(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end-1].isspace():
        end -= 1
    return start, end

def tokenize_document(text):
    encoding = tok.encode(text, add_special_tokens=False)
    ids = np.asarray(encoding.ids, dtype=np.int64)
    offsets = [trim_offset(text, int(s), int(e)) for s, e in encoding.offsets]
    if any(not (0 <= s <= e <= len(text)) for s, e in offsets):
        raise ValueError('Tokenizer offsets are not valid character offsets.')
    return {'ids': ids, 'offsets': offsets}

def align_entities(text, offsets, entities):
    labels = np.zeros(len(offsets), dtype=np.int64)
    occupied = np.zeros(len(offsets), dtype=bool)
    report = {'entities': len(entities), 'exact': 0, 'unalignable': []}
    prev_end = -1
    for start, end, label in sorted(entities):
        if start < prev_end:
            raise ValueError('Overlapping gold entities require a span head; this BIO notebook cannot encode them.')
        prev_end = end
        covered = [i for i, (s, e) in enumerate(offsets) if s < e and s < end and e > start]
        if not covered:
            report['unalignable'].append([start, end, label, 'no non-whitespace token'])
            continue
        first, last = covered[0], covered[-1]
        # Include intervening whitespace tokens to preserve contiguous entity labels.
        if occupied[first:last+1].any():
            raise ValueError('Separate entities share a BPE token. Use a finer tokenizer/span head for this data.')
        occupied[first:last+1] = True
        labels[first] = LABEL2ID['B-' + label]
        labels[first+1:last+1] = LABEL2ID['I-' + label]
        if offsets[first][0] == start and offsets[last][1] == end:
            report['exact'] += 1
        else:
            report['unalignable'].append([start, end, label, 'boundary inside token or whitespace'])
    return labels, report

def window_ranges(n_tokens, max_length=MAX_LENGTH, overlap=WINDOW_OVERLAP):
    if n_tokens == 0:
        return []
    ranges = []
    start = 0
    while True:
        end = min(start + max_length, n_tokens)
        ranges.append((start, end))
        if end == n_tokens:
            break
        start += max_length - overlap
    return ranges

def owned_positions(n_tokens, ranges):
    # Each token contributes once to training loss, using the window with most context.
    owners = np.full(n_tokens, -1, dtype=np.int64)
    best = np.full(n_tokens, -1, dtype=np.int64)
    for w, (start, end) in enumerate(ranges):
        weights = np.minimum(np.arange(end-start)+1, np.arange(end-start, 0, -1))
        take = weights > best[start:end]
        owners[start:end][take] = w
        best[start:end][take] = weights[take]
    assert n_tokens == 0 or np.all(owners >= 0)
    return owners

encoded_docs, alignment_report = {}, {}
for doc_id, text in tqdm(all_text.items(), desc='Tokenizing and aligning training documents', unit='doc'):
    item = tokenize_document(text)
    item['labels'], alignment_report[doc_id] = align_entities(text, item['offsets'], all_gold[doc_id])
    item['ranges'] = window_ranges(len(item['ids']))
    item['owners'] = owned_positions(len(item['ids']), item['ranges'])
    encoded_docs[doc_id] = item
save_json(alignment_report, PROJECT_DIR / 'alignment_report.json')
for split, ids in [('train', train_ids), ('validation', val_ids)]:
    total = sum(alignment_report[d]['entities'] for d in ids)
    exact = sum(alignment_report[d]['exact'] for d in ids)
    print(f'{split}: {exact}/{total} gold spans exactly representable by trimmed BPE boundaries')
    if exact < total:
        print('See alignment_report.json for boundary limitations of the BIO approach.')

class NERWindows(Dataset):
    def __init__(self, doc_ids, encoded):
        self.encoded = encoded
        self.rows = [(d, w, s, e) for d in doc_ids for w, (s, e) in enumerate(encoded[d]['ranges'])]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        d, w, s, e = self.rows[index]
        item = self.encoded[d]
        labels = item['labels'][s:e].copy()
        labels[item['owners'][s:e] != w] = -100
        return {'input_ids': item['ids'][s:e], 'labels': labels}

def collate_windows(examples):
    length = max(len(ex['input_ids']) for ex in examples)
    ids = torch.full((len(examples), length), PAD_ID, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, ex in enumerate(examples):
        size = len(ex['input_ids'])
        ids[i, :size] = torch.as_tensor(ex['input_ids'])
        mask[i, :size] = True
        if 'labels' in ex:
            labels[i, :size] = torch.as_tensor(ex['labels'])
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}

train_ds = NERWindows(train_ids, encoded_docs)
if not len(train_ds):
    raise ValueError('No training windows.')
print('Training windows:', len(train_ds), '| vocabulary:', VOCAB_SIZE_ACTUAL)
''')
md('## 6. Compatible Phase 1 architectures (definitions only; no pretraining)')
for index in [19, 21, 23, 25, 27]:
    code(''.join(phase1['cells'][index]['source']))
md('''
## 7. Strict Phase 1 loading and token-classification heads

Dimensions and layer counts are read from checkpoints. A strict load validates every original
parameter before the LM head is removed. Head counts use the configuration above; rotary head counts
cannot be inferred from weight shapes alone. Never run the Phase 1 training notebook to initialize
Phase 2: that would overwrite/retrain the weights instead of loading them.
''')
code('''
def torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def infer_phase1_config(architecture, state):
    vocab, dim = state['tok_emb.weight'].shape
    if vocab != VOCAB_SIZE_ACTUAL:
        raise ValueError(f'Phase 1 vocabulary {vocab} != tokenizer {VOCAB_SIZE_ACTUAL}')
    config = {'vocab_size': vocab}
    def count_layers(prefix):
        return len({int(k.split('.')[1]) for k in state if k.startswith(prefix + '.')})
    heads = PHASE1_HEADS[architecture]
    if architecture == 'decoder_only':
        config.update(n_layer=count_layers('blocks'), n_embd=dim, n_head=heads,
                      d_ff=state['blocks.0.mlp.0.weight'].shape[0], max_len=state['pos_emb.weight'].shape[0])
    elif architecture == 'encoder_decoder':
        stored_heads = state['enc_rel_bias.embedding.weight'].shape[1]
        if heads != stored_heads:
            raise ValueError('Encoder head count differs from stored relative-position bias.')
        config.update(n_enc_layer=count_layers('enc_layers'), n_dec_layer=count_layers('dec_layers'),
                      d_model=dim, n_head=heads, d_ff=state['enc_layers.0.ff.0.weight'].shape[0])
    elif architecture == 'hybrid_prefixlm':
        config.update(n_layer=count_layers('blocks'), d_model=dim, n_head=heads,
                      d_ff=state['blocks.0.ff.w1.weight'].shape[0], max_len=1152)
    else:
        head_dim = dim // heads
        config.update(n_layer=count_layers('blocks'), d_model=dim, n_head=heads,
                      n_kv_head=state['blocks.0.attn.k_proj.weight'].shape[0] // head_dim,
                      d_ff=state['blocks.0.ff.w1.weight'].shape[0], max_len=1024)
    if dim % heads:
        raise ValueError('Invalid Phase 1 attention head count.')
    return config

MODEL_CLASSES = {'decoder_only': DecoderOnlyGPT, 'encoder_decoder': EncoderDecoderTransformer,
                 'hybrid_prefixlm': HybridPrefixLM, 'lightweight': LightweightLlama}

def load_phase1(architecture):
    path = PHASE1_DIR / f'run_{architecture}' / 'final_model.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 1 weights: {path}')
    print('Loading:', path)
    state = torch_load(path)
    config = infer_phase1_config(architecture, state)
    print('Inferred config:', config)
    backbone = MODEL_CLASSES[architecture](**config)
    backbone.load_state_dict(state, strict=True)
    del state
    gc.collect()
    original_parameters = sum(p.numel() for p in backbone.parameters())
    del backbone.head
    if architecture == 'encoder_decoder':
        del backbone.dec_layers, backbone.dec_rel_bias, backbone.ln_dec_f
    return backbone, config, original_parameters

class LegalNER(nn.Module):
    def __init__(self, backbone, architecture):
        super().__init__()
        self.backbone = backbone
        self.architecture = architecture
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.classifier = nn.Linear(backbone.tok_emb.embedding_dim, NUM_LABELS)
        self.use_checkpointing = GRADIENT_CHECKPOINTING

    def layer_call(self, layer, *args):
        if self.training and self.use_checkpointing and any(p.requires_grad for p in layer.parameters()):
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, input_ids, attention_mask):
        b = self.backbone
        length = input_ids.size(1)
        x = b.tok_emb(input_ids)
        if self.architecture == 'decoder_only':
            if length > b.max_len:
                raise ValueError('MAX_LENGTH exceeds Phase 1 positional embedding length.')
            x = x + b.pos_emb(torch.arange(length, device=input_ids.device)[None, :])
            for layer in b.blocks:
                x = self.layer_call(layer, x)
            x = b.ln_f(x)
        elif self.architecture == 'encoder_decoder':
            bias = b.enc_rel_bias(length, length, input_ids.device)
            for layer in b.enc_layers:
                x = self.layer_call(layer, x, bias, attention_mask)
            x = b.ln_enc_f(x)
        else:
            cos, sin = build_rope_cache(length, b.d_model // b.n_head, input_ids.device)
            if self.architecture == 'hybrid_prefixlm':
                # All text is a bidirectional prefix; mask PAD keys explicitly.
                mask = attention_mask[:, None, None, :].expand(-1, 1, length, -1)
            else:
                causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()
                mask = causal[None, None, :, :] & attention_mask[:, None, None, :]
            for layer in b.blocks:
                x = self.layer_call(layer, x, mask, cos, sin)
            x = b.norm_f(x)
        return self.classifier(self.dropout(x))

class LowRankLinear(nn.Module):
    # LoRA: W + (alpha/r) B A.
    # LoRMA+ (paper Eq. 8): (I + (alpha/r) B A) W; bias remains separate.
    def __init__(self, base, method, rank=ADAPTER_RANK, alpha=ADAPTER_ALPHA):
        super().__init__()
        self.base = base
        self.method = method
        self.scale = alpha / rank
        self.base.requires_grad_(False)
        a_dim = base.in_features if method == 'lora' else base.out_features
        self.A = nn.Linear(a_dim, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        if self.method == 'lora':
            return self.base(x) + self.scale * self.B(self.A(x))
        wx = F.linear(x, self.base.weight, None)
        output = wx + self.scale * self.B(self.A(wx))
        return output if self.base.bias is None else output + self.base.bias

def add_attention_adapters(module, method):
    replaced = []
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and ('.attn.' in name or '.self_attn.' in name):
            if name.rsplit('.', 1)[-1] in {'qkv', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'proj'}:
                parent_name, leaf = name.rsplit('.', 1)
                parent = module.get_submodule(parent_name)
                setattr(parent, leaf, LowRankLinear(child, method))
                replaced.append(name)
    if not replaced:
        raise ValueError('No attention projections matched for adaptation.')
    return replaced

def configure_adaptation(model, method):
    model.backbone.requires_grad_(method in {'full', 'kd'})
    targets = []
    if method in {'lora', 'lorma'}:
        targets = add_attention_adapters(model.backbone, method)
    elif method == 'partial':
        layers = model.backbone.enc_layers if model.architecture == 'encoder_decoder' else model.backbone.blocks
        for layer in list(layers)[-2:]:
            layer.requires_grad_(True)
        norm = model.backbone.ln_enc_f if model.architecture == 'encoder_decoder' else (
            model.backbone.ln_f if model.architecture == 'decoder_only' else model.backbone.norm_f)
        norm.requires_grad_(True)
    elif method == 'bitfit':
        for name, p in model.backbone.named_parameters():
            p.requires_grad_(name.endswith('.bias'))
    model.classifier.requires_grad_(True)
    return targets

def make_model(spec):
    backbone, config, phase1_parameters = load_phase1(spec['architecture'])
    selected_layers = None
    if spec['method'] == 'kd':
        selected_layers = list(range(0, len(backbone.blocks), 2))
        backbone.blocks = nn.ModuleList([backbone.blocks[i] for i in selected_layers])
    model = LegalNER(backbone, spec['architecture'])
    targets = configure_adaptation(model, spec['method'])
    metadata = {'phase1_config': config, 'phase1_parameters': phase1_parameters,
                'selected_phase1_layers': selected_layers, 'adapter_targets': targets,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, metadata
''')
md('''
## 8. Decode whole documents and compute strict span metrics

Overlapping-window logits are averaged with larger weights in window centers, then BIO is decoded
**once over the complete document**, allowing entities to cross window boundaries. Right padding is
excluded from loss and decoding. Orphan `I` tags start a new span; a `B` tag always starts a new span.
All 12 official labels participate in macro averages (zero when a class has no true positives).
Local metrics use exact triples; compare this convention with the official evaluation script if supplied.
''')
code('''
def decode_bio(text, offsets, tag_ids):
    result = []
    current = None
    for (start, end), tag_id in zip(offsets, tag_ids):
        tag = ID2LABEL[int(tag_id)]
        if tag == 'O':
            if current is not None:
                result.append(current)
                current = None
            continue
        prefix, label = tag.split('-', 1)
        if start == end:
            continue  # Whitespace-only pieces carry no entity boundaries.
        if prefix == 'B' or current is None or current[2] != label:
            if current is not None:
                result.append(current)
            current = [start, end, label]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        result.append(current)
    return sorted({tuple(span) for span in result if 0 <= span[0] < span[1] <= len(text)})

def constrained_bio_tags(emissions, entity_bias=0.0):
    # Exact Viterbi path under BIO constraints. No learned CRF parameters.
    scores = np.asarray(emissions, dtype=np.float32).copy()
    if len(scores) == 0:
        return []
    scores[:, 1:] += entity_bias
    transition = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.float32)
    inside_ids = []
    for current, tag in ID2LABEL.items():
        if tag.startswith('I-'):
            inside_ids.append(current)
            label = tag[2:]
            transition[:, current] = -np.inf
            transition[LABEL2ID['B-' + label], current] = 0.0
            transition[LABEL2ID['I-' + label], current] = 0.0
    dp = scores[0].copy()
    dp[inside_ids] = -np.inf
    backpointers = np.zeros((len(scores), NUM_LABELS), dtype=np.int16)
    for t in range(1, len(scores)):
        previous_scores = dp[:, None] + transition
        backpointers[t] = previous_scores.argmax(axis=0)
        dp = previous_scores.max(axis=0) + scores[t]
        dp -= dp.max()  # Prevent numeric drift in very long documents.
    result = [int(dp.argmax())]
    for t in range(len(scores)-1, 0, -1):
        result.append(int(backpointers[t, result[-1]]))
    return result[::-1]

def tags_from_scores(scores, decoder_config=None):
    decoder_config = decoder_config or {'method': 'greedy', 'entity_bias': 0.0}
    if decoder_config['method'] == 'viterbi':
        return constrained_bio_tags(scores, decoder_config['entity_bias'])
    shifted = np.asarray(scores, dtype=np.float32).copy()
    shifted[:, 1:] += decoder_config['entity_bias']
    return shifted.argmax(axis=-1).tolist()

@torch.no_grad()
def predict_documents(model, documents, encoded=None, description='Predicting',
                      decoder_config=None, score_cache=None):
    was_training = model.training
    model.eval()
    result = {}
    progress = tqdm(documents.items(), desc=description, unit='doc')
    for doc_id, text in progress:
        item = encoded[doc_id] if encoded is not None else tokenize_document(text)
        n_tokens = len(item['ids'])
        if n_tokens == 0:
            result[doc_id] = []
            if score_cache is not None:
                score_cache[doc_id] = np.empty((0, NUM_LABELS), dtype=np.float32)
            continue
        ranges = window_ranges(n_tokens)
        logits_sum = torch.zeros((n_tokens, NUM_LABELS), dtype=torch.float32)
        weight_sum = torch.zeros(n_tokens, dtype=torch.float32)
        batches = range(0, len(ranges), BATCH_SIZE)
        for first in tqdm(batches, desc=f'Windows {doc_id}', unit='batch', leave=False, disable=len(ranges) <= BATCH_SIZE):
            subranges = ranges[first:first+BATCH_SIZE]
            examples = [{'input_ids': item['ids'][s:e]} for s, e in subranges]
            batch = collate_windows(examples)
            with amp_context():
                logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            logits = logits.float().cpu()
            for row, (start, end) in enumerate(subranges):
                size = end-start
                weights = torch.minimum(torch.arange(size)+1, torch.arange(size, 0, -1)).float()
                logits_sum[start:end] += logits[row, :size] * weights[:, None]
                weight_sum[start:end] += weights
        emissions = (logits_sum / weight_sum[:, None]).numpy()
        if score_cache is not None:
            score_cache[doc_id] = emissions
        tags = tags_from_scores(emissions, decoder_config)
        result[doc_id] = [list(span) for span in decode_bio(text, item['offsets'], tags)]
    model.train(was_training)
    return result

def strict_metrics(predictions, gold, doc_ids):
    per_label = {}
    for label in ENTITY_LABELS:
        tp = fp = fn = 0
        for doc_id in doc_ids:
            predicted = {tuple(s[:3]) for s in predictions[doc_id] if s[2] == label}
            reference = {tuple(s[:3]) for s in gold[doc_id] if s[2] == label}
            tp += len(predicted & reference)
            fp += len(predicted - reference)
            fn += len(reference - predicted)
        precision = tp / (tp+fp) if tp+fp else 0.0
        recall = tp / (tp+fn) if tp+fn else 0.0
        f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        per_label[label] = {'precision': precision, 'recall': recall, 'f1': f1,
                            'tp': tp, 'fp': fp, 'fn': fn, 'support': tp+fn}
    return {'macro_precision': sum(x['precision'] for x in per_label.values()) / len(ENTITY_LABELS),
            'macro_recall': sum(x['recall'] for x in per_label.values()) / len(ENTITY_LABELS),
            'macro_f1': sum(x['f1'] for x in per_label.values()) / len(ENTITY_LABELS),
            'per_label': per_label}

def validate_predictions(predictions, documents):
    assert set(predictions) == set(documents), 'Missing or extra official document IDs.'
    for doc_id, spans in predictions.items():
        assert len(spans) == len({tuple(s) for s in spans}), f'Duplicate spans: {doc_id}'
        for span in spans:
            assert len(span) == 3
            start, end, label = span
            assert type(start) is int and type(end) is int
            assert 0 <= start < end <= len(documents[doc_id])
            assert label in ENTITY_LABELS

def show_metric_report(metrics, title='Post-training validation — exact character spans'):
    print()
    print(title)
    print(f"Strict Macro-F1        : {metrics['macro_f1']:.6f} ({100*metrics['macro_f1']:.2f}%)  [PRIMARY]")
    print(f"Strict Macro-Precision : {metrics['macro_precision']:.6f} ({100*metrics['macro_precision']:.2f}%)")
    print(f"Strict Macro-Recall    : {metrics['macro_recall']:.6f} ({100*metrics['macro_recall']:.2f}%)")
    print(f"{'Label':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Gold':>8s} {'TP':>8s} {'FP':>8s} {'FN':>8s}")
    for label in ENTITY_LABELS:
        row = metrics['per_label'][label]
        print(f"{label:12s} {row['precision']:10.4f} {row['recall']:10.4f} {row['f1']:10.4f} "
              f"{row['support']:8d} {row['tp']:8d} {row['fp']:8d} {row['fn']:8d}")
    print('Held-out labeled training documents; official test labels are private.')

def build_class_weights():
    # Count entities on TRAIN ONLY; cap inverse-square-root weights for rare types.
    counts = Counter(span[2] for doc_id in train_ids for span in all_gold[doc_id])
    positive = [counts[label] for label in ENTITY_LABELS if counts[label] > 0]
    reference = float(np.median(positive)) if positive else 1.0
    weights = [O_LOSS_WEIGHT]
    for label in ENTITY_LABELS:
        weight = min(MAX_ENTITY_CLASS_WEIGHT, max(0.5, math.sqrt(reference / max(1, counts[label]))))
        weights.extend([weight, weight])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def optimizer_groups(model, base_lr):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith('classifier.')
        no_decay = p.ndim < 2 or name.endswith('.bias') or 'norm' in name or '.ln' in name
        key = (is_head, no_decay)
        groups.setdefault(key, []).append(p)
    return [{'params': params, 'lr': base_lr * (HEAD_LR_MULTIPLIER if is_head else 1.0),
             'weight_decay': 0.0 if no_decay else WEIGHT_DECAY}
            for (is_head, no_decay), params in groups.items()]

def verify_and_select_decoder(model, output_dir):
    val_docs = {doc_id: all_text[doc_id] for doc_id in val_ids}
    cached_scores = {}
    # Recompute from the RELOADED best checkpoint, rather than trusting a saved metric.
    baseline_predictions = predict_documents(model, val_docs, encoded_docs,
        'Post-training validation: reloaded best weights', score_cache=cached_scores)
    validate_predictions(baseline_predictions, val_docs)
    baseline_metrics = strict_metrics(baseline_predictions, all_gold, val_ids)
    recorded = json.loads((output_dir / 'best_validation_metrics.json').read_text())
    for key in ['macro_f1', 'macro_precision', 'macro_recall']:
        if not math.isclose(baseline_metrics[key], recorded[key], abs_tol=1e-10):
            raise RuntimeError(f'Reloaded checkpoint metric disagrees with recorded {key}.')
    candidates = [{'method': 'greedy', 'entity_bias': 0.0}]
    if OPTIMIZE_DECODING:
        candidates.extend({'method': 'viterbi', 'entity_bias': bias} for bias in DECODER_ENTITY_BIASES)
    best_decoder = candidates[0]
    best_metrics = baseline_metrics
    best_predictions = baseline_predictions
    results = []
    for candidate in tqdm(candidates, desc='Selecting decoder on validation spans', unit='candidate'):
        if candidate == candidates[0]:
            predictions, metrics = baseline_predictions, baseline_metrics
        else:
            predictions = {}
            for doc_id in tqdm(val_ids, desc=str(candidate), unit='doc', leave=False):
                tags = tags_from_scores(cached_scores[doc_id], candidate)
                predictions[doc_id] = [list(span) for span in decode_bio(
                    all_text[doc_id], encoded_docs[doc_id]['offsets'], tags)]
            validate_predictions(predictions, val_docs)
            metrics = strict_metrics(predictions, all_gold, val_ids)
        results.append({'decoder': candidate, **{k: metrics[k] for k in ['macro_f1', 'macro_precision', 'macro_recall']}})
        if metrics['macro_f1'] > best_metrics['macro_f1']:
            best_metrics, best_predictions, best_decoder = metrics, predictions, candidate
    save_json(results, output_dir / 'decoding_candidates.json')
    save_json(best_decoder, output_dir / 'decoder_config.json')
    save_json(best_predictions, output_dir / 'validation_prediction.json')
    # Independent JSON round-trip and exact-set count verification.
    reloaded_predictions = json.loads((output_dir / 'validation_prediction.json').read_text())
    validate_predictions(reloaded_predictions, val_docs)
    roundtrip_metrics = strict_metrics(reloaded_predictions, all_gold, val_ids)
    assert roundtrip_metrics == best_metrics
    for label in ENTITY_LABELS:
        expected_support = sum(sum(s[2] == label for s in set(all_gold[d])) for d in val_ids)
        assert best_metrics['per_label'][label]['support'] == expected_support
    assert best_metrics['macro_f1'] >= baseline_metrics['macro_f1']
    summary = {'baseline_greedy_metrics': baseline_metrics, 'selected_metrics': best_metrics,
               'selected_decoder': best_decoder, 'checkpoint_reload_matches_recorded_scores': True,
               'validation_schema_and_ids_passed': True, 'json_roundtrip_passed': True,
               'gold_support_counts_verified': True, 'validation_documents': len(val_ids),
               'score_scope': 'held-out validation used for checkpoint/decoder selection, not an unbiased test estimate'}
    save_json(summary, output_dir / 'post_training_validation.json')
    show_metric_report(best_metrics)
    print('Greedy baseline F1:', f"{baseline_metrics['macro_f1']:.6f}", '| Selected decoder:', best_decoder)
    print('Verification: checkpoint reload, exact spans/labels, all IDs, JSON round-trip, and support counts PASS.')
    return best_metrics, best_decoder, summary

def run_path(name):
    return PROJECT_DIR / f'{name}_{RUN_TAG}'

def choose_teacher():
    names = [TEACHER_EXPERIMENT] if TEACHER_EXPERIMENT else TEACHER_CANDIDATES
    choices = []
    for name in names:
        path = run_path(name)
        if (path / 'best_model.pt').is_file() and (path / 'metrics.json').is_file():
            metrics = json.loads((path / 'metrics.json').read_text())
            if metrics.get('training_complete'):
                choices.append((metrics['macro_f1'], name))
    if not choices:
        raise FileNotFoundError('KD needs a completed Phase 2 teacher. Train an architecture run first.')
    _, name = max(choices)
    checkpoint_data = torch_load(run_path(name) / 'best_model.pt')
    saved_config = checkpoint_data['run_config']
    if saved_config['data_fingerprint'] != DATA_FINGERPRINT or saved_config['split'] != SPLIT_INFO:
        raise ValueError('Teacher must use the same held-out split and dataset.')
    if saved_config['tokenizer_hash'] != TOKENIZER_HASH or saved_config['bio_labels'] != BIO_LABELS:
        raise ValueError('Teacher tokenizer/label mapping differs.')
    for key, current in [('adapter_rank', ADAPTER_RANK), ('adapter_alpha', ADAPTER_ALPHA),
                         ('phase1_heads', PHASE1_HEADS), ('max_length', MAX_LENGTH)]:
        if saved_config[key] != current:
            raise ValueError(f'Teacher config mismatch for {key}; use the settings from its run_config.json.')
    teacher, _ = make_model(saved_config['spec'])
    teacher.load_state_dict(checkpoint_data['model'], strict=True)
    teacher.eval().requires_grad_(False)
    print('KD teacher:', name)
    del checkpoint_data
    return teacher.to(DEVICE), name
''')
md('## 9. Pipeline checks before spending GPU time')
code('''
# Pure logic checks use the real Phase 1 tokenizer, including whitespace, punctuation, and Unicode.
sample_text = '  SUPREME COURT of India. José filed on 12/03/2020.  '
sample = tokenize_document(sample_text)
sample_spans = [(sample_text.index('SUPREME'), sample_text.index(' of'), 'COURT'),
                (sample_text.index('12/03'), sample_text.index('12/03')+10, 'DATE')]
sample_labels, sample_report = align_entities(sample_text, sample['offsets'], sample_spans)
assert sample_report['exact'] == len(sample_spans), sample_report
assert decode_bio(sample_text, sample['offsets'], sample_labels) == sample_spans
ranges = window_ranges(1100)
owners = owned_positions(1100, ranges)
assert set(owners) == set(range(len(ranges)))
# Entity spanning a window boundary must remain a single document-level span.
assert decode_bio('abcdef', [(i, i+1) for i in range(6)],
                  [0, LABEL2ID['B-APP'], LABEL2ID['I-APP'], LABEL2ID['I-APP'], 0, 0]) == [(1, 4, 'APP')]
for method in ['lora', 'lorma']:
    base = nn.Linear(7, 11)
    wrapped = LowRankLinear(copy.deepcopy(base), method, rank=3, alpha=3)
    x = torch.randn(2, 4, 7)
    torch.testing.assert_close(wrapped(x), base(x))
    wrapped(x).sum().backward()
    assert wrapped.B.weight.grad is not None and wrapped.B.weight.grad.abs().sum() > 0
    assert wrapped.base.weight.grad is None
    with torch.no_grad():
        wrapped.B.weight.normal_(std=0.02)
    delta = wrapped.B.weight @ wrapped.A.weight
    merged_weight = base.weight + wrapped.scale * delta if method == 'lora' else (
        base.weight + wrapped.scale * delta @ base.weight)
    torch.testing.assert_close(wrapped(x), F.linear(x, merged_weight, base.bias))
print('Offset round-trip, cross-window BIO, adapter identity, multiplicative formula, and gradient checks passed.')
''')
md('''
## 10. Train, validate, save, and resume

Validation uses complete held-out documents and selects the best epoch by **strict macro-F1**.
The loss is class-weighted supervised token cross-entropy (weights computed from training entities only);
rare entity types receive bounded weights and `O` is downweighted. The new classification head uses
a higher learning rate than the pretrained backbone, and norm/bias parameters have no weight decay.
KD additionally uses temperature-scaled token-level KL
from a frozen, trained Phase 2 teacher. Padding and duplicate overlap tokens are excluded from both.
Microbatches are accumulated with the correct divisor for a short final accumulation group.
Learning-rate scheduling advances only when AMP actually performs an optimizer update.

`last.pt` contains full training state and is saved periodically and at epoch boundaries. The
`best_model.pt` includes model and reconstruction metadata. `history.json`, `metrics.json`, configuration,
validation predictions, plots, and final submission are persisted in Drive. Resume restarts only the
unsaved portion since the latest optimizer checkpoint. Training uses a fixed shuffle per epoch and
restores random state, including dropout.
''')
code('''
def capture_rng():
    return {'python': random.getstate(), 'numpy': {
        'kind': np.random.get_state()[0], 'keys': np.random.get_state()[1].tolist(),
        'pos': np.random.get_state()[2], 'gauss': np.random.get_state()[3],
        'cached': np.random.get_state()[4]}, 'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else []}

def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32), n['pos'], n['gauss'], n['cached']))
    torch.set_rng_state(state['torch'])
    if DEVICE.type == 'cuda' and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])

def make_run_config(name):
    return {'experiment': name, 'spec': EXPERIMENTS[name], 'run_tag': RUN_TAG,
            'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'grad_accum_steps': GRAD_ACCUM_STEPS, 'max_length': MAX_LENGTH,
            'window_overlap': WINDOW_OVERLAP, 'weight_decay': WEIGHT_DECAY,
            'warmup_fraction': WARMUP_FRACTION, 'max_optimizer_steps': MAX_OPTIMIZER_STEPS,
            'adapter_rank': ADAPTER_RANK, 'adapter_alpha': ADAPTER_ALPHA,
            'head_dropout': HEAD_DROPOUT, 'gradient_checkpointing': GRADIENT_CHECKPOINTING,
            'kd_temperature': KD_TEMPERATURE, 'kd_weight': KD_WEIGHT,
            'use_class_weights': USE_CLASS_WEIGHTS, 'o_loss_weight': O_LOSS_WEIGHT,
            'max_entity_class_weight': MAX_ENTITY_CLASS_WEIGHT, 'head_lr_multiplier': HEAD_LR_MULTIPLIER,
            'optimize_decoding': OPTIMIZE_DECODING, 'decoder_entity_biases': DECODER_ENTITY_BIASES,
            'teacher_requested': TEACHER_EXPERIMENT, 'phase1_heads': PHASE1_HEADS,
            'tokenizer_hash': TOKENIZER_HASH, 'bio_labels': BIO_LABELS,
            'data_fingerprint': DATA_FINGERPRINT, 'split': SPLIT_INFO}

def run_experiment(name):
    seed_all(SEED)
    spec = EXPERIMENTS[name]
    output_dir = run_path(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_run_config(name)
    config_path = output_dir / 'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f'Config differs for {name}. Use a new RUN_TAG for changed settings.')
        if not RESUME and (output_dir / 'last.pt').exists():
            raise ValueError('Existing training state: use a new RUN_TAG instead of overwriting it.')
    save_json(config, config_path)
    model, metadata = make_model(spec)
    teacher = None
    teacher_name = None
    if spec['method'] == 'kd':
        teacher, teacher_name = choose_teacher()
        if sum(p.numel() for p in teacher.parameters()) <= metadata['total_parameters']:
            raise ValueError('KD teacher must be larger than the student. Select another teacher run.')
    metadata['teacher_experiment'] = teacher_name
    model = model.to(DEVICE)
    print(name, '| parameters:', metadata['total_parameters'], '| trainable:', metadata['trainable_parameters'])
    save_json(metadata, output_dir / 'model_metadata.json')
    save_json({'python': platform.python_version(), 'torch': str(torch.__version__),
               'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None},
              output_dir / 'environment.json')
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(optimizer_groups(model, spec['lr']))
    class_weights = build_class_weights() if USE_CLASS_WEIGHTS else None
    save_json({label: float(class_weights[i].item()) if class_weights is not None else 1.0
               for i, label in ID2LABEL.items()}, output_dir / 'loss_class_weights.json')
    batches_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
    steps_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM_STEPS)
    schedule_steps = EPOCHS * steps_per_epoch
    warmup = max(1, round(schedule_steps * WARMUP_FRACTION))
    def lr_scale(step):
        if step < warmup:
            return (step+1) / warmup
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, (step-warmup) / max(1, schedule_steps-warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda' and AMP_DTYPE == torch.float16)
    history = []
    best_f1 = -1.0
    global_step, start_epoch, start_batch = 0, 0, 0
    cumulative_seconds = 0.0
    last_path = output_dir / 'last.pt'
    if RESUME and last_path.is_file():
        state = torch_load(last_path)
        if state['run_config'] != config or state['metadata']['teacher_experiment'] != teacher_name:
            raise ValueError('Saved training state or KD teacher differs; choose a new RUN_TAG.')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, best_f1 = state['history'], state['best_f1']
        global_step, start_epoch, start_batch = state['step'], state['epoch'], state['next_batch']
        cumulative_seconds = state['training_seconds']
        restore_rng(state['rng'])
        del state
        print(f'Resuming epoch {start_epoch+1}, batch {start_batch}, optimizer step {global_step}')
    started = time.time()
    def persist(epoch, next_batch):
        save_torch({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'history': history, 'best_f1': best_f1, 'step': global_step,
                    'epoch': epoch, 'next_batch': next_batch, 'rng': capture_rng(),
                    'training_seconds': cumulative_seconds + time.time()-started,
                    'run_config': config, 'metadata': metadata}, last_path)
        save_json(history, output_dir / 'history.json')
    stopped = False
    training_complete = start_epoch >= EPOCHS
    for epoch in range(start_epoch, EPOCHS):
        if MAX_OPTIMIZER_STEPS is not None and global_step >= MAX_OPTIMIZER_STEPS:
            stopped = True
            break
        # Deterministic order and a dedicated loader RNG avoid changing dropout RNG on resume.
        order = list(range(len(train_ds)))
        random.Random(SEED + epoch).shuffle(order)
        begin = start_batch if epoch == start_epoch else 0
        remaining = order[begin*BATCH_SIZE:]
        loader = DataLoader(torch.utils.data.Subset(train_ds, remaining), batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_windows, num_workers=NUM_WORKERS,
                            generator=torch.Generator().manual_seed(SEED + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(loader, total=len(loader), desc=f'{name} epoch {epoch+1}/{EPOCHS}', unit='batch')
        group_loss = 0.0
        next_batch = begin
        for local_index, batch in enumerate(bar):
            batch_index = begin + local_index
            next_batch = batch_index + 1
            group_start = (batch_index // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
            group_size = min(GRAD_ACCUM_STEPS, batches_per_epoch-group_start)
            ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            valid = labels != -100
            if not valid.any():
                raise ValueError('A training batch has no owned labels.')
            with amp_context():
                logits = model(ids, attention_mask)
                supervised = F.cross_entropy(logits[valid].float(), labels[valid], weight=class_weights)
                loss = supervised
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(ids, attention_mask)
                    temperature = KD_TEMPERATURE
                    kd_loss = F.kl_div(F.log_softmax(logits[valid].float()/temperature, dim=-1),
                                       F.softmax(teacher_logits[valid].float()/temperature, dim=-1),
                                       reduction='batchmean') * temperature**2
                    loss = (1-KD_WEIGHT)*supervised + KD_WEIGHT*kd_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss. Latest saved optimizer checkpoint is preserved.')
            scaler.scale(loss / group_size).backward()
            group_loss += loss.detach().item() / group_size
            if next_batch % GRAD_ACCUM_STEPS == 0 or next_batch == batches_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= old_scale
                optimizer.zero_grad(set_to_none=True)
                if did_update:
                    scheduler.step()
                    global_step += 1
                    history.append({'kind': 'train', 'epoch': epoch+1, 'step': global_step,
                                    'loss': group_loss, 'lr': optimizer.param_groups[0]['lr']})
                    if global_step % SAVE_EVERY_STEPS == 0:
                        persist(epoch, next_batch)
                group_loss = 0.0
                bar.set_postfix(loss=f'{loss.item():.4f}', step=global_step,
                                lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
                if MAX_OPTIMIZER_STEPS is not None and global_step >= MAX_OPTIMIZER_STEPS:
                    stopped = True
                    break
            else:
                bar.set_postfix(loss=f'{loss.item():.4f}', step=global_step)
        bar.close()
        val_docs = {d: all_text[d] for d in val_ids}
        val_predictions = predict_documents(model, val_docs, encoded_docs, 'Validating strict spans')
        metrics = strict_metrics(val_predictions, all_gold, val_ids)
        history.append({'kind': 'validation', 'epoch': epoch+1, 'step': global_step,
                        'macro_f1': metrics['macro_f1'], 'macro_precision': metrics['macro_precision'],
                        'macro_recall': metrics['macro_recall']})
        print(f'{name}: strict macro-F1={metrics["macro_f1"]:.4f} | '
              f'precision={metrics["macro_precision"]:.4f} | recall={metrics["macro_recall"]:.4f}')
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            save_torch({'model': model.state_dict(), 'run_config': config, 'metadata': metadata,
                        'step': global_step, 'epoch': epoch+1, 'best_f1': best_f1}, output_dir / 'best_model.pt')
            save_json(val_predictions, output_dir / 'validation_prediction.json')
            save_json(metrics, output_dir / 'best_validation_metrics.json')
        finished_epoch = next_batch == batches_per_epoch
        persist(epoch+1 if finished_epoch else epoch, 0 if finished_epoch else next_batch)
        training_complete = finished_epoch and epoch+1 == EPOCHS
        if stopped:
            break
    if not (output_dir / 'best_model.pt').is_file():
        raise RuntimeError('No evaluated checkpoint is available. Remove the smoke-test cap or train longer.')
    checkpoint_data = torch_load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint_data['model'], strict=True)
    del checkpoint_data
    metrics, decoder_config, validation_summary = verify_and_select_decoder(model, output_dir)
    metrics['baseline_macro_f1'] = validation_summary['baseline_greedy_metrics']['macro_f1']
    metrics['selected_decoder'] = decoder_config
    metrics.update({'experiment': name, 'architecture': spec['architecture'], 'method': spec['method'],
                    'training_complete': training_complete, 'smoke_test_cap': MAX_OPTIMIZER_STEPS,
                    'optimizer_steps': global_step, 'training_seconds': cumulative_seconds + time.time()-started,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else 0,
                    **metadata})
    save_json(metrics, output_dir / 'metrics.json')
    train_records = [r for r in history if r['kind'] == 'train']
    eval_records = [r for r in history if r['kind'] == 'validation']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r['step'] for r in train_records], [r['loss'] for r in train_records])
    axes[0].set(xlabel='Optimizer step', ylabel='Training loss', title=name)
    axes[1].plot([r['step'] for r in eval_records], [r['macro_f1'] for r in eval_records], marker='o')
    axes[1].set(xlabel='Optimizer step', ylabel='Validation strict macro-F1')
    fig.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.show()
    plt.close(fig)
    if teacher is not None:
        del teacher
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    predictions = predict_documents(model, test_text, description=f'{name}: official test',
                                    decoder_config=decoder_config)
    validate_predictions(predictions, test_text)
    prediction_path = output_dir / 'prediction.json'
    save_json(predictions, prediction_path)
    zip_path = output_dir / 'submission.zip'
    zip_temporary = output_dir / 'submission.zip.tmp'
    with zipfile.ZipFile(zip_temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname='prediction.json')
    os.replace(zip_temporary, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ['prediction.json']
        assert json.loads(archive.read('prediction.json')) == predictions
    save_json({'validation_checks_passed': True, 'prediction_ids_and_offsets_passed': True,
               'test_documents': len(test_text), 'predicted_entities': sum(map(len, predictions.values())),
               'zip_root_and_json_passed': True, 'official_test_score': None}, output_dir / 'verification.json')
    print('Submission verification PASS: all official IDs, valid spans and labels, and ZIP root format.')
    print('Saved submission:', zip_path)
    if not training_complete:
        print('Training was capped before completion: this is a smoke-test artifact, not a completed experiment.')
    del model, optimizer
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return metrics
''')
md('## 11. Execute the selected experiment (or all six sequentially)')
code('''
selected_experiments = list(EXPERIMENTS) if RUN_ALL else [EXPERIMENT]
run_results = []
for name in tqdm(selected_experiments, desc='Experiments', unit='run'):
    run_results.append(run_experiment(name))
print('Results stored in:', PROJECT_DIR)
''')
md('''
## 12. Post-training validation and submission verification

Like the professor's post-training evaluation section, this cell displays the saved metrics **after
training**. Phase 2 uses strict macro-F1 (primary), strict macro-precision, and strict macro-recall
(secondary), rather than Phase 1 perplexity. All scores refer to exact character-span + label matches.
The training function first reloads the best checkpoint and recomputes these metrics. The report includes
per-label TP/FP/FN and counts, checks the saved JSON, and verifies the official submission ZIP.

The small decoder search uses this same validation split, so its scores are selection scores, not an
unbiased estimate of test performance. Private test scores are available only from Codabench.
''')
code('''
for name in selected_experiments:
    output_dir = run_path(name)
    report = json.loads((output_dir / 'post_training_validation.json').read_text())
    show_metric_report(report['selected_metrics'], title=f'{name}: post-training validation score')
    print('Selected decoder:', report['selected_decoder'])
    checks = json.loads((output_dir / 'verification.json').read_text())
    print('Verification checks:', checks)
    with zipfile.ZipFile(output_dir / 'submission.zip') as archive:
        assert archive.namelist() == ['prediction.json']
        saved_predictions = json.loads(archive.read('prediction.json'))
    validate_predictions(saved_predictions, test_text)
    assert saved_predictions == json.loads((output_dir / 'prediction.json').read_text())
    print('Reopened submission.zip: schema, document IDs, and JSON contents PASS.')
''')
md('''
## 13. Saved experiment table and submission coverage

This table contains measured values only. Compare method/architecture choices yourself and use your
own observations in the final report. Scores are from the held-out split, not the private leaderboard.
The audit checks that files exist and that four Phase 1 origins, at least three adaptation techniques,
LORMA, and KD appear among completed runs. It cannot confirm Codabench submission acceptance.
''')
code('''
import csv
summary = []
for name in EXPERIMENTS:
    metric_file = run_path(name) / 'metrics.json'
    if metric_file.is_file():
        m = json.loads(metric_file.read_text())
        summary.append({key: m.get(key) for key in ['experiment', 'architecture', 'method', 'macro_f1',
            'macro_precision', 'macro_recall', 'total_parameters', 'trainable_parameters',
            'training_seconds', 'training_complete']})
if summary:
    with open(PROJECT_DIR / f'experiment_summary_{RUN_TAG}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f'{row["experiment"]:22s} F1={row["macro_f1"]:.4f} completed={row["training_complete"]}')
completed = [r for r in summary if r['training_complete']]
architectures = {r['architecture'] for r in completed}
methods = {r['method'] for r in completed}
checks = {'six_completed_experiments': len(completed) >= 6,
          'all_four_phase1_origins': architectures == set(MODEL_CLASSES),
          'three_adaptation_methods': len(methods - {'kd'}) >= 3,
          'lorma_completed': 'lorma' in methods, 'kd_completed': 'kd' in methods,
          'all_completed_zip_files_exist': all((run_path(r['experiment']) / 'submission.zip').is_file() for r in completed)}
save_json(checks, PROJECT_DIR / f'coverage_{RUN_TAG}.json')
print('Coverage:', checks)
print("Upload each distinct run submission.zip manually to Codabench.")
''')
md('''
## Output layout and references

```
MyDrive/cs781_phase2/
  bpe_tokenizer/                   # exact Phase 1 tokenizer copy
  split_seed_42.json               # held-out document IDs and dataset fingerprint
  labels.json
  alignment_report.json
  decoder_full_v2/                 # one directory for each selected experiment
    run_config.json
    model_metadata.json
    environment.json
    last.pt                       # optimizer, scheduler, scaler, RNG, resume position
    best_model.pt                 # checkpoint selected by strict validation macro-F1
    history.json
    best_validation_metrics.json
    validation_prediction.json
    metrics.json
    loss_class_weights.json
    post_training_validation.json
    decoding_candidates.json
    decoder_config.json
    verification.json
    training_curves.png
    prediction.json
    submission.zip                # prediction.json directly at archive root
  experiment_summary_v2.csv
  coverage_v2.json
```

After a disconnect, reconnect Drive, run the setup/definition cells and the training cell with the
**same experiment and settings**. You can lower batch size for a fresh run if GPU memory is insufficient;
use a new run tag because changing batch size changes optimizer/resume semantics.
If your Phase 1 dimensions changed, most dimensions are inferred, but update `PHASE1_HEADS` explicitly.
Your Drive checkpoint files are state dictionaries, so architecture code is required when reloading.

References:
- Local `Capstone Document_edited.pdf`, §§1, 8, 19, for requirements and output schema.
- Your `CS781_Phase1_Solution_Colab.ipynb` for compatible Phase 1 definitions and BPE tokenizer.
- Professor's `CS781_Phase1_Starter_Code.ipynb` for workflow and submission archive structure.
- [LoRA paper](https://arxiv.org/abs/2106.09685).
- [LoRMA paper](https://aclanthology.org/2025.findings-acl.527/), additive rank inflation (LoRMA+, Eq. 8).
- [Knowledge distillation](https://arxiv.org/abs/1503.02531).
- [BitFit](https://aclanthology.org/2022.acl-short.1/).
- [Constrained decoding for NER](https://aclanthology.org/2020.findings-emnlp.166/).

This notebook does not download another pretrained language model or call an external LLM API.
Real scores, runtime, and leaderboard acceptance must come from your actual runs.
''')
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
      'language_info': {'name': 'python'}, 'colab': {'name': 'CS781_Phase2_Solution_Colab.ipynb', 'provenance': []}},
      'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'phase2-{i:02d}'
output = ROOT / 'Phase2' / 'CS781_Phase2_Solution_Colab.ipynb'
output.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print(f'Created {output} with {len(cells)} cells')
