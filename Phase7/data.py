def case_key(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('Case IDs must be strings or integers; original leading zeros are preserved.')
    return str(value)


def text_value(value):
    if value is None: return ''
    if isinstance(value, str): return value
    if isinstance(value, dict): value = list(value.values())
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value): return '\n'.join(value)
        if all(isinstance(item, dict) and isinstance(item.get('text'), str) for item in value):
            return '\n'.join(item['text'] for item in value)
    raise ValueError('Expected case text or an ordered list of sentence strings/text objects.')


def normalize_records(raw, queries=False, labeled=False):
    if isinstance(raw, dict) and 'id' in raw and 'text' in raw: entries = [(None, raw)]
    elif isinstance(raw, dict) and isinstance(raw.get('data'), list): entries = [(None, row) for row in raw['data']]
    elif isinstance(raw, dict): entries = list(raw.items())
    elif isinstance(raw, list): entries = [(None, row) for row in raw]
    else: raise ValueError('Expected case objects or an ID-keyed mapping.')
    result = {}
    for outer, row in entries:
        if not queries and isinstance(row, (str, list)): row = {'id': outer, 'text': row}
        if not isinstance(row, dict): raise ValueError('Invalid case object.')
        original = row.get('id', outer); key = case_key(original)
        if outer is not None and case_key(outer) != key: raise ValueError('Embedded ID differs from mapping key.')
        text = text_value(row.get('text', row.get('document')))
        relevant = None
        if queries and labeled:
            relevant = row.get('relevant_candidates')
            if isinstance(relevant, str):
                try: relevant = json.loads(relevant)
                except json.JSONDecodeError:
                    try: relevant = __import__('ast').literal_eval(relevant)
                    except (ValueError, SyntaxError): raise ValueError(f'{key}: relevant_candidates is not a valid list.')
            if not isinstance(relevant, list): raise ValueError(f'{key}: relevant_candidates must be a list on train/dev queries.')
            relevant = list(dict.fromkeys(case_key(value['id'] if isinstance(value, dict) else value) for value in relevant))
        normalized = {'id': key, 'original_id': original, 'text': text, 'relevant': relevant}
        if key in result:
            if result[key] != normalized: raise ValueError(f'Conflicting duplicate case ID {key}; no rows were overwritten.')
            continue
        result[key] = normalized
    return result


QUERIES = {}; CANDIDATES = {}; DATA_AUDIT = {}
for split in ['train', 'dev', 'test']:
    QUERIES[split] = normalize_records(load_download(split + '_queries'), True, split != 'test')
    CANDIDATES[split] = normalize_records(load_download(split + '_candidates'))
    if not QUERIES[split] or len(CANDIDATES[split]) < TOP_K:
        raise ValueError(f'{split}: need queries and at least {TOP_K} candidate IDs.')
    numeric = {}
    for candidate in CANDIDATES[split]:
        if candidate.isdigit(): numeric.setdefault(str(int(candidate)), []).append(candidate)
    conversions = 0
    for query, row in QUERIES[split].items():
        if split == 'test':
            assert row['relevant'] is None
            continue
        resolved = []
        for candidate in row['relevant']:
            if candidate not in CANDIDATES[split]:
                options = numeric.get(str(int(candidate)), []) if candidate.isdigit() else []
                if len(options) != 1: raise ValueError(f'{split}/{query}: gold ID {candidate} is absent/ambiguous in this split candidate pool.')
                candidate = options[0]; conversions += 1
            resolved.append(candidate)
        row['relevant'] = list(dict.fromkeys(resolved))
        if EXCLUDE_SELF_MATCH and query in row['relevant']: raise ValueError('Gold includes self ID; disable EXCLUDE_SELF_MATCH.')
    if EXCLUDE_SELF_MATCH and any(len(CANDIDATES[split]) - (query in CANDIDATES[split]) < TOP_K for query in QUERIES[split]):
        raise ValueError('Self exclusion leaves fewer than seven candidates.')
    DATA_AUDIT[split] = {'queries': len(QUERIES[split]), 'candidates': len(CANDIDATES[split]),
                         'empty_query_texts': sum(not row['text'].strip() for row in QUERIES[split].values()),
                         'empty_candidate_texts': sum(not row['text'].strip() for row in CANDIDATES[split].values()),
                         'unambiguous_gold_id_conversions': conversions,
                         'queries_without_gold': sum(not row['relevant'] for row in QUERIES[split].values()) if split != 'test' else None}
    if all(not row['text'].strip() for row in CANDIDATES[split].values()): raise ValueError(f'{split}: all candidate texts are empty; inspect source schema.')
    if all(not row['text'].strip() for row in QUERIES[split].values()): raise ValueError(f'{split}: all query texts are empty; inspect source schema.')
    print(split, DATA_AUDIT[split])
TRAIN_KEYS = [key for key, row in QUERIES['train'].items() if row['relevant']]
TRAIN_KEY_SET = set(TRAIN_KEYS)
if not TRAIN_KEYS: raise ValueError('No training queries have positive candidates.')
fingerprint = hashlib.sha256(str(EXCLUDE_SELF_MATCH).encode())
for split in ['train', 'dev', 'test']:
    for kind, records in [('queries', QUERIES[split]), ('candidates', CANDIDATES[split])]:
        for key, row in records.items(): fingerprint.update(json.dumps([split, kind, key, row], sort_keys=True, ensure_ascii=False).encode())
DATA_FINGERPRINT = fingerprint.hexdigest()
DATA_SELF_EXCLUSION = EXCLUDE_SELF_MATCH
save_json({'fingerprint': DATA_FINGERPRINT, 'files': FILES, 'splits': DATA_AUDIT,
           'train_queries_with_positive_supervision': TRAIN_KEYS,
           'train_queries_without_positive_supervision': [key for key in QUERIES['train'] if key not in TRAIN_KEY_SET]}, PROJECT_DIR / 'dataset_manifest.json')

TOKENIZER_DIR = PHASE1_DIR / 'bpe_tokenizer'
tok = ByteLevelBPETokenizer(str(TOKENIZER_DIR / 'vocab.json'), str(TOKENIZER_DIR / 'merges.txt'))
VOCAB_SIZE_ACTUAL = tok.get_vocab_size()
PAD_ID = tok.token_to_id('<pad>'); BOS_ID = tok.token_to_id('<bos>'); EOS_ID = tok.token_to_id('<eos>'); UNK_ID = tok.token_to_id('<unk>')
if any(value is None for value in [PAD_ID, BOS_ID, EOS_ID, UNK_ID]): raise ValueError('Missing Phase 1 BPE special tokens.')
TOKENIZER_HASH = hashlib.sha256((TOKENIZER_DIR / 'vocab.json').read_bytes() + (TOKENIZER_DIR / 'merges.txt').read_bytes()).hexdigest()


class TokenStore:
    def __init__(self, path, index):
        self.data = np.memmap(path, mode='r', dtype='<i4') if path.stat().st_size else np.empty(0, dtype='<i4'); self.index = index
    def __getitem__(self, key):
        start, length = self.index[key]; return self.data[start:start + length].tolist()


TOKENS = {}
for split in ['train', 'dev', 'test']:
    TOKENS[split] = {}
    for kind, records in [('queries', QUERIES[split]), ('candidates', CANDIDATES[split])]:
        stem = f'tokens_{split}_{kind}_{DATA_FINGERPRINT[:16]}_{TOKENIZER_HASH[:12]}'
        path = PROJECT_DIR / (stem + '.bin'); index_path = PROJECT_DIR / (stem + '.json')
        if not path.is_file() or not index_path.is_file():
            temporary = path.with_name(path.name + '.' + __import__('uuid').uuid4().hex + '.tmp'); index = {}; offset = 0
            with temporary.open('wb') as stream:
                for key, row in tqdm(records.items(), desc='Tokenizing ' + split + '/' + kind, unit='case'):
                    ids = np.asarray(tok.encode(row['text'], add_special_tokens=False).ids, dtype='<i4')
                    stream.write(ids.tobytes()); index[key] = [offset, len(ids)]; offset += len(ids)
            os.replace(temporary, path); save_json(index, index_path)
        index = json.loads(index_path.read_text())
        if set(index) != set(records): raise ValueError('Token cache IDs disagree.')
        active = path
        if IN_COLAB:
            staging = Path('/content/cs781_phase7_tokens'); staging.mkdir(exist_ok=True); active = staging / path.name
            if not active.is_file() or active.stat().st_size != path.stat().st_size:
                temporary = active.with_name(active.name + '.' + __import__('uuid').uuid4().hex + '.tmp')
                with path.open('rb') as source, temporary.open('wb') as target:
                    with tqdm.wrapattr(source, 'read', total=path.stat().st_size, desc='Staging ' + split + '/' + kind, bytes=True) as progress:
                        shutil.copyfileobj(progress, target, 8 * 1024 * 1024)
                os.replace(temporary, active)
        TOKENS[split][kind] = TokenStore(active, index)
length_report = {}
for split in TOKENS:
    length_report[split] = {}
    for role, store in TOKENS[split].items():
        lengths = np.asarray([length for offset, length in store.index.values()])
        length_report[split][role] = {'cases': len(lengths), 'tokens_p50': float(np.median(lengths)),
                                     'tokens_p90': float(np.quantile(lengths, 0.9)), 'tokens_max': int(lengths.max())}
save_json(length_report, PROJECT_DIR / 'document_lengths.json')
print('Token length audit:', length_report)
