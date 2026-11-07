def original_key(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('Document IDs must be strings or integers.')
    return str(value)


NORMALIZATION_AUDIT = {}
def normalize_summaries(raw, labeled, invalid_policy=None, max_invalid_fraction=None):
    split = 'train' if labeled else 'test'
    policy = (invalid_policy or INVALID_TRAIN_ROW_POLICY) if labeled else 'error'
    limit = MAX_INVALID_TRAIN_FRACTION if max_invalid_fraction is None else max_invalid_fraction
    if policy not in ('exclude', 'error') or not 0 <= limit <= 1:
        raise ValueError('Invalid training-row policy or fraction limit.')
    if isinstance(raw, dict) and ('document' in raw or 'text' in raw) and 'id' in raw:
        entries = [(None, raw)]
    elif isinstance(raw, dict) and isinstance(raw.get('data'), list):
        entries = [(None, row) for row in raw['data']]
    elif isinstance(raw, dict):
        entries = list(raw.items())
    elif isinstance(raw, list):
        entries = [(None, row) for row in raw]
    else:
        raise ValueError('Expected a JSON document list or ID-keyed mapping.')
    result = {}; invalid = []; normalized = Counter()
    def record_audit():
        NORMALIZATION_AUDIT[split] = {'input_rows': len(entries), 'usable_rows': len(result),
            'invalid_rows': len(invalid), 'policy': policy,
            'excluded_rows': len(invalid) if policy == 'exclude' else 0,
            'reason_counts': dict(Counter(reason for row in invalid for reason in row['reasons'])),
            'paragraph_lists_joined': dict(normalized), 'invalid_examples': invalid[:20]}
        save_json(NORMALIZATION_AUDIT, PROJECT_DIR / 'data_quality_report.json')
        if labeled:
            save_json(invalid if policy == 'exclude' else [], PROJECT_DIR / 'excluded_training_rows.json')
    for i, (outer, row) in enumerate(entries):
        if not labeled and isinstance(row, str):
            row = {'id': outer, 'document': row}
        if not isinstance(row, dict):
            raise ValueError(f'Invalid document at row {i}.')
        doc_id = row.get('id', outer); key = original_key(doc_id)
        if outer is not None and str(outer) != key:
            raise ValueError('Mapping key differs from original document ID.')
        text = row.get('document', row.get('text'))
        if isinstance(text, list) and all(isinstance(t, str) for t in text):
            text = '\n\n'.join(text)
            normalized['document'] += 1
        summary = row.get('summary') if labeled else None
        if labeled and isinstance(summary, list) and all(isinstance(t, str) for t in summary):
            summary = '\n\n'.join(summary)
            normalized['summary'] += 1
        reasons = []
        if not isinstance(text, str):
            reasons.append('missing_document' if text is None else 'unsupported_document_type')
        if labeled:
            if not isinstance(summary, str):
                reasons.append('missing_summary' if summary is None else 'unsupported_summary_type')
            elif not summary.strip():
                reasons.append('empty_summary')
        if reasons:
            invalid.append({'row_index': i, 'id': doc_id, 'reasons': reasons, 'available_fields': sorted(row),
                            'document_type': type(text).__name__, 'summary_type': type(summary).__name__,
                            'document_characters': len(text) if isinstance(text, str) else None,
                            'summary_characters': len(summary.strip()) if isinstance(summary, str) else None})
            if policy == 'error':
                record_audit()
                raise ValueError(f'{split} ID {key}: {", ".join(reasons)} '
                                 f'(document={type(text).__name__}, summary={type(summary).__name__}). '
                                 'See data_quality_report.json; test rows are never skipped.')
            continue
        internal = f'public:row:{i}' if labeled else key
        if internal in result:
            raise ValueError(f'Duplicate original test ID {key}. No document was overwritten.')
        result[internal] = {'id': doc_id, 'text': text, 'summary': summary}
    record_audit()
    if labeled and invalid and len(invalid) / max(1, len(entries)) > limit:
        raise ValueError(f'{len(invalid)}/{len(entries)} training rows are unusable, above MAX_INVALID_TRAIN_FRACTION={limit}. '
                         'Inspect data_quality_report.json for missing fields/types before changing this limit.')
    print(f'{split}: retained {len(result)}/{len(entries)} rows; unusable rows: {len(invalid)}; paragraph lists joined: {dict(normalized)}')
    for row in invalid[:5]:
        print('  Excluded training ID:', row['id'], '; reasons:', ', '.join(row['reasons']))
    return result


PUBLIC = normalize_summaries(load_download('train'), True)
TEST = normalize_summaries(load_download('test'), False)
if not PUBLIC or not TEST:
    raise ValueError('Both official files must be nonempty.')
def document_hash(row):
    return hashlib.sha256(row['text'].encode()).hexdigest()
test_hashes = {document_hash(row) for row in TEST.values() if row['text'].strip()}
excluded = [key for key, row in PUBLIC.items() if row['text'].strip() and document_hash(row) in test_hashes] if EXCLUDE_TRAIN_TEST_OVERLAP else []
excluded_set = set(excluded)
groups = {}
for key, row in PUBLIC.items():
    if key not in excluded_set:
        groups.setdefault(document_hash(row), []).append(key)
if len(groups) < 2:
    raise ValueError('Need at least two distinct training documents after overlap filtering.')
ordered = sorted(groups, key=lambda h: hashlib.sha256((str(SEED) + h).encode()).hexdigest())
count = max(1, min(len(ordered) - 1, round(len(ordered) * VALIDATION_FRACTION)))
validation_hashes = set(ordered[:count])
DATA = {'train': {}, 'dev': {}, 'test': TEST}
for key, row in PUBLIC.items():
    if key not in excluded_set:
        DATA['dev' if document_hash(row) in validation_hashes else 'train'][key] = row
assert not ({document_hash(row) for row in DATA['train'].values()} & {document_hash(row) for row in DATA['dev'].values()})
fingerprint = hashlib.sha256(json.dumps([SEED, VALIDATION_FRACTION, EXCLUDE_TRAIN_TEST_OVERLAP]).encode())
for split, rows in DATA.items():
    for key, row in rows.items():
        fingerprint.update(json.dumps([split, key, row], sort_keys=True, ensure_ascii=False).encode())
DATA_FINGERPRINT = fingerprint.hexdigest()
save_json({'fingerprint': DATA_FINGERPRINT, 'official_files': FILES, 'validation_seed': SEED,
           'validation_fraction': VALIDATION_FRACTION, 'public_instances': len(PUBLIC),
           'raw_training_instances': NORMALIZATION_AUDIT['train']['input_rows'],
           'excluded_invalid_training_instances': NORMALIZATION_AUDIT['train']['excluded_rows'],
           'counts': {s: len(d) for s, d in DATA.items()}, 'excluded_train_test_matches': len(excluded),
           'train_ids': list(DATA['train']), 'validation_ids': list(DATA['dev']),
           'excluded_ids': [{'key': key, 'id': PUBLIC[key]['id']} for key in excluded]}, PROJECT_DIR / 'split_manifest.json')
del PUBLIC
print('Fixed grouped split:', {split: len(rows) for split, rows in DATA.items()}, '; excluded test copies:', len(excluded))

TOKENIZER_DIR = PHASE1_DIR / 'bpe_tokenizer'
tok = ByteLevelBPETokenizer(str(TOKENIZER_DIR / 'vocab.json'), str(TOKENIZER_DIR / 'merges.txt'))
VOCAB_SIZE_ACTUAL = tok.get_vocab_size()
def special_id(names):
    for name in names:
        value = tok.token_to_id(name)
        if value is not None:
            return value
    raise ValueError('Missing Phase 1 special token: ' + repr(names))
PAD_ID = special_id(['<pad>', '[PAD]'])
BOS_ID = special_id(['<bos>', '<s>', '[BOS]'])
EOS_ID = special_id(['<eos>', '</s>', '[EOS]'])
UNK_ID = special_id(['<unk>', '[UNK]'])
TOKENIZER_HASH = hashlib.sha256((TOKENIZER_DIR / 'vocab.json').read_bytes() + (TOKENIZER_DIR / 'merges.txt').read_bytes()).hexdigest()
for filename in ['vocab.json', 'merges.txt']:
    (PROJECT_DIR / 'bpe_tokenizer').mkdir(exist_ok=True)
    shutil.copy2(TOKENIZER_DIR / filename, PROJECT_DIR / 'bpe_tokenizer' / filename)


class TokenStore:
    def __init__(self, path, index):
        self.data = np.memmap(path, mode='r', dtype='<i4') if path.stat().st_size else np.empty(0, dtype='<i4')
        self.index = index
    def __getitem__(self, key):
        offset, length = self.index[key]
        return self.data[offset:offset + length].tolist()


TOKENS = {}; REFERENCES = {}
for split, rows in DATA.items():
    for field, store in [('text', TOKENS), ('summary', REFERENCES)]:
        if split == 'test' and field == 'summary':
            continue
        stem = f'{split}_{field}_{DATA_FINGERPRINT[:16]}_{TOKENIZER_HASH[:12]}'
        binary = PROJECT_DIR / (stem + '.bin'); index_path = PROJECT_DIR / (stem + '.json')
        if not binary.is_file() or not index_path.is_file():
            temporary = binary.with_name(binary.name + '.' + __import__('uuid').uuid4().hex + '.tmp')
            index = {}; offset = 0
            with temporary.open('wb') as stream:
                for key, row in tqdm(rows.items(), desc='Tokenizing ' + split + '/' + field, unit='document'):
                    ids = np.asarray(tok.encode(row[field], add_special_tokens=False).ids, dtype='<i4')
                    stream.write(ids.tobytes()); index[key] = [offset, len(ids)]; offset += len(ids)
            os.replace(temporary, binary); save_json(index, index_path)
        index = json.loads(index_path.read_text())
        if set(index) != set(rows):
            raise ValueError('Token cache IDs disagree.')
        active = binary
        if IN_COLAB:
            staging = Path('/content/cs781_phase6_tokens'); staging.mkdir(exist_ok=True)
            active = staging / binary.name
            if not active.is_file() or active.stat().st_size != binary.stat().st_size:
                temporary = active.with_name(active.name + '.tmp')
                with binary.open('rb') as source, temporary.open('wb') as target:
                    with tqdm.wrapattr(source, 'read', total=binary.stat().st_size, desc='Staging ' + stem, bytes=True) as progress:
                        shutil.copyfileobj(progress, target, 8 * 1024 * 1024)
                os.replace(temporary, active)
        store[split] = TokenStore(active, index)


def select_source(text, ids, budget, policy):
    if len(ids) <= budget:
        return ids or [UNK_ID], 1.0
    if policy == 'head':
        return ids[:budget], budget / len(ids)
    if policy == 'tail':
        return ids[-budget:], budget / len(ids)
    if policy == 'head_tail':
        first = (budget + 1) // 2
        return ids[:first] + ids[-(budget - first):] if budget > first else ids[:budget], budget / len(ids)
    if policy == 'spread':
        windows = min(4, budget); lengths = [budget // windows + (i < budget % windows) for i in range(windows)]
        boundaries = np.linspace(0, len(ids), windows + 1).astype(int)
        starts = [boundaries[i] if i < windows - 1 else len(ids) - lengths[i] for i in range(windows)]
        return [token for start, size in zip(starts, lengths) for token in ids[start:start + size]], budget / len(ids)
    if policy == 'sentence_mmr':
        import re
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        words = [Counter(re.findall(r'[a-z]+', sentence.lower())) for sentence in sentences]
        centroid = sum(words, Counter()); norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
        selected = []; chosen = []; used = 0
        candidates = []
        for i, (sentence, vector) in enumerate(zip(sentences, words)):
            tokens = tok.encode(sentence, add_special_tokens=False).ids
            if not tokens or len(tokens) > budget:
                continue
            relevance = sum(v * centroid[k] for k, v in vector.items()) / (norm * (math.sqrt(sum(v * v for v in vector.values())) or 1))
            candidates.append((i, tokens, set(vector), relevance))
        while candidates:
            ranked = [(0.7 * item[3] - 0.3 * max((len(item[2] & old) / max(1, len(item[2] | old)) for old in chosen), default=0), j)
                      for j, item in enumerate(candidates) if used + len(item[1]) <= budget]
            if not ranked:
                break
            _, j = max(ranked); item = candidates.pop(j); selected.append(item); chosen.append(item[2]); used += len(item[1])
        if not selected:
            return select_source(text, ids, budget, 'head_tail')
        return [token for item in sorted(selected) for token in item[1]], min(1.0, used / len(ids))
    raise ValueError('Unknown source selection policy.')
