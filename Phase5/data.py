def case_key(case_id):
    if isinstance(case_id, bool) or not isinstance(case_id, (str, int)):
        raise ValueError('Case IDs must be strings or integers.')
    return str(case_id)


def parse_statutes(value):
    import ast
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError('labels must contain a list of integer statute IDs.') from exc
    if not isinstance(value, list):
        raise ValueError('Expected a multi-label list, not a single class.')
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError('Statute IDs must be nonnegative integers (booleans are invalid).')
    return sorted(set(value))


def normalize_cases(raw, labeled, split=None):
    if isinstance(raw, dict) and 'id' in raw and 'text' in raw:
        entries = [(None, raw)]
    elif isinstance(raw, dict) and isinstance(raw.get('data'), list):
        entries = [(None, row) for row in raw['data']]
    elif isinstance(raw, dict):
        entries = list(raw.items())
    elif isinstance(raw, list):
        entries = [(None, row) for row in raw]
    else:
        raise ValueError('Expected case objects or an ID-keyed mapping.')
    result = {}
    for index, (outer_id, record) in enumerate(entries):
        if isinstance(record, str) and not labeled and outer_id is not None:
            record = {'id': outer_id, 'text': record}
        if not isinstance(record, dict):
            raise ValueError(f'{split}: invalid case object at row {index}.')
        original_id = record.get('id', outer_id)
        key = case_key(original_id)
        if outer_id is not None and str(outer_id) != key:
            raise ValueError('Mapping key disagrees with embedded case ID.')
        text = record.get('text')
        if isinstance(text, list) and all(isinstance(x, str) for x in text):
            text = '\n\n'.join(text)
        if not isinstance(text, str):
            raise ValueError(f'{split} row {index}: text must be a string or ordered string fragments.')
        labels = parse_statutes(record.get('labels', record.get('label'))) if labeled else None
        internal_key = f'{split}:row:{index}' if split in ('train', 'dev') else key
        if internal_key in result:
            raise ValueError(f'{split}: duplicate original test ID {key}; no rows were overwritten.')
        result[internal_key] = {'id': original_id, 'text': text, 'label': labels}
    return result


def resolve_split_overlaps(data, policy):
    if policy not in ('exclude_train', 'error', 'report'):
        raise ValueError('SPLIT_OVERLAP_POLICY must be exclude_train, error or report.')
    indexes = {}
    for split, docs in data.items():
        index = {}
        for key, doc in tqdm(docs.items(), desc='Auditing exact texts: ' + split, unit='case'):
            if doc['text'].strip():
                digest = hashlib.sha256(doc['text'].encode()).hexdigest()
                index.setdefault(digest, []).append(key)
        indexes[split] = index
    report = {'policy': policy, 'source_counts': {split: len(docs) for split, docs in data.items()}}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        shared_ids = {case_key(d['id']) for d in data[a].values()} & {case_key(d['id']) for d in data[b].values()}
        shared_texts = set(indexes[a]) & set(indexes[b])
        examples = []
        for digest in sorted(shared_texts)[:20]:
            akeys = indexes[a][digest]; bkeys = indexes[b][digest]
            example = {'sha256': digest, 'characters': len(data[a][akeys[0]]['text']),
                       a + '_ids': [data[a][key]['id'] for key in akeys[:5]],
                       b + '_ids': [data[b][key]['id'] for key in bkeys[:5]]}
            if a != 'test' and b != 'test':
                label_sets = {tuple(data[split][key]['label']) for split, keys in [(a, akeys), (b, bkeys)] for key in keys}
                example['different_label_sets'] = len(label_sets) > 1
            examples.append(example)
        report[a + '/' + b] = {'shared_original_ids': len(shared_ids), 'identical_texts': len(shared_texts),
                              a + '_matching_instances': sum(len(indexes[a][h]) for h in shared_texts),
                              b + '_matching_instances': sum(len(indexes[b][h]) for h in shared_texts),
                              'examples': examples}
        print(f"Split audit {a}/{b}: {len(shared_ids)} shared ID values; {len(shared_texts)} identical nonempty texts.")
    heldout = set(indexes['dev']) | set(indexes['test'])
    remove = {key for digest in set(indexes['train']) & heldout for key in indexes['train'][digest]}
    filtered = dict(data)
    removed = []
    if policy == 'exclude_train':
        removed = [{'instance_key': key, 'id': doc['id'], 'sha256': hashlib.sha256(doc['text'].encode()).hexdigest()}
                   for key, doc in data['train'].items() if key in remove]
        filtered['train'] = {key: doc for key, doc in data['train'].items() if key not in remove}
    report['excluded_train_instances'] = len(removed)
    report['excluded_train_examples'] = removed[:20]
    report['retained_counts'] = {split: len(docs) for split, docs in filtered.items()}
    report['remaining_train_instances_matching_heldout_text'] = 0 if policy == 'exclude_train' else len(remove)
    save_json(report, PROJECT_DIR / 'split_audit.json')
    save_json(removed, PROJECT_DIR / 'excluded_train_instances.json')
    if policy == 'error' and any(report[a + '/' + b]['identical_texts'] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
        raise ValueError('Identical nonempty texts across splits; see split_audit.json. Use exclude_train to remove held-out copies from training.')
    if not filtered['train']:
        raise ValueError('No training cases remain after overlap filtering; inspect split_audit.json before choosing another policy.')
    if policy == 'exclude_train':
        retained_hashes = {hashlib.sha256(doc['text'].encode()).hexdigest() for doc in filtered['train'].values() if doc['text'].strip()}
        assert not (retained_hashes & heldout), 'Overlap filtering left held-out copies in training.'
        print(f"Excluded {len(removed)} training instances matching dev/test text; kept {len(filtered['train'])} training instances.")
    if report['dev/test']['identical_texts']:
        print('Official dev/test contain identical texts. All dev/test cases and IDs are retained; overlap counts are recorded in split_audit.json.')
    if policy == 'report' and remove:
        print('Report-only policy: matching training instances are retained. Dev scores include exact-text overlap with training.')
    return filtered, report


DATA = {split: load_case_split(split) for split in FILES}
# Catalog IDs come from supplied labeled files, including excluded training instances.
observed = {x for split in ('train', 'dev') for row in DATA[split].values() for x in row['label']}
if not observed:
    raise ValueError('No statute labels in TRAIN/DEV.')
if STATUTE_IDS is not None:
    LABEL_IDS = parse_statutes(STATUTE_IDS)
    if not observed <= set(LABEL_IDS):
        raise ValueError('Observed labels are absent from the supplied official statute catalog.')
else:
    LABEL_IDS = sorted(observed)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_IDS)}
NUM_LABELS = len(LABEL_IDS)
DATA, split_audit = resolve_split_overlaps(DATA, SPLIT_OVERLAP_POLICY)


def encode_labels(labels):
    array = np.zeros(NUM_LABELS, dtype=np.float32)
    if labels is not None:
        for label in labels:
            array[LABEL_TO_INDEX[label]] = 1.0
    return array


TRAIN_COUNTS = np.zeros(NUM_LABELS, dtype=np.int64)
DEV_COUNTS = np.zeros(NUM_LABELS, dtype=np.int64)
audit = {}
for split, docs in DATA.items():
    if not docs:
        raise ValueError(f'Empty {split} split.')
    counter = Counter(case_key(row['id']) for row in docs.values())
    audit[split] = {'instances': len(docs), 'unique_ids': len(counter),
                    'repeated_ids': sum(v > 1 for v in counter.values()),
                    'empty_texts': sum(not row['text'].strip() for row in docs.values())}
    if split != 'test':
        counts = TRAIN_COUNTS if split == 'train' else DEV_COUNTS
        for row in docs.values():
            counts += encode_labels(row['label']).astype(np.int64)
        audit[split]['mean_labels'] = float(np.mean([len(row['label']) for row in docs.values()]))
    print(split, audit[split])
fingerprint = hashlib.sha256(json.dumps([LABEL_IDS, SPLIT_OVERLAP_POLICY]).encode())
for split in sorted(DATA):
    for key, doc in DATA[split].items():
        fingerprint.update(json.dumps([split, key, doc], sort_keys=True, ensure_ascii=False).encode())
DATA_FINGERPRINT = fingerprint.hexdigest()
save_json({'files': FILES, 'fingerprint': DATA_FINGERPRINT, 'counts': audit,
           'source_counts': split_audit['source_counts'], 'overlap_policy': SPLIT_OVERLAP_POLICY,
           'excluded_train_instances': split_audit['excluded_train_instances']}, PROJECT_DIR / 'dataset_manifest.json')
save_json({'label_ids': LABEL_IDS, 'num_labels': NUM_LABELS, 'source': 'official override' if STATUTE_IDS is not None else 'train+dev',
           'train_counts': TRAIN_COUNTS.tolist(), 'dev_counts': DEV_COUNTS.tolist(),
           'unseen_in_train': [x for x, n in zip(LABEL_IDS, TRAIN_COUNTS) if not n],
           'absent_from_dev': [x for x, n in zip(LABEL_IDS, DEV_COUNTS) if not n]}, PROJECT_DIR / 'label_catalog.json')
print('Output labels:', NUM_LABELS, '; IDs:', LABEL_IDS)
print('Labels absent from TRAIN:', [x for x, n in zip(LABEL_IDS, TRAIN_COUNTS) if not n])
