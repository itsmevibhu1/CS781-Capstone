def validate_predictions(predictions, split):
    if not isinstance(predictions, dict) or set(predictions) != set(QUERIES[split]):
        raise ValueError('Predictions must contain every query ID exactly once.')
    for query, values in predictions.items():
        if not isinstance(values, list) or len(values) != TOP_K:
            raise ValueError(f'{query}: expected exactly seven unique candidate IDs.')
        if any(type(value) is not str or value not in CANDIDATES[split] for value in values):
            raise ValueError(f'{query}: candidate IDs must be original strings from the {split} candidate pool.')
        if len(set(values)) != TOP_K: raise ValueError(f'{query}: candidate IDs must be unique.')
        if EXCLUDE_SELF_MATCH and query in values: raise ValueError('Self candidate was excluded by configuration.')
    return True


def retrieval_metrics(predictions, split='dev'):
    if split == 'test': raise ValueError('Public test gold is private; no test score can be computed.')
    validate_predictions(predictions, split)
    tp = 0; gold_count = 0; per_query = {}; ceiling = 0
    for query, row in QUERIES[split].items():
        gold = set(row['relevant']); selected = set(predictions[query][:TOP_K]); hits = len(gold & selected)
        tp += hits; gold_count += len(gold); ceiling += min(TOP_K, len(gold))
        per_query[query] = {'hits_at_7': hits, 'gold_count': len(gold), 'precision_at_7': hits / TOP_K,
                            'recall_at_7': hits / len(gold) if gold else 0.0,
                            'missed': sorted(gold - selected), 'irrelevant_retrieved': sorted(selected - gold)}
    predicted_count = len(predictions) * TOP_K; denominator = predicted_count + gold_count
    return {'micro_f1_at_7': 2 * tp / denominator if denominator else 0.0,
            'micro_precision_at_7': tp / predicted_count if predicted_count else 0.0,
            'micro_recall_at_7': tp / gold_count if gold_count else 0.0,
            'true_positives': tp, 'predicted_count': predicted_count, 'gold_count': gold_count,
            'queries': len(predictions), 'oracle_micro_f1_at_7': 2 * ceiling / denominator if denominator else 0.0,
            'per_query': per_query}


def show_scores(metrics, title):
    print(title)
    print(f"Micro-F1@7 [PRIMARY]: {metrics['micro_f1_at_7']:.6f}; Micro-Precision@7: {metrics['micro_precision_at_7']:.6f}; Micro-Recall@7: {metrics['micro_recall_at_7']:.6f}")
    print('Queries:', metrics['queries'], '; hits:', metrics['true_positives'], '; complete gold count:', metrics['gold_count'])


def save_submission(output, predictions):
    validate_predictions(predictions, 'test'); save_json(predictions, output / 'prediction.json')
    temporary = output / 'submission.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / 'prediction.json', arcname='prediction.json')
    os.replace(temporary, output / 'submission.zip')
    with zipfile.ZipFile(output / 'submission.zip') as archive:
        assert archive.namelist() == ['prediction.json']
        saved = json.loads(archive.read('prediction.json')); validate_predictions(saved, 'test'); assert saved == predictions


def stable_order(scores, ids):
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(ids) or not np.isfinite(values).all(): raise ValueError('Nonfinite or misaligned ranking scores.')
    return np.lexsort((np.asarray(ids, dtype=str), -values))


def allowed_mask(query, ids):
    return np.asarray([not EXCLUDE_SELF_MATCH or candidate != query for candidate in ids])


def top_predictions(scores, ids, query):
    order = stable_order(scores, ids)
    return [ids[index] for index in order if not EXCLUDE_SELF_MATCH or ids[index] != query][:TOP_K]
