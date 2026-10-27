def gold_matrix(split, keys=None):
    return np.stack([encode_labels(DATA[split][key]['label']) for key in (keys or list(DATA[split]))]).astype(np.int64)


def multilabel_metrics(gold, predicted):
    y = np.asarray(gold)
    p = np.asarray(predicted)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != NUM_LABELS or not len(y):
        raise ValueError('Expected nonempty case-by-statute matrices.')
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('Expected binary multi-hot matrices.')
    y = y.astype(np.int64); p = p.astype(np.int64)
    tp = (y * p).sum(0); fp = ((1 - y) * p).sum(0); fn = (y * (1 - p)).sum(0)
    support = y.sum(0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(NUM_LABELS), where=(2 * tp + fp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros(NUM_LABELS), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(NUM_LABELS), where=tp + fn > 0)
    present = support > 0
    reference_max = max((label for label, n in zip(LABEL_IDS, support) if n), default=-1)
    known_macro = float(f1.mean())
    present_macro = float(f1[present].mean()) if present.any() else 0.0
    range_macro = float(sum(v for label, v in zip(LABEL_IDS, f1) if label <= reference_max) / (reference_max + 1)) if reference_max >= 0 else 0.0
    primary = {'known': known_macro, 'reference_present': present_macro, 'reference_range': range_macro}[METRIC_LABEL_SCOPE]
    mask = np.ones(NUM_LABELS, dtype=bool)
    if METRIC_LABEL_SCOPE == 'reference_present':
        mask = present
    elif METRIC_LABEL_SCOPE == 'reference_range':
        mask = np.asarray([label <= reference_max for label in LABEL_IDS])
    total_tp = int(tp[mask].sum()); total_fp = int(fp[mask].sum()); total_fn = int(fn[mask].sum())
    denominator = 2 * total_tp + total_fp + total_fn
    known_denominator = int(2 * tp.sum() + fp.sum() + fn.sum())
    return {'macro_f1': primary, 'macro_f1_known': known_macro, 'macro_f1_reference_present': present_macro,
            'macro_f1_reference_range': range_macro, 'metric_label_scope': METRIC_LABEL_SCOPE,
            'micro_f1': 2 * total_tp / denominator if denominator else 0.0,
            'subset_accuracy': float(np.all(y[:, mask] == p[:, mask], axis=1).mean()),
            'micro_f1_known': float(2 * tp.sum() / known_denominator) if known_denominator else 0.0,
            'subset_accuracy_known': float(np.all(y == p, axis=1).mean()),
            'precision': total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
            'recall': total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
            'cases': len(y), 'mean_predicted_labels': float(p.sum(1).mean()),
            'per_label': {str(label): {'precision': float(pr), 'recall': float(re), 'f1': float(f),
                          'support': int(s), 'train_support': int(t), 'tp': int(a), 'fp': int(b), 'fn': int(c)}
                          for label, pr, re, f, s, t, a, b, c in zip(LABEL_IDS, precision, recall, f1, support, TRAIN_COUNTS, tp, fp, fn)}}


def decisions(probabilities, policy):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != NUM_LABELS or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError('Invalid case-by-statute probabilities.')
    thresholds = np.asarray(policy['thresholds'], dtype=float)
    if thresholds.shape != (NUM_LABELS,) or not np.isfinite(thresholds).all():
        raise ValueError('Invalid threshold vector.')
    selected = p >= thresholds
    if not policy['allow_empty']:
        empty = ~selected.any(1)
        selected[np.flatnonzero(empty), p[empty].argmax(1)] = True
    return selected.astype(np.int64)


def exact_label_threshold(p, y):
    # Positive-class F1 for a single statute; this is not binary outcome Macro-F1.
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=np.int64)
    if y.sum() == 0:
        return 0.5
    order = np.argsort(-p, kind='stable'); ps = p[order]; ys = y[order]
    ends = np.flatnonzero(np.r_[ps[:-1] != ps[1:], True])
    tp = np.cumsum(ys)[ends]; count = ends + 1
    scores = 2 * tp / (int(y.sum()) + count)
    best = scores.max()
    options = [(abs(float(t) - 0.5), float(t)) for t, s in zip(ps[ends], scores) if abs(float(s) - best) < 1e-12]
    # Retain 0.5 on a tie rather than using an arbitrary boundary.
    baseline_pred = p >= 0.5
    base_den = int(y.sum()) + int(baseline_pred.sum())
    base_score = 2 * int((baseline_pred * y).sum()) / base_den if base_den else 0.0
    if abs(base_score - best) < 1e-12:
        options.append((0.0, 0.5))
    return min(options)[1]


def optimal_threshold(probabilities, gold, tune=True, cfg=None):
    cfg = cfg or {'threshold_mode': THRESHOLD_MODE, 'threshold_grid': THRESHOLD_GRID,
                  'min_dev_positives': MIN_DEV_POSITIVES, 'threshold_shrinkage': THRESHOLD_SHRINKAGE,
                  'allow_empty': ALLOW_EMPTY_PREDICTIONS}
    p = np.asarray(probabilities, dtype=float); y = np.asarray(gold, dtype=np.int64)
    baseline_policy = {'mode': 'fixed_0.5', 'thresholds': [0.5] * NUM_LABELS, 'allow_empty': cfg['allow_empty']}
    baseline = multilabel_metrics(y, decisions(p, baseline_policy))
    if not tune:
        return baseline_policy, baseline, baseline
    # Compute candidate scores from confusion counts, avoiding per-label JSON reports per grid point.
    def score(policy):
        pred = decisions(p, policy)
        tp = (pred * y).sum(0); den = y.sum(0) + pred.sum(0)
        f = np.divide(2 * tp, den, out=np.zeros(NUM_LABELS), where=den > 0)
        support = y.sum(0)
        if METRIC_LABEL_SCOPE == 'reference_present':
            return float(f[support > 0].mean()) if support.any() else 0.0
        if METRIC_LABEL_SCOPE == 'reference_range':
            maximum = max((label for label, n in zip(LABEL_IDS, support) if n), default=-1)
            return float(sum(v for label, v in zip(LABEL_IDS, f) if label <= maximum) / (maximum + 1)) if maximum >= 0 else 0.0
        return float(f.mean())
    best = baseline_policy; best_score = baseline['macro_f1']
    for threshold in sorted(set(cfg['threshold_grid']), key=lambda t: (abs(t - 0.5), t)):
        policy = {'mode': 'global', 'thresholds': [float(threshold)] * NUM_LABELS, 'allow_empty': cfg['allow_empty']}
        value = score(policy)
        if value > best_score + 1e-12:
            best = policy; best_score = value
    global_policy = copy.deepcopy(best)
    if cfg['threshold_mode'] in ('auto', 'per_label'):
        support = y.sum(0); thresholds = list(global_policy['thresholds'])
        for j in range(NUM_LABELS):
            if support[j] >= cfg['min_dev_positives']:
                local = exact_label_threshold(p[:, j], y[:, j])
                weight = support[j] / (support[j] + cfg['threshold_shrinkage'])
                thresholds[j] = float(weight * local + (1 - weight) * thresholds[j])
        policy = {'mode': 'shrunk_per_label', 'thresholds': thresholds, 'allow_empty': cfg['allow_empty']}
        value = score(policy)
        if value > best_score + 1e-12:
            best = policy; best_score = value
    selected = multilabel_metrics(y, decisions(p, best))
    assert selected['macro_f1'] + 1e-12 >= baseline['macro_f1']
    return best, selected, baseline


def show_scores(metrics, title):
    print(title)
    print(f"Macro-F1 [PRIMARY, {metrics['metric_label_scope']}]: {metrics['macro_f1']:.6f}")
    print(f"Micro-F1: {metrics['micro_f1']:.6f}; Subset Accuracy: {metrics['subset_accuracy']:.6f}")
    print(f"Micro precision/recall: {metrics['precision']:.6f}/{metrics['recall']:.6f}")
    print('Macro-F1 conventions:', {k: metrics[k] for k in ('macro_f1_reference_range', 'macro_f1_reference_present', 'macro_f1_known')})


def probability_vector(logits):
    return torch.sigmoid(logits.double()).numpy()


def make_prediction_rows(split, keys, probabilities, policy):
    chosen = decisions(probabilities, policy)
    return [{'id': DATA[split][key]['id'], 'pred': [int(label) for label, yes in zip(LABEL_IDS, row) if yes]}
            for key, row in zip(keys, chosen)]


def validate_submission(predictions, docs):
    if not isinstance(predictions, list) or len(predictions) != len(docs):
        raise ValueError('Expected exactly one prediction per input instance.')
    expected = {}
    for key, doc in docs.items():
        expected.setdefault((type(doc['id']), doc['id']), []).append(key)
    consumed = Counter(); matched = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {'id', 'pred'}:
            raise ValueError('Expected exactly id and pred fields.')
        case_key(row['id']); identity = (type(row['id']), row['id']); occurrence = consumed[identity]
        if identity not in expected or occurrence >= len(expected[identity]):
            raise ValueError('Unknown, duplicate or changed original case ID.')
        labels = row['pred']
        if not isinstance(labels, list) or any(type(x) is not int or x not in LABEL_TO_INDEX for x in labels):
            raise ValueError('pred must contain valid integer statute IDs only.')
        if len(labels) != len(set(labels)):
            raise ValueError('Duplicate statute IDs within a prediction.')
        matched[expected[identity][occurrence]] = labels; consumed[identity] += 1
    if set(matched) != set(docs):
        raise ValueError('Missing prediction instances.')
    return matched


def save_predictions(output, probabilities, policy):
    rows = make_prediction_rows('test', list(DATA['test']), probabilities, policy)
    validate_submission(rows, DATA['test']); save_json(rows, output / 'prediction.json')
    assert json.loads((output / 'prediction.json').read_text()) == rows
    temporary = output / 'submission.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / 'prediction.json', arcname='prediction.json')
    os.replace(temporary, output / 'submission.zip')
    with zipfile.ZipFile(output / 'submission.zip') as archive:
        assert archive.namelist() == ['prediction.json']
        validate_submission(json.loads(archive.read('prediction.json')), DATA['test'])
        assert json.loads(archive.read('prediction.json')) == rows
    return rows


def save_probabilities(output, split, keys, p):
    save_json({'label_ids': LABEL_IDS, 'rows': [{'id': DATA[split][key]['id'], 'instance_key': key,
              'probabilities': [float(v) for v in values]} for key, values in zip(keys, p)]},
              output / (split + '_probabilities.json'))


def load_probability_file(path, split):
    saved = json.loads((path / (split + '_probabilities.json')).read_text())
    if saved['label_ids'] != LABEL_IDS:
        raise ValueError('Probability label ordering differs.')
    rows = saved['rows']; mapping = {row['instance_key']: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(DATA[split]):
        raise ValueError('Probability instance keys differ.')
    for key, row in mapping.items():
        original = DATA[split][key]['id']
        if type(row['id']) is not type(original) or row['id'] != original:
            raise ValueError('Probability cache original ID changed.')
    values = np.asarray([mapping[key]['probabilities'] for key in DATA[split]], dtype=float)
    if values.shape != (len(DATA[split]), NUM_LABELS) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Invalid saved probabilities.')
    return values
