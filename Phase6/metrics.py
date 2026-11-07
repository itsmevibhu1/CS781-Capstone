def metric_config():
    return {'model': BERTSCORE_MODEL, 'layers': BERTSCORE_LAYERS, 'rescale': BERTSCORE_RESCALE,
            'idf': False, 'rouge_use_stemmer': ROUGE_USE_STEMMER}


SCORER = None
SCORER_SIGNATURE = None
def get_scorer():
    global SCORER, SCORER_SIGNATURE
    signature = (BERTSCORE_MODEL, BERTSCORE_LAYERS, BERTSCORE_RESCALE, BERTSCORE_DEVICE)
    if SCORER is None or SCORER_SIGNATURE != signature:
        SCORER = None
        gc.collect()
        from bert_score import BERTScorer
        print('Loading evaluation-only BERTScore model:', BERTSCORE_MODEL, 'on', BERTSCORE_DEVICE)
        rng = capture_rng()
        try:
            SCORER = BERTScorer(model_type=BERTSCORE_MODEL, num_layers=BERTSCORE_LAYERS, lang='en',
                                rescale_with_baseline=BERTSCORE_RESCALE, idf=False,
                                device=BERTSCORE_DEVICE, batch_size=BERTSCORE_BATCH_SIZE)
            SCORER_SIGNATURE = signature
        finally:
            restore_rng(rng)
    return SCORER


def score_summaries(rows, split, keys):
    from rouge_score import rouge_scorer
    validate_rows(rows, split, keys)
    references = [DATA[split][key]['summary'] for key in keys]
    if any(not isinstance(reference, str) or not reference.strip() for reference in references):
        raise ValueError('Reference summaries required; test scoring is not available.')
    predictions = [row['summary'] for row in rows]
    scorer = get_scorer()
    with torch.no_grad():
        precision, recall, f1 = scorer.score(predictions, references, batch_size=BERTSCORE_BATCH_SIZE, verbose=True)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=ROUGE_USE_STEMMER)
    scores = [rouge.score(reference, prediction) for reference, prediction in
              tqdm(zip(references, predictions), total=len(keys), desc='ROUGE validation', unit='document')]
    result = {'bertscore_f1': float(f1.mean()), 'bertscore_precision': float(precision.mean()),
              'bertscore_recall': float(recall.mean()),
              'rouge1_f1': float(np.mean([row['rouge1'].fmeasure for row in scores])),
              'rougeL_f1': float(np.mean([row['rougeL'].fmeasure for row in scores])),
              'cases': len(keys), 'bert_score_hash': scorer.hash, 'metric_config': metric_config(),
              'mean_generated_words': float(np.mean([len(text.split()) for text in predictions])),
              'mean_reference_words': float(np.mean([len(text.split()) for text in references]))}
    assert all(math.isfinite(result[k]) for k in ['bertscore_f1', 'rouge1_f1', 'rougeL_f1'])
    return result


def show_scores(scores, title):
    print(title)
    print(f"BERTScore F1 [PRIMARY]: {scores['bertscore_f1']:.6f}; ROUGE-1 F1: {scores['rouge1_f1']:.6f}; ROUGE-L F1: {scores['rougeL_f1']:.6f}")
    print('Documents:', scores['cases'], '; mean generated/reference words:', scores['mean_generated_words'], '/', scores['mean_reference_words'])


def fixed_validation_keys(limit):
    keys = sorted(DATA['dev'], key=lambda key: hashlib.sha256((str(SEED) + key).encode()).hexdigest())
    return keys[:limit] if limit else keys


@torch.no_grad()
def validation_loss(model, cfg):
    model.eval(); keys = fixed_validation_keys(cfg['eval_loss_limit']); total = 0.0; tokens = 0
    for start in tqdm(range(0, len(keys), cfg['batch_size']), desc='Validation target loss', unit='batch'):
        batch = pack_batch('dev', keys[start:start + cfg['batch_size']], cfg)
        with amp_context(): logits = model(batch)
        count = int((batch['labels'] != -100).sum())
        value = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch['labels'].reshape(-1), ignore_index=-100, reduction='sum')
        total += float(value); tokens += count
    return total / max(1, tokens)
