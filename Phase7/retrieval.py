@torch.no_grad()
def encode_pool(model, split, role, cfg):
    was_training = model.training; model.eval(); records = QUERIES[split] if role == 'queries' else CANDIDATES[split]
    keys = list(records); vectors = []; chunks = []; owners = []; offset = 0; coverage = []
    for start in tqdm(range(0, len(keys), cfg['encode_batch_size']), desc='Encoding ' + split + '/' + role, unit='batch'):
        selected = keys[start:start + cfg['encode_batch_size']]; batch = make_chunks(split, role, selected, cfg)
        with amp_context(): document, window, owner = model(batch, role, return_chunks=True)
        vectors.append(document.float().cpu()); chunks.append(window.float().cpu()); owners.append(owner.cpu() + offset)
        coverage.extend(batch['coverage']); offset += len(selected)
    model.train(was_training)
    result = {'ids': keys, 'vectors': torch.cat(vectors), 'chunks': torch.cat(chunks), 'owners': torch.cat(owners),
              'mean_token_coverage': float(np.mean(coverage))}
    assert result['vectors'].shape == (len(keys), cfg['embedding_dim'])
    if not torch.isfinite(result['vectors']).all() or not torch.isfinite(result['chunks']).all(): raise FloatingPointError('Nonfinite retrieval embeddings.')
    return result


def citation_signal(text, ids):
    import re
    referenced = set(re.findall(r'(?:indiankanoon\.org/)?doc/(\d+)', text))
    normalized = {str(int(value)) for value in referenced}
    return np.asarray([float(candidate.isdigit() and str(int(candidate)) in normalized) for candidate in ids])


def ranking_components(query_embeddings, candidate_embeddings, split, cfg):
    if query_embeddings['ids'] != list(QUERIES[split]) or candidate_embeddings['ids'] != list(CANDIDATES[split]):
        raise ValueError('Embeddings do not align with this split/pool.')
    ids = candidate_embeddings['ids']; qpositions = {key: i for i, key in enumerate(query_embeddings['ids'])}
    result = {}
    for query in tqdm(QUERIES[split], desc='Ranking ' + split + ' against its own pool', unit='query'):
        i = qpositions[query]
        dense = (query_embeddings['vectors'][i:i + 1] @ candidate_embeddings['vectors'].T)[0].numpy()
        lexical = BM25_RANKS[split][query]
        if len(lexical) < min(cfg['rrf_depth'], len(ids)):
            lexical = stable_order(BM25[split].scores(QUERIES[split][query]['text']), ids)[:cfg['rrf_depth']].astype(np.int32)
            BM25_RANKS[split][query] = lexical
        components = {'dense': stable_order(dense, ids), 'bm25': lexical}
        if cfg['search_late_interaction']:
            q = query_embeddings['chunks'][query_embeddings['owners'] == i]
            similarities = q @ candidate_embeddings['chunks'].T
            maximum = torch.full((len(q), len(ids)), -float('inf'))
            maximum.scatter_reduce_(1, candidate_embeddings['owners'][None].expand(len(q), -1), similarities, reduce='amax', include_self=True)
            late = maximum.mean(0).numpy()
            components['late'] = stable_order(late, ids)
        if cfg['citation_enabled']: components['citation'] = citation_signal(QUERIES[split][query]['text'], ids)
        result[query] = components
    return result


def rank_with_policy(components, split, policy):
    ids = list(CANDIDATES[split]); predictions = {}
    for query, values in components.items():
        mode = policy['neural_mode']; scores = np.zeros(len(ids)); weight = policy['dense_weight']
        # Truncated RRF limits noise from the distant tail; stable candidate-ID ties are deterministic.
        for order, importance in [(values[mode], weight), (values['bm25'], 1 - weight)]:
            count = min(len(order), policy['rrf_depth']); selected = order[:count]
            scores[selected] += importance / (policy['rrf_k'] + np.arange(1, count + 1))
        if policy.get('citation_weight', 0): scores += policy['citation_weight'] * values['citation']
        predictions[query] = top_predictions(scores, ids, query)
    validate_predictions(predictions, split)
    return predictions


def select_policy(components, cfg):
    best = None; trials = []
    modes = ['dense', 'late'] if cfg['search_late_interaction'] else ['dense']
    for mode in modes:
        for weight in cfg['dense_weights']:
            for citation in (cfg['citation_weights'] if cfg['citation_enabled'] else [0.0]):
                policy = {'neural_mode': mode, 'dense_weight': float(weight), 'rrf_k': cfg['rrf_k'], 'rrf_depth': cfg['rrf_depth'], 'citation_weight': float(citation)}
                predictions = rank_with_policy(components, 'dev', policy); metrics = retrieval_metrics(predictions)
                trials.append({'policy': policy, 'micro_f1_at_7': metrics['micro_f1_at_7']})
                # Prefer a larger neural contribution and fewer heuristics when scores tie.
                key = (metrics['micro_f1_at_7'], weight, -citation, mode == 'dense')
                if best is None or key > best[0]: best = (key, policy, predictions, metrics)
    return best[1], best[2], best[3], trials


def dense_predictions(query_embeddings, candidate_embeddings, split):
    ids = candidate_embeddings['ids']; predictions = {}
    for i, query in enumerate(query_embeddings['ids']):
        scores = (query_embeddings['vectors'][i:i + 1] @ candidate_embeddings['vectors'].T)[0].numpy()
        predictions[query] = top_predictions(scores, ids, query)
    validate_predictions(predictions, split)
    return predictions


def evaluate_model(model, cfg, split='dev'):
    queries = encode_pool(model, split, 'queries', cfg); candidates = encode_pool(model, split, 'candidates', cfg)
    components = ranking_components(queries, candidates, split, cfg)
    dense = dense_predictions(queries, candidates, split)
    if split != 'dev': return queries, candidates, components, dense
    policy, predictions, metrics, trials = select_policy(components, cfg)
    return {'queries': queries, 'candidates': candidates, 'components': components,
            'policy': policy, 'predictions': predictions, 'metrics': metrics, 'trials': trials,
            'dense_metrics': retrieval_metrics(dense), 'dense_predictions': dense}


def sample_training_batch(query_keys, cfg, epoch, batch_index, mined=None):
    rng = random.Random(cfg['seed'] + epoch * 1000003 + batch_index * 9743)
    pool = list(CANDIDATES['train']); candidates = []; per_query = {}
    for query in query_keys:
        gold = set(QUERIES['train'][query]['relevant']); positives = sorted(gold)
        selected = rng.sample(positives, min(len(positives), cfg['positives']))
        hard_source = list(dict.fromkeys((mined or {}).get(query, []) + LEXICAL_TOP['train'][query]))
        hard = [key for key in hard_source if key not in gold and (not EXCLUDE_SELF_MATCH or key != query)]
        hard_count = min(len(hard), round(cfg['negatives'] * cfg['hard_fraction']))
        negatives = rng.sample(hard, hard_count)
        available = [key for key in pool if key not in gold and key not in negatives and (not EXCLUDE_SELF_MATCH or key != query)]
        negatives += rng.sample(available, min(len(available), cfg['negatives'] - len(negatives)))
        per_query[query] = set(selected + negatives); candidates.extend(selected + negatives)
    candidates = list(dict.fromkeys(candidates)); positive = []; allowed = []
    for query in query_keys:
        gold = set(QUERIES['train'][query]['relevant'])
        positive.append([key in gold for key in candidates])
        allowed.append([not EXCLUDE_SELF_MATCH or key != query for key in candidates] if cfg['in_batch'] else [key in per_query[query] for key in candidates])
    positive = torch.tensor(positive, dtype=torch.bool, device=DEVICE); allowed = torch.tensor(allowed, dtype=torch.bool, device=DEVICE)
    assert (positive & allowed).any(1).all()
    return candidates, positive, allowed


def retrieval_loss(scores, positive, allowed, cfg):
    logits = (scores / cfg['temperature']).masked_fill(~allowed, -1e4)
    labels = (positive & allowed).float(); labels = labels / labels.sum(1, keepdim=True)
    contrastive = -(labels * F.log_softmax(logits, dim=-1)).sum(1).mean()
    pairwise = []
    for i in range(len(scores)):
        positives = scores[i][positive[i] & allowed[i]]; negatives = scores[i][~positive[i] & allowed[i]]
        if len(negatives): pairwise.append(F.softplus((negatives.max() - positives[:, None] + cfg['margin']) / cfg['temperature']).mean())
    pairwise_loss = torch.stack(pairwise).mean() if pairwise else scores.sum() * 0
    if cfg['loss_name'] == 'multi_positive': return contrastive
    if cfg['loss_name'] == 'pairwise': return pairwise_loss
    return 0.5 * (contrastive + pairwise_loss)


def distillation_loss(scores, teacher_scores, allowed, cfg, teacher_cfg):
    student = (scores / cfg['temperature'] / cfg['kd_temperature']).masked_fill(~allowed, -1e4)
    teacher = (teacher_scores / teacher_cfg['temperature'] / cfg['kd_temperature']).masked_fill(~allowed, -1e4)
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction='batchmean') * cfg['kd_temperature'] ** 2
