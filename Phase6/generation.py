def clean_summary(text, deduplicate=False):
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    if deduplicate:
        seen = set(); kept = []
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            key = re.sub(r'\W+', '', sentence.lower())
            if key and key not in seen: kept.append(sentence); seen.add(key)
        text = ' '.join(kept)
    return text


def block_repeats(logits, sequences, policy):
    for row, sequence in enumerate(sequences):
        penalty = policy['repetition_penalty']
        if sequence and penalty != 1:
            indices = torch.tensor(sorted(set(sequence)), device=logits.device)
            values = logits[row, indices]
            logits[row, indices] = torch.where(values < 0, values * penalty, values / penalty)
        n = policy['no_repeat_ngram_size']
        if n > 0 and len(sequence) >= n - 1:
            prefix = tuple(sequence[-(n - 1):]) if n > 1 else ()
            banned = {sequence[j + n - 1] for j in range(len(sequence) - n + 1)
                      if tuple(sequence[j:j + n - 1]) == prefix}
            if banned: logits[row, list(banned)] = float('-inf')
    return logits


@torch.no_grad()
def generate_tokens(model, source_ids, cfg, policy):
    architecture = cfg['spec']['architecture']; model.eval()
    soft = len(model.soft_prompt) if model.soft_prompt is not None else 0
    instruction = tok.encode(cfg['instruction'], add_special_tokens=False).ids
    separator = tok.encode(cfg['summary_prefix'], add_special_tokens=False).ids
    budget = cfg['max_source_tokens'] if architecture == 'encoder_decoder' else min(
        cfg['max_source_tokens'], model.max_context - policy['max_new_tokens'] - soft - len(instruction) - len(separator) - 1)
    if budget < 1: raise ValueError('Generation length leaves no source context.')
    # Rebudget an already selected context by retaining its beginning/end.
    if len(source_ids) > budget: source_ids, _ = select_source('', source_ids, budget, 'head_tail')
    source_ids = source_ids or [UNK_ID]
    encoder = None
    if architecture == 'encoder_decoder':
        src = torch.tensor([[BOS_ID] + source_ids + [EOS_ID]], device=DEVICE)
        with amp_context(): encoder = model.encode(src, torch.ones_like(src, dtype=torch.bool))
        prompt = [BOS_ID]
    else:
        prompt = [BOS_ID] + instruction + source_ids + separator
    ids = torch.tensor([prompt], device=DEVICE)
    with amp_context():
        hidden, cache = model.hidden(ids, torch.ones_like(ids, dtype=torch.bool),
                                     torch.tensor([len(prompt)], device=DEVICE), encoder=encoder, use_cache=True)
        next_logits = model.backbone.head(hidden[:, -1]).float()
    sequences = [[]]; scores = torch.zeros(1, device=DEVICE); finished = []; beams = policy['num_beams']
    for step in range(policy['max_new_tokens']):
        logits = block_repeats(next_logits.clone(), sequences, policy)
        blocked = [PAD_ID, BOS_ID, UNK_ID]
        sep_id = tok.token_to_id('<sep>')
        if sep_id is not None: blocked.append(sep_id)
        logits[:, blocked] = float('-inf')
        if step < policy['min_new_tokens']: logits[:, EOS_ID] = float('-inf')
        if not torch.isfinite(logits).any(dim=1).all(): raise FloatingPointError('No finite decoding candidates.')
        log_probs = torch.log_softmax(logits, dim=-1)
        candidate = scores[:, None] + log_probs
        values, indices = candidate.flatten().topk(min(candidate.numel(), beams * 2))
        alive = []; origins = []; tokens = []; next_scores = []
        for value, index in zip(values.tolist(), indices.tolist()):
            origin, token = divmod(index, logits.shape[-1]); sequence = sequences[origin] + [token]
            if token == EOS_ID:
                finished.append((value / max(1, len(sequence)) ** policy['length_penalty'], sequence))
            elif len(alive) < beams:
                alive.append(sequence); origins.append(origin); tokens.append(token); next_scores.append(value)
        # For greedy decoding EOS terminates immediately, rather than taking the second-best token.
        if beams == 1 and int(indices[0] % logits.shape[-1]) == EOS_ID: break
        if not alive: break
        sequences = alive; scores = torch.tensor(next_scores, device=DEVICE)
        order = torch.tensor(origins, device=DEVICE)
        cache = [tuple(value.index_select(0, order) for value in layer) for layer in cache]
        if encoder is not None: encoder = (encoder[0].index_select(0, order), encoder[1].index_select(0, order))
        if step + 1 == policy['max_new_tokens']: break
        ids = torch.tensor(tokens, device=DEVICE)[:, None]
        keep = torch.ones(len(tokens), cache[0][0].shape[2] + 1, dtype=torch.bool, device=DEVICE)
        with amp_context():
            hidden, cache = model.hidden(ids, keep, past=cache, encoder=encoder, use_cache=True)
            next_logits = model.backbone.head(hidden[:, -1]).float()
    for score, sequence in zip(scores.tolist(), sequences):
        if sequence: finished.append((score / len(sequence) ** policy['length_penalty'], sequence))
    if not finished: raise RuntimeError('Decoding produced no candidates.')
    best = max(finished, key=lambda item: item[0])[1]
    return [token for token in best if token not in (EOS_ID, PAD_ID, BOS_ID)]


@torch.no_grad()
def generate_summary(model, split, key, cfg, policy):
    ids = TOKENS[split][key]
    selected, _ = select_source(DATA[split][key]['text'], ids, cfg['max_source_tokens'], cfg['source_policy'])
    if policy['long_document_mode'] == 'map_reduce' and len(ids) > cfg['max_source_tokens']:
        ranges = [(i, min(i + cfg['max_source_tokens'], len(ids))) for i in range(0, len(ids), cfg['max_source_tokens'])]
        if len(ranges) > policy['map_reduce_max_chunks']:
            chosen = np.linspace(0, len(ranges) - 1, policy['map_reduce_max_chunks']).round().astype(int)
            ranges = [ranges[i] for i in chosen]
        intermediate = []
        for start, stop in ranges: intermediate.extend(generate_tokens(model, ids[start:stop], cfg, policy))
        selected, _ = select_source(tok.decode(intermediate), intermediate, cfg['max_source_tokens'], cfg['source_policy'])
    generated = generate_tokens(model, selected, cfg, policy)
    summary = clean_summary(tok.decode(generated), policy['deduplicate_sentences'])
    if not summary:
        raise RuntimeError(f'Empty generated summary for {key}; inspect model/decoding before submission.')
    return summary


def generate_rows(model, split, keys, cfg, policy):
    return [{'id': DATA[split][key]['id'], 'summary': generate_summary(model, split, key, cfg, policy)}
            for key in tqdm(keys, desc='Generating ' + split + ' (KV cache)', unit='document')]


def validate_rows(rows, split, keys):
    if not isinstance(rows, list) or len(rows) != len(keys): raise ValueError('Expected one summary per document.')
    for row, key in zip(rows, keys):
        original = DATA[split][key]['id']
        if set(row) != {'id', 'summary'} or type(row['id']) is not type(original) or row['id'] != original:
            raise ValueError('Summary fields/order/original IDs disagree.')
        if not isinstance(row['summary'], str) or not row['summary'].strip():
            raise ValueError('Every generated summary must be nonempty text.')
    if split == 'test' and len({original_key(row['id']) for row in rows}) != len(rows):
        raise ValueError('Duplicate test ID.')
    return True


def save_submission(output, rows):
    keys = list(DATA['test']); validate_rows(rows, 'test', keys)
    save_json(rows, output / 'prediction.json')
    temporary = output / 'submission.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / 'prediction.json', arcname='prediction.json')
    os.replace(temporary, output / 'submission.zip')
    with zipfile.ZipFile(output / 'submission.zip') as archive:
        assert archive.namelist() == ['prediction.json']
        saved = json.loads(archive.read('prediction.json')); validate_rows(saved, 'test', keys)
        assert saved == rows
