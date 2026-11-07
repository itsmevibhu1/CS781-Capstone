def base_config(name, overrides=None):
    cfg = {'implementation_version': 1, 'experiment': name, 'spec': copy.deepcopy(EXPERIMENTS[name]), 'run_tag': RUN_TAG, 'seed': SEED,
        'epochs': EPOCHS, 'query_batch_size': QUERY_BATCH_SIZE, 'grad_accum': GRAD_ACCUM_STEPS,
        'positives': POSITIVES_PER_QUERY, 'negatives': NEGATIVES_PER_QUERY, 'hard_fraction': HARD_NEGATIVE_FRACTION,
        'in_batch': USE_IN_BATCH_NEGATIVES, 'mine_dense_negatives': MINE_DENSE_NEGATIVES,
        'chunk_length': CHUNK_LENGTH, 'chunk_overlap': CHUNK_OVERLAP, 'max_chunks': MAX_CHUNKS,
        'chunk_batch_size': CHUNK_BATCH_SIZE, 'encode_batch_size': ENCODE_DOCUMENT_BATCH_SIZE,
        'chunk_policy': CHUNK_POLICY, 'pooling': POOLING, 'embedding_dim': EMBEDDING_DIM,
        'dropout': HEAD_DROPOUT, 'share_projection': SHARE_QUERY_CANDIDATE_PROJECTION,
        'loss_name': LOSS_NAME, 'temperature': TEMPERATURE, 'margin': PAIRWISE_MARGIN,
        'head_lr_multiplier': HEAD_LR_MULTIPLIER, 'weight_decay': WEIGHT_DECAY, 'warmup_fraction': WARMUP_FRACTION,
        'patience': EARLY_STOPPING_PATIENCE, 'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'rank': RANK, 'alpha': ALPHA, 'prompt_tokens': PROMPT_TOKENS, 'partial_layers': PARTIAL_LAYERS,
        'kd_weight': KD_WEIGHT, 'kd_temperature': KD_TEMPERATURE, 'teacher_requested': TEACHER_RUN_DIR,
        'rrf_k': RRF_K, 'rrf_depth': RRF_DEPTH, 'dense_weights': DENSE_WEIGHTS,
        'search_late_interaction': SEARCH_LATE_INTERACTION, 'citation_enabled': ENABLE_CITATION_FEATURE, 'citation_weights': CITATION_WEIGHTS,
        'bm25': {'k1': BM25_K1, 'b': BM25_B, 'max_features': BM25_MAX_FEATURES, 'stop_words': BM25_STOP_WORDS, 'hard_depth': HARD_NEGATIVE_DEPTH},
        'exclude_self_match': EXCLUDE_SELF_MATCH, 'max_optimizer_steps': MAX_OPTIMIZER_STEPS,
        'phase1_heads': PHASE1_HEADS, 'data_fingerprint': DATA_FINGERPRINT, 'tokenizer_hash': TOKENIZER_HASH}
    if LEARNING_RATE is not None: cfg['spec']['lr'] = LEARNING_RATE
    for key, value in (overrides or {}).items():
        if key == 'lr': cfg['spec']['lr'] = value
        elif key in cfg: cfg[key] = value
        else: raise ValueError('Unknown tuning override: ' + key)
    if cfg['bm25'] != LEXICAL_SIGNATURE: raise ValueError('BM25 settings changed; rerun Section 5 before training.')
    if cfg['exclude_self_match'] != DATA_SELF_EXCLUSION: raise ValueError('Self-exclusion changed; rerun Sections 4–5.')
    assert cfg['loss_name'] in ('multi_positive', 'pairwise', 'combined')
    assert cfg['spec']['method'] in ('full', 'partial', 'lorma', 'prompt', 'ia3', 'kd')
    assert 4 <= cfg['chunk_length'] <= 1024 and 0 <= cfg['chunk_overlap'] < cfg['chunk_length'] - 2
    assert cfg['max_chunks'] >= 0 and cfg['chunk_batch_size'] > 0 and cfg['encode_batch_size'] > 0
    assert cfg['positives'] > 0 and cfg['negatives'] > 0 and 0 <= cfg['hard_fraction'] <= 1
    assert cfg['chunk_policy'] in ('spread', 'head', 'tail', 'head_tail') and cfg['pooling'] in ('attention', 'mean', 'max', 'mean_max')
    assert cfg['temperature'] > 0 and cfg['embedding_dim'] > 0 and 0 <= cfg['kd_weight'] <= 1 and cfg['kd_temperature'] > 0
    assert cfg['rrf_k'] > 0 and cfg['rrf_depth'] >= TOP_K + int(EXCLUDE_SELF_MATCH)
    assert cfg['dense_weights'] and all(0 < weight <= 1 for weight in cfg['dense_weights'])
    return cfg


def run_path(name): return PROJECT_DIR / (name + '_' + RUN_TAG)


def resolve_teacher(cfg, student_parameters):
    paths = [Path(cfg['teacher_requested'])] if cfg['teacher_requested'] else [path.parent for path in PROJECT_DIR.glob('*/metrics.json')]
    choices = []
    for path in paths:
        if not (path / 'run_config.json').is_file() or not (path / 'best_model.pt').is_file(): continue
        saved = json.loads((path / 'run_config.json').read_text()); metric = json.loads((path / 'metrics.json').read_text())
        if saved['data_fingerprint'] != DATA_FINGERPRINT or saved['tokenizer_hash'] != TOKENIZER_HASH or saved['phase1_heads'] != PHASE1_HEADS: continue
        if not metric['training_complete'] or metric['total_parameters'] <= student_parameters or saved['spec']['method'] == 'kd': continue
        choices.append((metric['dense_micro_f1_at_7'], str(path)))
    if not choices: raise FileNotFoundError('KD requires a completed compatible larger Phase 7 retrieval teacher. Run encoder_full/decoder_partial first.')
    path = Path(max(choices)[1]); teacher_cfg = json.loads((path / 'run_config.json').read_text())
    model, metadata = make_model(teacher_cfg); state = torch_load(path / 'best_model.pt'); model.load_state_dict(state['model'], strict=True); del state
    stat = (path / 'best_model.pt').stat()
    info = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'teacher_parameters': metadata['total_parameters'], 'selection_metric': 'dense_micro_f1_at_7'}
    return model.to(DEVICE).eval().requires_grad_(False), teacher_cfg, info


def optimizer_groups(model, cfg):
    backbone = []; head = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: (backbone if name.startswith('features.backbone.') else head).append(parameter)
    groups = []
    if backbone: groups.append({'params': backbone, 'lr': cfg['spec']['lr']})
    if head: groups.append({'params': head, 'lr': cfg['spec']['lr'] * cfg['head_lr_multiplier']})
    return groups


def run_experiment(name, overrides=None, run_name=None):
    cfg = base_config(name, overrides); run_name = run_name or name; output = run_path(run_name); output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    if config_path.is_file() and json.loads(config_path.read_text()) != cfg: raise ValueError('Settings changed under existing run; choose a fresh RUN_TAG: ' + str(output))
    if not RESUME and (output / 'last.pt').exists(): raise ValueError('Use a fresh tag instead of overwriting training.')
    if RESUME and all((output / filename).is_file() for filename in ['metrics.json', 'verification.json', 'dev_prediction.json', 'prediction.json', 'submission.zip', 'best_model.pt', 'last.pt']):
        cached = json.loads((output / 'metrics.json').read_text()); verified = json.loads((output / 'verification.json').read_text())
        if cached['training_complete'] and verified.get('checkpoint_reload_ranks_match') and verified.get('json_metrics_match'):
            predictions = json.loads((output / 'prediction.json').read_text()); validate_predictions(predictions, 'test')
            with zipfile.ZipFile(output / 'submission.zip') as archive:
                assert archive.namelist() == ['prediction.json'] and json.loads(archive.read('prediction.json')) == predictions
            measured = retrieval_metrics(json.loads((output / 'dev_prediction.json').read_text()))
            assert measured['micro_f1_at_7'] == cached['micro_f1_at_7']
            print('Reused completed verified run:', output); return cached
    save_json(cfg, config_path); seed_all(cfg['seed']); model, metadata = make_model(cfg)
    teacher = None; teacher_cfg = None; teacher_info = None
    if cfg['spec']['method'] == 'kd':
        pinned = cfg
        if (output / 'last.pt').is_file() and (output / 'model_metadata.json').is_file():
            old_teacher = json.loads((output / 'model_metadata.json').read_text()).get('teacher')
            if old_teacher: pinned = {**cfg, 'teacher_requested': old_teacher['path']}
        rng = capture_rng()
        try: teacher, teacher_cfg, teacher_info = resolve_teacher(pinned, metadata['total_parameters'])
        finally: restore_rng(rng)
    metadata['teacher'] = teacher_info; save_json(metadata, output / 'model_metadata.json'); model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg), weight_decay=cfg['weight_decay'])
    batches = math.ceil(len(TRAIN_KEYS) / cfg['query_batch_size']); total_steps = cfg['epochs'] * math.ceil(batches / cfg['grad_accum'])
    warmup = max(1, round(total_steps * cfg['warmup_fraction']))
    def schedule(step): return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda' and AMP_DTYPE == torch.float16)
    epoch0 = 0; batch0 = 0; step = 0; best = (-1., -1.); bad = 0; history = []; complete = False; elapsed = 0.0; mined = {}
    if RESUME and (output / 'last.pt').is_file():
        state = torch_load(output / 'last.pt')
        if state['config'] != cfg or state['teacher'] != teacher_info: raise ValueError('Resume config or teacher signature changed.')
        model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
        epoch0 = state['epoch']; batch0 = state['next_batch']; step = state['step']; best = tuple(state['best']); bad = state['bad']
        history = state['history']; complete = state['complete']; elapsed = state['seconds']; mined = state['mined']; del state
    start = time.time()
    if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    def persist(epoch, next_batch):
        save_torch({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'rng': capture_rng(), 'config': cfg, 'teacher': teacher_info, 'epoch': epoch, 'next_batch': next_batch, 'step': step,
                    'best': best, 'bad': bad, 'history': history, 'complete': complete, 'mined': mined, 'seconds': elapsed + time.time() - start}, output / 'last.pt')
        save_json(history, output / 'history.json')
    if not complete:
        for epoch in range(epoch0, cfg['epochs']):
            if cfg['max_optimizer_steps'] is not None and step >= cfg['max_optimizer_steps']: break
            order = list(TRAIN_KEYS); random.Random(cfg['seed'] + epoch).shuffle(order)
            begin = batch0 if epoch == epoch0 else 0; next_batch = begin; capped = False; accumulated = 0.0
            model.train(); optimizer.zero_grad(set_to_none=True)
            progress = tqdm(range(begin, batches), desc=f'{run_name} epoch {epoch + 1}/{cfg["epochs"]}', unit='query batch')
            for i in progress:
                query_keys = order[i * cfg['query_batch_size']:(i + 1) * cfg['query_batch_size']]
                candidates, positive, allowed = sample_training_batch(query_keys, cfg, epoch, i, mined)
                query_batch = make_chunks('train', 'queries', query_keys, cfg); candidate_batch = make_chunks('train', 'candidates', candidates, cfg)
                group_start = (i // cfg['grad_accum']) * cfg['grad_accum']
                group_queries = min(cfg['grad_accum'] * cfg['query_batch_size'], len(order) - group_start * cfg['query_batch_size'])
                fraction = len(query_keys) / group_queries
                with amp_context():
                    q = model(query_batch, 'queries'); c = model(candidate_batch, 'candidates'); scores = q @ c.T
                    loss = retrieval_loss(scores.float(), positive, allowed, cfg)
                    if teacher is not None:
                        with torch.no_grad():
                            tq = teacher(make_chunks('train', 'queries', query_keys, teacher_cfg), 'queries')
                            tc = teacher(make_chunks('train', 'candidates', candidates, teacher_cfg), 'candidates')
                        loss = (1 - cfg['kd_weight']) * loss + cfg['kd_weight'] * distillation_loss(scores.float(), (tq @ tc.T).float(), allowed, cfg, teacher_cfg)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite retrieval loss.')
                scaler.scale(loss * fraction).backward(); accumulated += loss.item() * fraction; next_batch = i + 1
                progress.set_postfix(loss=f'{loss.item():.4f}', step=step)
                if next_batch % cfg['grad_accum'] == 0 or next_batch == batches:
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    previous = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= previous:
                        scheduler.step(); step += 1; history.append({'kind': 'train', 'epoch': epoch + 1, 'step': step, 'loss': accumulated})
                        if step % SAVE_EVERY_STEPS == 0: persist(epoch, next_batch)
                    accumulated = 0.0
                    if cfg['max_optimizer_steps'] is not None and step >= cfg['max_optimizer_steps']: capped = True; break
            progress.close(); evaluated = evaluate_model(model, cfg)
            metrics = evaluated['metrics']; dense_metric = evaluated['dense_metrics']['micro_f1_at_7']; key = (metrics['micro_f1_at_7'], dense_metric)
            history.append({'kind': 'dev', 'epoch': epoch + 1, 'step': step, 'micro_f1_at_7': metrics['micro_f1_at_7'], 'dense_micro_f1_at_7': dense_metric})
            show_scores(metrics, 'Full DEV — dev-selected neural/BM25 policy'); print('Dense-only DEV Micro-F1@7:', dense_metric)
            if key > best:
                best = key; bad = 0
                save_torch({'model': model.state_dict(), 'config': cfg, 'metadata': metadata, 'epoch': epoch + 1}, output / 'best_model.pt')
                save_json(evaluated['policy'], output / 'retrieval_policy.json'); save_json(metrics, output / 'best_dev_metrics.json')
                save_json(evaluated['predictions'], output / 'best_dev_prediction.json')
            else: bad += 1
            finished = next_batch == batches
            complete = finished and (epoch + 1 == cfg['epochs'] or (cfg['patience'] > 0 and bad >= cfg['patience']))
            if cfg['mine_dense_negatives'] and finished and not complete and not capped:
                tq = encode_pool(model, 'train', 'queries', cfg); tc = encode_pool(model, 'train', 'candidates', cfg)
                mined = {}
                for j, query in enumerate(tq['ids']):
                    order_dense = stable_order((tq['vectors'][j:j + 1] @ tc['vectors'].T)[0].numpy(), tc['ids'])
                    mined[query] = [tc['ids'][k] for k in order_dense if tc['ids'][k] not in set(QUERIES['train'][query]['relevant'])][:HARD_NEGATIVE_DEPTH]
            persist(epoch + 1 if finished else epoch, 0 if finished else next_batch)
            del evaluated
            if complete or capped: break
    if teacher is not None: del teacher; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    if not (output / 'best_model.pt').is_file(): raise RuntimeError('No evaluated checkpoint exists.')
    state = torch_load(output / 'best_model.pt'); model.load_state_dict(state['model'], strict=True); best_epoch = state['epoch']; del state
    final = evaluate_model(model, cfg)
    saved_policy = json.loads((output / 'retrieval_policy.json').read_text())
    verified_predictions = rank_with_policy(final['components'], 'dev', saved_policy)
    assert verified_predictions == json.loads((output / 'best_dev_prediction.json').read_text()), 'Reloaded rankings differ.'
    measured = retrieval_metrics(verified_predictions)
    assert measured == json.loads((output / 'best_dev_metrics.json').read_text()), 'Reloaded metrics differ.'
    save_json(verified_predictions, output / 'dev_prediction.json')
    assert retrieval_metrics(json.loads((output / 'dev_prediction.json').read_text())) == measured
    save_json(final['dense_metrics'], output / 'dense_dev_metrics.json'); save_json(final['trials'], output / 'ranking_trials.json')
    save_json(measured['per_query'], output / 'dev_errors.json')
    save_torch({'queries': final['queries'], 'candidates': final['candidates'], 'data_fingerprint': DATA_FINGERPRINT}, output / 'dev_embeddings.pt')
    coverage = {'dev_queries': final['queries']['mean_token_coverage'], 'dev_candidates': final['candidates']['mean_token_coverage']}
    metrics = {**measured, 'dense_micro_f1_at_7': final['dense_metrics']['micro_f1_at_7'], 'bm25_micro_f1_at_7': bm25_metrics['micro_f1_at_7'],
               'experiment': name, 'run_name': run_name, 'architecture': cfg['spec']['architecture'], 'method': cfg['spec']['method'],
               'best_epoch': best_epoch, 'training_complete': complete, 'optimizer_steps': step,
               'training_seconds': elapsed + time.time() - start, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else 0, **metadata}
    save_json(metrics, output / 'metrics.json'); show_scores(metrics, 'Best checkpoint — independently verified DEV')
    del final
    queries, candidates, components, _ = evaluate_model(model, cfg, 'test')
    predictions = rank_with_policy(components, 'test', saved_policy); save_submission(output, predictions)
    save_torch({'queries': queries, 'candidates': candidates, 'data_fingerprint': DATA_FINGERPRINT}, output / 'test_embeddings.pt')
    coverage.update(test_queries=queries['mean_token_coverage'], test_candidates=candidates['mean_token_coverage'])
    save_json(coverage, output / 'chunk_coverage.json')
    save_json({'checkpoint_reload_ranks_match': True, 'checkpoint_reload_metrics_match': True, 'json_metrics_match': True,
               'split_specific_candidate_pool': True, 'seven_unique_original_candidate_ids': True, 'zip_verified': True,
               'training_complete': complete, 'private_test_score': None}, output / 'verification.json')
    train_rows = [row for row in history if row['kind'] == 'train']; dev_rows = [row for row in history if row['kind'] == 'dev']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); axes[0].plot([r['step'] for r in train_rows], [r['loss'] for r in train_rows])
    axes[0].set(xlabel='Optimizer step', ylabel='Retrieval loss'); axes[1].plot([r['epoch'] for r in dev_rows], [r['micro_f1_at_7'] for r in dev_rows], label='Selected hybrid')
    axes[1].plot([r['epoch'] for r in dev_rows], [r['dense_micro_f1_at_7'] for r in dev_rows], label='Dense only'); axes[1].legend()
    axes[1].set(xlabel='Epoch', ylabel='DEV Micro-F1@7'); fig.tight_layout(); fig.savefig(output / 'training_curves.png', dpi=130); plt.show(); plt.close(fig)
    print('Verification PASS. Saved:', output / 'submission.zip')
    del model, optimizer, queries, candidates, components; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return metrics
