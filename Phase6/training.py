def base_config(name, overrides=None):
    cfg = {'implementation_version': 1, 'experiment': name, 'spec': copy.deepcopy(EXPERIMENTS[name]), 'run_tag': RUN_TAG, 'seed': SEED,
           'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'grad_accum': GRAD_ACCUM_STEPS,
           'max_source_tokens': MAX_SOURCE_TOKENS, 'max_target_tokens': MAX_TARGET_TOKENS,
           'source_policy': SOURCE_POLICY, 'encoder_chunk_size': ENCODER_CHUNK_SIZE,
           'instruction': INSTRUCTION, 'summary_prefix': SUMMARY_PREFIX, 'weight_decay': WEIGHT_DECAY,
           'warmup_fraction': WARMUP_FRACTION, 'label_smoothing': LABEL_SMOOTHING,
           'gradient_checkpointing': GRADIENT_CHECKPOINTING, 'patience': EARLY_STOPPING_PATIENCE,
           'rank': RANK, 'alpha': ALPHA, 'bottleneck_dim': BOTTLENECK_DIM, 'prompt_tokens': PROMPT_TOKENS,
           'partial_layers': PARTIAL_LAYERS, 'kd_weight': KD_WEIGHT, 'kd_temperature': KD_TEMPERATURE,
           'teacher_requested': TEACHER_RUN_DIR, 'generation': copy.deepcopy(GENERATION),
           'epoch_validation_limit': EPOCH_VALIDATION_LIMIT, 'final_validation_limit': FINAL_VALIDATION_LIMIT,
           'eval_loss_limit': EVAL_LOSS_LIMIT, 'metric_config': metric_config(),
           'data_fingerprint': DATA_FINGERPRINT, 'tokenizer_hash': TOKENIZER_HASH,
           'phase1_heads': PHASE1_HEADS, 'max_optimizer_steps': MAX_OPTIMIZER_STEPS}
    if LEARNING_RATE is not None: cfg['spec']['lr'] = LEARNING_RATE
    for key, value in (overrides or {}).items():
        if key == 'lr': cfg['spec']['lr'] = value
        elif key in cfg: cfg[key] = value
        else: raise ValueError('Unknown tuning override: ' + key)
    architecture = cfg['spec']['architecture']
    # Precise absolute/rotary limits are verified again against the actual Phase 1 state.
    cfg['context_limit'] = 4096 if architecture == 'encoder_decoder' else 1024
    assert cfg['max_source_tokens'] > 0 and 2 <= cfg['max_target_tokens'] < cfg['context_limit']
    assert cfg['source_policy'] in ['head', 'tail', 'head_tail', 'spread', 'sentence_mmr']
    assert cfg['rank'] > 0 and cfg['prompt_tokens'] > 0 and cfg['batch_size'] > 0 and cfg['grad_accum'] > 0
    assert cfg['epochs'] > 0 and cfg['encoder_chunk_size'] >= 0
    validate_generation(cfg['generation'])
    return cfg


def validate_generation(policy):
    assert policy['num_beams'] >= 1 and 0 <= policy['min_new_tokens'] < policy['max_new_tokens']
    assert policy['repetition_penalty'] >= 1 and policy['no_repeat_ngram_size'] >= 0
    assert policy['long_document_mode'] in ['select', 'map_reduce'] and policy['map_reduce_max_chunks'] > 0


def run_path(name):
    return PROJECT_DIR / (name + '_' + RUN_TAG)


def resolve_teacher(cfg, student_parameters):
    paths = [Path(cfg['teacher_requested'])] if cfg['teacher_requested'] else [p.parent for p in PROJECT_DIR.glob('*/metrics.json')]
    choices = []
    for path in paths:
        if not (path / 'best_model.pt').is_file() or not (path / 'run_config.json').is_file(): continue
        saved = json.loads((path / 'run_config.json').read_text()); metrics = json.loads((path / 'metrics.json').read_text())
        if saved['data_fingerprint'] != DATA_FINGERPRINT or saved['tokenizer_hash'] != TOKENIZER_HASH: continue
        if saved['metric_config'] != cfg['metric_config'] or saved['spec']['method'] == 'kd': continue
        if not metrics['training_complete'] or not metrics['full_validation'] or metrics['total_parameters'] <= student_parameters: continue
        choices.append((metrics['bertscore_f1'], str(path)))
    if not choices: raise FileNotFoundError('KD requires a completed larger Phase 6 teacher with full held-out evaluation. Run encoder_full first.')
    path = Path(max(choices)[1]); teacher_cfg = json.loads((path / 'run_config.json').read_text())
    if teacher_cfg['phase1_heads'] != PHASE1_HEADS: raise ValueError('Teacher Phase 1 head counts differ.')
    state = torch_load(path / 'best_model.pt'); teacher, metadata = make_model(teacher_cfg)
    teacher.load_state_dict(state['model'], strict=True); del state
    teacher = teacher.to(DEVICE).eval().requires_grad_(False)
    stat = (path / 'best_model.pt').stat()
    return teacher, teacher_cfg, {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                  'teacher_parameters': metadata['total_parameters'], 'mode': 'token_kl_on_train_references_only'}


def representation_report(cfg, output):
    prompt_length = len(tok.encode(cfg['instruction'] + cfg['summary_prefix'], add_special_tokens=False).ids)
    soft = cfg['prompt_tokens'] if cfg['spec']['method'] == 'prompt' else 0
    budget = cfg['max_source_tokens'] if cfg['spec']['architecture'] == 'encoder_decoder' else min(
        cfg['max_source_tokens'], cfg['context_limit'] - cfg['generation']['max_new_tokens'] - soft - prompt_length - 2)
    if budget < 1: raise ValueError('Generation budget leaves no source context.')
    report = {'source_policy': cfg['source_policy'], 'source_budget_at_generation': budget, 'splits': {}}
    for split in DATA:
        lengths = np.asarray([TOKENS[split].index[key][1] for key in DATA[split]])
        row = {'documents': len(lengths), 'source_tokens_p50': float(np.median(lengths)),
               'source_tokens_p90': float(np.quantile(lengths, 0.9)),
               'full_source_within_budget_fraction': float((lengths <= budget).mean()),
               'mean_budget_coverage_upper_bound': float(np.minimum(1.0, budget / np.maximum(lengths, 1)).mean())}
        if split != 'test':
            references = np.asarray([REFERENCES[split].index[key][1] + 1 for key in DATA[split]])
            row.update(reference_tokens_p50=float(np.median(references)), reference_tokens_p90=float(np.quantile(references, 0.9)),
                       reference_truncation_fraction=float((references > cfg['max_target_tokens']).mean()))
        report['splits'][split] = row
    save_json(report, output / 'context_coverage.json')
    print('Context/target length audit:', report)
    return report


def target_loss(logits, labels, smoothing=0.0):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, label_smoothing=smoothing)


def kd_loss(student, teacher, labels, temperature):
    per_token = F.kl_div(F.log_softmax(student.float() / temperature, dim=-1),
                          F.softmax(teacher.float() / temperature, dim=-1), reduction='none').sum(-1)
    mask = labels != -100
    return per_token[mask].mean() * temperature ** 2


def evaluate_generation(model, cfg, keys, policy=None):
    model.eval(); policy = policy or cfg['generation']
    rows = generate_rows(model, 'dev', keys, cfg, policy)
    return rows, score_summaries(rows, 'dev', keys)


def run_experiment(name, overrides=None, run_name=None):
    cfg = base_config(name, overrides); run_name = run_name or name; output = run_path(run_name)
    output.mkdir(parents=True, exist_ok=True); config_path = output / 'run_config.json'
    if config_path.is_file() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Settings changed under an existing run. Use a new RUN_TAG: ' + str(output))
    if not RESUME and (output / 'last.pt').exists(): raise ValueError('Choose a fresh run tag instead of overwriting training.')
    required = ['metrics.json', 'verification.json', 'prediction.json', 'submission.zip', 'best_model.pt', 'last.pt']
    if RESUME and all((output / filename).is_file() for filename in required):
        cached = json.loads((output / 'metrics.json').read_text()); verified = json.loads((output / 'verification.json').read_text())
        checks = ['checkpoint_reload_text_match', 'checkpoint_reload_scores_match', 'json_metric_roundtrip',
                  'one_summary_per_test_document', 'original_ids_preserved', 'zip_verified', 'training_complete']
        if cached.get('training_complete') and all(verified.get(check) is True for check in checks):
            rows = json.loads((output / 'prediction.json').read_text()); validate_rows(rows, 'test', list(DATA['test']))
            with zipfile.ZipFile(output / 'submission.zip') as archive:
                assert archive.namelist() == ['prediction.json'] and json.loads(archive.read('prediction.json')) == rows
            print('Reused verified completed run:', output, '(no retraining or repeated decoding)')
            return cached
    save_json(cfg, config_path); seed_all(cfg['seed'])
    representation_report(cfg, output)
    model, metadata = make_model(cfg); teacher = None; teacher_cfg = None; teacher_info = None
    if cfg['spec']['method'] == 'kd':
        pinned = cfg
        if (output / 'last.pt').is_file() and (output / 'model_metadata.json').is_file():
            previous_teacher = json.loads((output / 'model_metadata.json').read_text()).get('teacher')
            if previous_teacher: pinned = {**cfg, 'teacher_requested': previous_teacher['path']}
        rng = capture_rng()
        try: teacher, teacher_cfg, teacher_info = resolve_teacher(pinned, metadata['total_parameters'])
        finally: restore_rng(rng)
    metadata['teacher'] = teacher_info; save_json(metadata, output / 'model_metadata.json')
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['spec']['lr'], weight_decay=cfg['weight_decay'])
    keys = list(DATA['train']); batches = math.ceil(len(keys) / cfg['batch_size'])
    steps_per_epoch = math.ceil(batches / cfg['grad_accum']); total_steps = steps_per_epoch * cfg['epochs']
    warmup = max(1, round(total_steps * cfg['warmup_fraction']))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda' and AMP_DTYPE == torch.float16)
    start_epoch = 0; start_batch = 0; step = 0; best = -float('inf'); bad = 0; complete = False; history = []; elapsed = 0.0
    if RESUME and (output / 'last.pt').is_file():
        saved = torch_load(output / 'last.pt')
        if saved['config'] != cfg or saved['teacher'] != teacher_info: raise ValueError('Saved config/teacher signature differs.')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
        start_epoch = saved['epoch']; start_batch = saved['next_batch']; step = saved['step']; best = saved['best']; bad = saved['bad']
        history = saved['history']; elapsed = saved['seconds']; complete = saved['complete']; del saved
    start = time.time()
    if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    epoch_keys = fixed_validation_keys(cfg['epoch_validation_limit'])
    save_json({'keys': epoch_keys, 'scope': 'fixed held-out subset for checkpoint selection'}, output / 'selection_manifest.json')
    def persist(epoch, next_batch):
        save_torch({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'rng': capture_rng(), 'config': cfg, 'teacher': teacher_info,
                    'epoch': epoch, 'next_batch': next_batch, 'step': step, 'best': best, 'bad': bad,
                    'history': history, 'seconds': elapsed + time.time() - start, 'complete': complete}, output / 'last.pt')
        save_json(history, output / 'history.json')
    if not complete:
        for epoch in range(start_epoch, cfg['epochs']):
            if cfg['max_optimizer_steps'] is not None and step >= cfg['max_optimizer_steps']: break
            order = list(keys); random.Random(cfg['seed'] + epoch).shuffle(order)
            begin = start_batch if epoch == start_epoch else 0; next_batch = begin; capped = False
            model.train(); optimizer.zero_grad(set_to_none=True); accumulated = 0.0
            progress = tqdm(range(begin, batches), desc=f'{run_name} epoch {epoch + 1}/{cfg["epochs"]}', unit='batch')
            for i in progress:
                current = order[i * cfg['batch_size']:(i + 1) * cfg['batch_size']]
                batch = pack_batch('train', current, cfg)
                # Normalize by all supervised tokens in this accumulation group, including the short last group.
                group_start = (i // cfg['grad_accum']) * cfg['grad_accum']
                group_keys = order[group_start * cfg['batch_size']:min(batches, group_start + cfg['grad_accum']) * cfg['batch_size']]
                total_tokens = sum(min(len(REFERENCES['train'][key]) + 1, cfg['max_target_tokens']) for key in group_keys)
                token_fraction = int((batch['labels'] != -100).sum()) / total_tokens
                with amp_context():
                    logits = model(batch); loss = target_loss(logits, batch['labels'], cfg['label_smoothing'])
                    if teacher is not None:
                        teacher_batch = pack_batch('train', current, teacher_cfg, batch['targets_list'])
                        with torch.no_grad(): teacher_logits = teacher(teacher_batch)
                        loss = (1 - cfg['kd_weight']) * loss + cfg['kd_weight'] * kd_loss(logits, teacher_logits, batch['labels'], cfg['kd_temperature'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite summarization loss.')
                scaler.scale(loss * token_fraction).backward(); accumulated += loss.item() * token_fraction; next_batch = i + 1
                progress.set_postfix(loss=f'{loss.item():.4f}', step=step)
                if next_batch % cfg['grad_accum'] == 0 or next_batch == batches:
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    previous = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= previous:
                        scheduler.step(); step += 1
                        history.append({'kind': 'train', 'epoch': epoch + 1, 'step': step, 'loss': accumulated})
                        if step % SAVE_EVERY_STEPS == 0: persist(epoch, next_batch)
                    accumulated = 0.0
                    if cfg['max_optimizer_steps'] is not None and step >= cfg['max_optimizer_steps']: capped = True; break
            progress.close()
            rows, metrics = evaluate_generation(model, cfg, epoch_keys)
            metrics['validation_loss'] = validation_loss(model, cfg)
            history.append({'kind': 'dev', 'epoch': epoch + 1, 'step': step, **metrics})
            show_scores(metrics, 'Fixed validation subset — epoch ' + str(epoch + 1))
            if metrics['bertscore_f1'] > best + 1e-12:
                best = metrics['bertscore_f1']; bad = 0
                save_torch({'model': model.state_dict(), 'config': cfg, 'metadata': metadata, 'epoch': epoch + 1}, output / 'best_model.pt')
                save_json(metrics, output / 'best_selection_metrics.json'); save_json(rows, output / 'best_selection_prediction.json')
            else: bad += 1
            finished = next_batch == batches
            complete = finished and (epoch + 1 == cfg['epochs'] or (cfg['patience'] > 0 and bad >= cfg['patience']))
            persist(epoch + 1 if finished else epoch, 0 if finished else next_batch)
            if complete or capped: break
            model.train()
    if teacher is not None: del teacher; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    if not (output / 'best_model.pt').is_file(): raise RuntimeError('No evaluated checkpoint exists.')
    state = torch_load(output / 'best_model.pt'); model.load_state_dict(state['model'], strict=True); best_epoch = state['epoch']; del state
    verify_rows, verified = evaluate_generation(model, cfg, epoch_keys)
    assert verify_rows == json.loads((output / 'best_selection_prediction.json').read_text()), 'Reloaded summaries differ from the selected checkpoint.'
    recorded = json.loads((output / 'best_selection_metrics.json').read_text())
    for metric in ['bertscore_f1', 'rouge1_f1', 'rougeL_f1']:
        assert math.isclose(verified[metric], recorded[metric], rel_tol=0, abs_tol=1e-6), 'Reloaded metrics differ.'
    final_keys = fixed_validation_keys(cfg['final_validation_limit'])
    if final_keys == epoch_keys: final_rows, metrics = verify_rows, verified
    else: final_rows, metrics = evaluate_generation(model, cfg, final_keys)
    save_json(final_rows, output / 'validation_prediction.json')
    saved_rows = json.loads((output / 'validation_prediction.json').read_text())
    roundtrip = score_summaries(saved_rows, 'dev', final_keys)
    for metric in ['bertscore_f1', 'rouge1_f1', 'rougeL_f1']:
        assert math.isclose(roundtrip[metric], metrics[metric], abs_tol=1e-6)
    metrics.update({'experiment': name, 'run_name': run_name, 'architecture': cfg['spec']['architecture'], 'method': cfg['spec']['method'],
                    'best_epoch': best_epoch, 'training_complete': complete, 'full_validation': len(final_keys) == len(DATA['dev']),
                    'optimizer_steps': step, 'training_seconds': elapsed + time.time() - start,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else 0, **metadata})
    save_json(metrics, output / 'metrics.json'); save_json(cfg['generation'], output / 'generation_config.json')
    show_scores(metrics, 'Best checkpoint — final held-out evaluation')
    test_rows = generate_rows(model, 'test', list(DATA['test']), cfg, cfg['generation']); save_submission(output, test_rows)
    save_json({'checkpoint_reload_text_match': True, 'checkpoint_reload_scores_match': True, 'json_metric_roundtrip': True,
               'one_summary_per_test_document': True, 'original_ids_preserved': True, 'zip_verified': True,
               'training_complete': complete, 'full_validation': metrics['full_validation'], 'private_test_score': None}, output / 'verification.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); tr = [r for r in history if r['kind'] == 'train']; dv = [r for r in history if r['kind'] == 'dev']
    axes[0].plot([r['step'] for r in tr], [r['loss'] for r in tr]); axes[0].set(xlabel='Optimizer step', ylabel='Target loss')
    axes[1].plot([r['epoch'] for r in dv], [r['bertscore_f1'] for r in dv]); axes[1].set(xlabel='Epoch', ylabel='Validation BERTScore F1')
    fig.tight_layout(); fig.savefig(output / 'training_curves.png', dpi=130); plt.show(); plt.close(fig)
    print('Verification PASS. Saved:', output / 'submission.zip')
    del model, optimizer; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return metrics


def search_decoding(name):
    source = run_path(name); cfg = json.loads((source / 'run_config.json').read_text())
    model, _ = make_model(cfg); state = torch_load(source / 'best_model.pt'); model.load_state_dict(state['model'], strict=True); del state
    model = model.to(DEVICE).eval(); keys = fixed_validation_keys(cfg['epoch_validation_limit'])
    _, baseline = evaluate_generation(model, cfg, keys); best_score = baseline['bertscore_f1']; best_policy = cfg['generation']; trials = []
    for trial in tqdm(DECODING_CANDIDATES, desc='Validation decoding search', unit='policy'):
        policy = {**cfg['generation'], **trial['overrides']}; validate_generation(policy)
        _, metrics = evaluate_generation(model, cfg, keys, policy); trials.append({'name': trial['name'], 'policy': policy, 'metrics': metrics})
        if metrics['bertscore_f1'] > best_score: best_score = metrics['bertscore_f1']; best_policy = policy
    output = run_path(name + '__decoding'); output.mkdir(exist_ok=True)
    save_json({'source': str(source), 'selection_scope': keys, 'selected_policy': best_policy, 'trials': trials}, output / 'decoding_search.json')
    final_keys = fixed_validation_keys(cfg['final_validation_limit'])
    rows, metrics = evaluate_generation(model, cfg, final_keys, best_policy); save_json(rows, output / 'validation_prediction.json'); save_json(metrics, output / 'metrics.json')
    save_submission(output, generate_rows(model, 'test', list(DATA['test']), cfg, best_policy))
    del model; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return metrics
