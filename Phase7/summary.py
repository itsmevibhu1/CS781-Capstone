import csv
records = []
for file in PROJECT_DIR.glob('*/metrics.json'):
    path = file.parent
    if not (path / 'run_config.json').is_file(): continue
    cfg = json.loads((path / 'run_config.json').read_text())
    if cfg['data_fingerprint'] != DATA_FINGERPRINT or cfg['tokenizer_hash'] != TOKENIZER_HASH: continue
    row = json.loads(file.read_text()); row['path'] = str(path); records.append(row)
complete = [row for row in records if row['training_complete']]
methods = {row['method'] for row in complete}; architectures = {row['architecture'] for row in complete}
digests = []; best_per_origin = {}
for row in sorted(complete, key=lambda r: r['micro_f1_at_7'], reverse=True):
    path = Path(row['path']); predictions = json.loads((path / 'prediction.json').read_text()); validate_predictions(predictions, 'test')
    digests.append(hashlib.sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest())
    best_per_origin.setdefault(row['architecture'], {'run': row['run_name'], 'method': row['method'],
        'dev_micro_f1_at_7': row['micro_f1_at_7'], 'submission': str(path / 'submission.zip')})
coverage = {'four_completed_runs': len(complete) >= 4, 'six_completed_runs': len(complete) >= 6,
    'six_distinct_prediction_contents': len(set(digests)) >= 6,
    'all_four_phase1_origins': architectures >= set(MODEL_CLASSES),
    'three_non_kd_adaptation_methods': len(methods - {'kd'}) >= 3,
    'four_listed_techniques': len(methods & {'full', 'partial', 'lorma', 'prompt', 'ia3', 'kd'}) >= 4,
    'compulsory_lorma': 'lorma' in methods, 'compulsory_kd': 'kd' in methods,
    'competition_acceptance_verified': False}
save_json(coverage, PROJECT_DIR / 'coverage.json'); save_json(best_per_origin, PROJECT_DIR / 'best_per_architecture.json')
columns = ['run_name', 'architecture', 'method', 'micro_f1_at_7', 'micro_precision_at_7', 'micro_recall_at_7', 'dense_micro_f1_at_7',
           'bm25_micro_f1_at_7', 'best_epoch', 'total_parameters', 'trainable_parameters', 'training_seconds', 'training_complete', 'path']
with (PROJECT_DIR / 'experiment_summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in sorted(records, key=lambda r: r['micro_f1_at_7'], reverse=True):
        writer.writerow({key: row.get(key) for key in columns}); show_scores(row, row['run_name'])
print('Requirement coverage:', coverage)
print('Best per Phase 1 origin:', best_per_origin)
