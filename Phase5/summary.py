import csv
summary = []
for file in PROJECT_DIR.glob('*/metrics.json'):
    folder = file.parent
    if not (folder / 'run_config.json').is_file():
        continue
    cfg = json.loads((folder / 'run_config.json').read_text())
    if cfg['data_fingerprint'] != DATA_FINGERPRINT or cfg['tokenizer_hash'] != TOKENIZER_HASH:
        continue
    if cfg.get('metric_label_scope') != METRIC_LABEL_SCOPE or cfg.get('label_ids') != LABEL_IDS:
        continue
    metrics = json.loads(file.read_text())
    summary.append({**metrics, 'path': str(folder)})
completed = [row for row in summary if row['training_complete']]
methods = {row['method'] for row in completed}
origins = {row['architecture'] for row in completed}
digests = []
for row in completed:
    path = Path(row['path']) / 'prediction.json'
    if path.is_file():
        predictions = json.loads(path.read_text())
        validate_submission(predictions, DATA['test'])
        digests.append(hashlib.sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest())
coverage = {'four_completed_runs': len(completed) >= 4, 'six_completed_runs': len(completed) >= 6,
            'six_distinct_prediction_contents': len(set(digests)) >= 6,
            'all_four_phase1_origins': origins >= set(MODEL_CLASSES),
            'three_adaptation_methods_excluding_kd': len(methods & {'full', 'partial', 'lorma', 'adapters', 'gradual'}) >= 3,
            'four_listed_techniques': len(methods & {'full', 'partial', 'lorma', 'adapters', 'kd'}) >= 4,
            'compulsory_lorma': 'lorma' in methods, 'compulsory_kd': 'kd' in methods,
            'codabench_acceptance_verified': False}
save_json(coverage, PROJECT_DIR / 'coverage.json')
best_per_origin = {}
for row in sorted(completed, key=lambda item: item['macro_f1'], reverse=True):
    best_per_origin.setdefault(row['architecture'], {'run_name': row['run_name'], 'macro_f1': row['macro_f1'],
                                'method': row['method'], 'submission': str(Path(row['path']) / 'submission.zip')})
save_json(best_per_origin, PROJECT_DIR / 'best_per_architecture.json')
columns = ['run_name', 'architecture', 'method', 'macro_f1', 'micro_f1', 'subset_accuracy', 'baseline_macro_f1',
           'best_epoch', 'total_parameters', 'training_seconds', 'peak_gpu_memory_bytes', 'training_complete', 'path']
with (PROJECT_DIR / 'experiment_summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in sorted(summary, key=lambda item: item['macro_f1'], reverse=True):
        writer.writerow({key: row.get(key) for key in columns})
        print(row['run_name'], 'DEV Macro-F1=', round(row['macro_f1'], 6), 'Micro-F1=', round(row['micro_f1'], 6),
              'Subset Accuracy=', round(row['subset_accuracy'], 6), 'complete=', row['training_complete'])
print('Best per Phase 1 origin:', best_per_origin)
print('Coverage (local artifacts, not verified competition submissions):', coverage)
