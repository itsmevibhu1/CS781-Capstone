import csv
records = []
for file in PROJECT_DIR.glob('*/metrics.json'):
    folder = file.parent
    if not (folder / 'run_config.json').is_file(): continue
    cfg = json.loads((folder / 'run_config.json').read_text())
    if cfg['data_fingerprint'] != DATA_FINGERPRINT or cfg['tokenizer_hash'] != TOKENIZER_HASH or cfg['metric_config'] != metric_config(): continue
    row = json.loads(file.read_text()); row['path'] = str(folder); records.append(row)
completed = [row for row in records if row['training_complete'] and row['full_validation']]
methods = {row['method'] for row in completed}; origins = {row['architecture'] for row in completed}
digests = []
best_per_origin = {}
for row in sorted(completed, key=lambda item: item['bertscore_f1'], reverse=True):
    folder = Path(row['path']); prediction = json.loads((folder / 'prediction.json').read_text())
    validate_rows(prediction, 'test', list(DATA['test']))
    digests.append(hashlib.sha256(json.dumps(prediction, sort_keys=True).encode()).hexdigest())
    best_per_origin.setdefault(row['architecture'], {'run': row['run_name'], 'bertscore_f1': row['bertscore_f1'], 'submission': str(folder / 'submission.zip')})
coverage = {'four_completed_experiments': len(completed) >= 4, 'six_completed_experiments': len(completed) >= 6,
            'six_distinct_prediction_contents': len(set(digests)) >= 6, 'all_four_phase1_origins': origins >= set(MODEL_CLASSES),
            'three_non_kd_adaptations': len(methods - {'kd'}) >= 3,
            'four_techniques_listed_in_phase6': len(methods & {'lora', 'lorma', 'houlsby', 'pfeiffer', 'prompt', 'kd'}) >= 4,
            'compulsory_lorma': 'lorma' in methods, 'compulsory_kd': 'kd' in methods,
            'competition_acceptance_verified': False}
save_json(coverage, PROJECT_DIR / 'coverage.json'); save_json(best_per_origin, PROJECT_DIR / 'best_per_architecture.json')
columns = ['run_name', 'architecture', 'method', 'bertscore_f1', 'rouge1_f1', 'rougeL_f1', 'best_epoch', 'training_seconds',
           'total_parameters', 'trainable_parameters', 'peak_gpu_memory_bytes', 'training_complete', 'full_validation', 'path']
with (PROJECT_DIR / 'experiment_summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in sorted(records, key=lambda item: item['bertscore_f1'], reverse=True):
        writer.writerow({key: row.get(key) for key in columns})
        print(row['run_name'], 'validation BERTScore=', round(row['bertscore_f1'], 6), 'complete=', row['training_complete'])
print('Required experiment coverage:', coverage)
print('Best per Phase 1 architecture:', best_per_origin)
