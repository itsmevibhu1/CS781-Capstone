"""Offline LSI checks: real Phase 1 loading + independent metrics + synthetic end-to-end runs."""
import ast
import csv
import io
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs781-phase5-mpl')
ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'Phase5' / 'CS781_Phase5_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
ns = {'__name__': 'phase5_offline_checks'}


def get(prefix):
    return next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))


def execute(prefix):
    exec(compile(get(prefix), prefix, 'exec'), ns)


for cell in notebook['cells']:
    s = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and not s.startswith('%'):
        compile(s, cell['id'], 'exec')
setup = ast.parse(get('import os, json'))
parts = [node for node in setup.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=parts, type_ignores=[]), 'setup', 'exec'), ns)
ns.update(BASE_DIR=ROOT, PHASE1_DIR=ROOT, PROJECT_DIR=Path(tempfile.mkdtemp(prefix='cs781-phase5-tests-')),
          DEVICE=ns['torch'].device('cpu'), AMP_DTYPE=ns['torch'].float16, IN_COLAB=False)
torch = ns['torch']; np = ns['np']; torch.set_num_threads(2)
execute('SEED =')
mock = ns['PROJECT_DIR'] / 'mock'; mock.mkdir(); files = {}
for split in ['train', 'dev', 'test']:
    records = []
    for i in range(8 if split == 'train' else 4):
        row = {'id': f'{split}_{i}', 'text': f'The {split} court discusses appeal {i}. ' + 'The statute applies to this legal claim. ' * 20}
        if split != 'test':
            row['labels'] = [[1, 4], [9], [1], [4, 9]][i % 4]
        if split != 'test' and i == 1:
            row['id'] = f'{split}_0'
        if split == 'test' and i == 0:
            row['id'] = 123  # Original integer ID must survive every export.
            row['text'] = ''
        if split == 'test':
            row['labels'] = 'PRIVATE_LABELS_MUST_BE_IGNORED'
        records.append(row)
    if split == 'train':
        # Official-style cross-split duplicates, with different training labels.
        records.extend([
            {'id': 'train_copy_of_dev', 'text': 'The dev court discusses appeal 2. ' + 'The statute applies to this legal claim. ' * 20, 'labels': [9]},
            {'id': 'train_copy_of_test', 'text': 'The test court discusses appeal 1. ' + 'The statute applies to this legal claim. ' * 20, 'labels': [4]},
        ])
    path = mock / (split + '.json')
    if split == 'dev':
        buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=['id', 'text', 'labels'])
        writer.writeheader()
        for row in records:
            row = dict(row); row['labels'] = json.dumps(row['labels']); writer.writerow(row)
        path.write_text(buffer.getvalue())
    elif split == 'test':
        path.write_text('\n'.join(json.dumps(row) for row in records))
    else:
        path.write_text(json.dumps(records))
    files[f'Phase_5_LSI/{"Test" if split == "test" else "Train"}/{split}.json'] = path
ns['HF_TOKEN'] = 'hf_offline_test'
ns['hf_hub_download'] = lambda filename, **kwargs: files[filename]
execute('FILES =')
assert ns['LABEL_IDS'] == [1, 4, 9] and ns['NUM_LABELS'] == 3
assert ns['TRAIN_COUNTS'].tolist() == [4, 4, 4]
assert len(ns['DATA']['train']) == 8 and len(ns['DATA']['dev']) == 4
assert all(row['label'] is None for row in ns['DATA']['test'].values())
assert ns['split_audit']['source_counts']['train'] == 10
assert ns['split_audit']['excluded_train_instances'] == 2
assert ns['split_audit']['remaining_train_instances_matching_heldout_text'] == 0
assert ns['split_audit']['retained_counts'] == {'train': 8, 'dev': 4, 'test': 4}
assert [row['id'] for row in ns['DATA']['test'].values()][0] == 123
# A document occurring in all three files is removed only from TRAIN.
sample = {'train': {'a': {'id': 'a', 'text': 'shared', 'label': [1]},
                    'b': {'id': 'b', 'text': 'shared', 'label': [9]},
                    'c': {'id': 'c', 'text': 'unique training', 'label': [4]},
                    'empty': {'id': 'empty', 'text': '  ', 'label': []}},
          'dev': {'d': {'id': 'd', 'text': 'shared', 'label': [4]}},
          'test': {'t': {'id': 456, 'text': 'shared', 'label': None}}}
before = json.dumps(sample, sort_keys=True)
filtered, audit = ns['resolve_split_overlaps'](sample, 'exclude_train')
assert set(filtered['train']) == {'c', 'empty'}
assert filtered['dev'] is sample['dev'] and filtered['test'] is sample['test']
assert json.dumps(sample, sort_keys=True) == before
assert audit['excluded_train_instances'] == 2 and audit['dev/test']['identical_texts'] == 1
assert audit['train/dev']['examples'][0]['different_label_sets'] is True
reported, audit = ns['resolve_split_overlaps'](sample, 'report')
assert len(reported['train']) == 4 and audit['remaining_train_instances_matching_heldout_text'] == 2
try: ns['resolve_split_overlaps'](sample, 'error')
except ValueError: pass
else: raise AssertionError('Strict overlap policy did not stop')
all_overlap = {**sample, 'train': {'a': sample['train']['a']}}
try: ns['resolve_split_overlaps'](all_overlap, 'exclude_train')
except ValueError as exc: assert 'No training cases remain' in str(exc)
else: raise AssertionError('Empty filtered training set accepted')
ns['save_json'](ns['split_audit'], ns['PROJECT_DIR'] / 'split_audit.json')
print('PASS: cross-split copies excluded from training; every held-out ID retained; statistics recomputed from filtered training.', flush=True)
assert ns['normalize_cases']([{'id': 7, 'text': ['a', 'b'], 'labels': [1, 1, 9]}], True)['7']['label'] == [1, 9]
for invalid in [True, 1, 'word', [True], [-1], [1.0], ['1']]:
    try:
        ns['parse_statutes'](invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid label list accepted: ' + repr(invalid))
fixture = mock / 'formats.json'
for data, encoding in [(json.dumps([{'id': 'x', 'text': 'न्याय', 'labels': [1]}]), 'utf-16'),
                       (repr([{'id': 'x', 'text': 'न्याय', 'labels': [1]}]), 'utf-8')]:
    fixture.write_text(data, encoding=encoding)
    raw, fmt = ns['parse_case_file'](fixture)
    assert raw[0]['labels'] == [1]
for bad in ['', '<html>error</html>', '[{"id": 1,', '{"id": 1}\nBROKEN']:
    fixture.write_text(bad)
    try: ns['parse_case_file'](fixture)
    except ValueError: pass
    else: raise AssertionError('Malformed case export accepted')
execute('TOKENIZER_DIR=')
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention',
               'class GQAAttention', 'def torch_load', 'class CaseFeatures', 'class LowRankMultiplicative',
               'def gold_matrix', 'def capture_rng', 'def restore_rng', 'def base_config', 'def positive_weights', 'def optimizer_groups']:
    execute(prefix)
execute('assert NUM_LABELS')
ensemble_tree = ast.parse(get('def build_ensemble'))
exec(compile(ast.Module(body=[node for node in ensemble_tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[]), 'ensemble', 'exec'), ns)
print('PASS: notebook syntax/schema, robust three-file parsing, duplicate IDs, label mapping and test-label isolation.', flush=True)

# Independent set counting across random multi-label cases checks macro, micro and exact sets.
rng = np.random.default_rng(23)
original_scope = ns['METRIC_LABEL_SCOPE']
ns['METRIC_LABEL_SCOPE'] = 'known'
for count in [1, 2, 5, 11]:
    for _ in range(20):
        gold = rng.integers(0, 2, (count, 3)); pred = rng.integers(0, 2, (count, 3))
        metrics = ns['multilabel_metrics'](gold, pred)
        f1 = []
        for label in range(3):
            reference = {i for i in range(count) if gold[i, label]}
            prediction = {i for i in range(count) if pred[i, label]}
            f1.append(2 * len(reference & prediction) / (len(reference) + len(prediction)) if reference or prediction else 0.0)
        assert math.isclose(metrics['macro_f1_known'], sum(f1) / 3, abs_tol=1e-12)
        pairs_gold = {(i, j) for i in range(count) for j in range(3) if gold[i, j]}
        pairs_pred = {(i, j) for i in range(count) for j in range(3) if pred[i, j]}
        denominator = len(pairs_gold) + len(pairs_pred)
        independent = 2 * len(pairs_gold & pairs_pred) / denominator if denominator else 0.0
        assert math.isclose(metrics['micro_f1'], independent, abs_tol=1e-12)
        assert metrics['subset_accuracy'] == sum(set(np.flatnonzero(a)) == set(np.flatnonzero(b)) for a, b in zip(gold, pred)) / count
        for j in range(3):
            probabilities = np.round(rng.random(count), 2)
            selected = ns['exact_label_threshold'](probabilities, gold[:, j])
            def score(threshold):
                chosen = probabilities >= threshold
                den = int(chosen.sum() + gold[:, j].sum())
                return 2 * int((chosen * gold[:, j]).sum()) / den if den else 0.0
            if gold[:, j].sum():
                assert math.isclose(score(selected), max(score(t) for t in [0.5, *probabilities.tolist()]), abs_tol=1e-12)
        policy, selected, baseline = ns['optimal_threshold'](rng.random((count, 3)), gold)
        assert selected['macro_f1'] + 1e-12 >= baseline['macro_f1']
ns['METRIC_LABEL_SCOPE'] = original_scope
perfect = ns['multilabel_metrics'](np.eye(3, dtype=int), np.eye(3, dtype=int))
assert perfect['macro_f1_known'] == 1.0 and perfect['macro_f1_reference_present'] == 1.0
assert math.isclose(perfect['macro_f1_reference_range'], 0.3)  # Sparse IDs 1,4,9; omitted columns count zero.
# Under a reference-derived dimension, columns above max gold ID do not enter the scorer.
ranged = ns['multilabel_metrics']([[1, 0, 0]], [[1, 0, 1]])
assert ranged['micro_f1'] == 1.0 and ranged['subset_accuracy'] == 1.0
assert ranged['micro_f1_known'] < 1.0 and ranged['subset_accuracy_known'] == 0.0
for length in [0, 1, 31, 32, 10000]:
    ranges = ns['window_ranges'](length, 32, 6)
    assert ns['covered_tokens'](ranges) == length
    for policy in ['spread', 'head', 'tail', 'head_tail', 'random_mix', 'all']:
        assert ns['select_ranges'](ranges, 4, policy, 'case', 2) == ns['select_ranges'](ranges, 4, policy, 'case', 2)
print('PASS: independent multi-label metric formulas, sparse-ID conventions, exact label thresholds and safe calibration.', flush=True)

# All core real checkpoints must load strictly and produce independent statute logits/gradients.
for name in ns['CORE_EXPERIMENTS']:
    cfg = ns['base_config'](name, {'chunk_length': 32, 'chunk_overlap': 6, 'max_chunks': 2, 'chunk_batch_size': 2, 'doc_hidden': 16})
    model, metadata = ns['make_model'](cfg)
    dataset = ns['CaseDataset']('train', cfg)
    one = ns['collate_cases']([dataset[0]])
    assert one['labels'].shape == (1, 3) and one['labels'].dtype == torch.float32
    model.eval()
    with torch.no_grad():
        logits = ns['forward_batch'](model, one)
        grouped = ns['forward_batch'](model, ns['collate_cases']([dataset[0], dataset[1]]))[:1]
        torch.testing.assert_close(logits, grouped, atol=3e-5, rtol=3e-5)
    model.train()
    logits = ns['forward_batch'](model, one)
    loss = ns['multilabel_loss'](logits, one['labels'], ns['positive_weights'](cfg), cfg)
    loss.backward()
    assert logits.shape == (1, 3) and torch.isfinite(loss) and model.classifier[-1].weight.grad is not None
    if cfg['spec']['method'] == 'lorma':
        assert any(m.B.weight.grad is not None and m.B.weight.grad.abs().sum() > 0
                   for m in model.modules() if isinstance(m, ns['LowRankMultiplicative']))
    if cfg['spec']['method'] == 'adapters':
        assert any(m.up.weight.grad.abs().sum() > 0 for m in model.modules() if isinstance(m, ns['ResidualAdapter']))
    print('PASS:', name, 'real Phase 1 load, finite multi-label forward/backward, batch invariance.', flush=True)
    del model, logits, loss; ns['gc'].collect()

# Small compatible architectures exercise all actual training/resume/export paths quickly.
def tiny_loader(architecture):
    cfg = {'vocab_size': ns['VOCAB_SIZE_ACTUAL']}
    if architecture == 'decoder_only': cfg.update(n_layer=6, n_embd=16, n_head=2, d_ff=32, max_len=384)
    elif architecture == 'encoder_decoder': cfg.update(n_enc_layer=3, n_dec_layer=2, d_model=16, n_head=2, d_ff=32)
    elif architecture == 'hybrid_prefixlm': cfg.update(n_layer=3, d_model=16, n_head=2, d_ff=32, max_len=384)
    else: cfg.update(n_layer=4, d_model=16, n_head=2, n_kv_head=1, d_ff=32, max_len=384)
    backbone = ns['MODEL_CLASSES'][architecture](**cfg); count = sum(p.numel() for p in backbone.parameters()); del backbone.head
    if architecture == 'encoder_decoder': del backbone.dec_layers, backbone.dec_rel_bias, backbone.ln_dec_f
    return backbone, cfg, count


ns['load_phase1'] = tiny_loader
ns.update(EPOCHS=1, BATCH_SIZE=2, GRAD_ACCUM_STEPS=3, CHUNK_LENGTH=32, CHUNK_OVERLAP=6, MAX_CHUNKS=2,
          CHUNK_BATCH_SIZE=2, DOC_HIDDEN=16, SAVE_EVERY_STEPS=1, RUN_TAG='tiny')
for name in ns['EXTRA_EXPERIMENTS']:
    cfg = ns['base_config'](name); model, _ = ns['make_model'](cfg)
    if cfg['spec']['method'] == 'gradual': ns['unfreeze_last_layers'](model, 2)
    batch = ns['collate_cases']([ns['CaseDataset']('train', cfg)[0]])
    loss = ns['multilabel_loss'](ns['forward_batch'](model, batch), batch['labels'], ns['positive_weights'](cfg), cfg)
    loss.backward(); assert torch.isfinite(loss)
for loss_name in ['bce', 'focal', 'asymmetric']:
    cfg = ns['base_config']('decoder_full', {'loss_name': loss_name})
    logits = torch.tensor([[-80., 80., 0.], [2., -2., 1.]], requires_grad=True)
    targets = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
    loss = ns['multilabel_loss'](logits, targets, ns['positive_weights'](cfg), cfg)
    loss.backward(); assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
for pooling in ['attention', 'mean', 'max', 'mean_max', 'transformer']:
    cfg = ns['base_config']('decoder_full', {'pooling': pooling, 'label_graph_weight': 0.2})
    model, _ = ns['make_model'](cfg)
    batch = ns['collate_cases']([ns['CaseDataset']('train', cfg)[0]])
    output = ns['forward_batch'](model, batch)
    assert output.shape == (1, 3) and torch.isfinite(output).all()
    output.sum().backward()
full = ns['run_experiment']('decoder_full')
assert full['training_complete'] and full['optimizer_steps'] == 2
assert ns['run_experiment']('decoder_full')['optimizer_steps'] == 2
for name in ['encoder_partial', 'hybrid_lorma', 'hybrid_adapters', 'lightweight_lorma']:
    assert ns['run_experiment'](name)['training_complete']
ns.update(RUN_TAG='kd_other', MAX_CHUNKS=3)
student = ns['run_experiment']('lightweight_kd')
assert student['training_complete'] and student['teacher_run'] is not None
assert student['total_parameters'] < json.loads((Path(student['teacher_run']) / 'metrics.json').read_text())['total_parameters']
teacher_cache = Path(student['teacher_run']) / 'teacher_train_logits.pt'
mtime = teacher_cache.stat().st_mtime_ns
ns['RUN_TAG'] = 'kd_warm'
ns['run_experiment']('lightweight_kd')
cold = ns['torch_load'](ns['PROJECT_DIR'] / 'lightweight_kd_kd_other' / 'last.pt')
warm = ns['torch_load'](ns['run_path']('lightweight_kd') / 'last.pt')
for key, value in cold['model'].items(): torch.testing.assert_close(value, warm['model'][key], atol=0, rtol=0)
assert teacher_cache.stat().st_mtime_ns == mtime
ns.update(RUN_TAG='interrupted', MAX_CHUNKS=2, MAX_OPTIMIZER_STEPS=1)
assert not ns['run_experiment']('decoder_full')['training_complete']
state = ns['torch_load'](ns['run_path']('decoder_full') / 'last.pt')
assert state['next_batch'] == 3 and state['epoch'] == 0
ns['MAX_OPTIMIZER_STEPS'] = None
cfg = ns['base_config']('decoder_full'); state['run_config'] = cfg
ns['save_json'](cfg, ns['run_path']('decoder_full') / 'run_config.json')
ns['save_torch'](state, ns['run_path']('decoder_full') / 'last.pt')
assert ns['run_experiment']('decoder_full')['optimizer_steps'] == 2
resumed = ns['torch_load'](ns['run_path']('decoder_full') / 'last.pt')
uninterrupted = ns['torch_load'](ns['PROJECT_DIR'] / 'decoder_full_tiny' / 'last.pt')
for key, value in resumed['model'].items(): torch.testing.assert_close(value, uninterrupted['model'][key], atol=0, rtol=0)
ns['RUN_TAG'] = 'gradual'
gradual = ns['run_experiment']('decoder_gradual')
assert gradual['trainable_parameters_final'] > gradual['trainable_parameters_initial']
assert ns['build_ensemble']() is not None
rows = [{'id': row['id'], 'pred': [1, 9]} for row in ns['DATA']['test'].values()]
ns['validate_submission'](rows, ns['DATA']['test'])
for invalid in [[True], [1, 1], [999], ['1'], 1]:
    bad = [dict(row) for row in rows]; bad[0]['pred'] = invalid
    try: ns['validate_submission'](bad, ns['DATA']['test'])
    except ValueError: pass
    else: raise AssertionError('Malformed submission accepted: ' + repr(invalid))
folder = ns['PROJECT_DIR'] / 'decoder_full_tiny'
before = ns['load_probability_file'](folder, 'dev')
assert before.shape == (4, 3)
saved = json.loads((folder / 'dev_probabilities.json').read_text())
saved['label_ids'] = [9, 4, 1]
ns['save_json'](saved, folder / 'dev_probabilities.json')
try: ns['load_probability_file'](folder, 'dev')
except ValueError: pass
else: raise AssertionError('Mismatched label order accepted')
saved['label_ids'] = [1, 4, 9]; ns['save_json'](saved, folder / 'dev_probabilities.json')
execute('import csv\nsummary')
coverage = json.loads((ns['PROJECT_DIR'] / 'coverage.json').read_text())
assert coverage['all_four_phase1_origins'] and coverage['compulsory_lorma'] and coverage['compulsory_kd']
assert len(json.loads((ns['PROJECT_DIR'] / 'best_per_architecture.json').read_text())) == 4
print('PASS: six core real checkpoints, seven extra methods, all pooling/loss choices, larger-teacher KD/cache, exact resume, ensemble, reports and ZIP verification.', flush=True)
print('Artifacts:', ns['PROJECT_DIR'], flush=True)
