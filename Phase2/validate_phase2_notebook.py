"""Offline checks: real Phase 1 checkpoints plus tiny end-to-end training/resumption/KD."""
import ast
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs781-phase2-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'CS781_Phase2_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
namespace = {'__name__': '__phase2_test__'}

def source(index):
    return ''.join(notebook['cells'][index]['source'])

def execute(index):
    exec(compile(source(index), f'notebook cell {index}', 'exec'), namespace)

for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and not source(index).startswith('%'):
        compile(source(index), f'notebook cell {index}', 'exec')
print('Notebook format and syntax validated.', flush=True)
# Import setup modules and utility functions, excluding interactive Drive/HF authentication.
tree = ast.parse(source(4))
selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=selected, type_ignores=[]), 'test setup', 'exec'), namespace)
namespace.update({'BASE_DIR': ROOT, 'PHASE1_DIR': ROOT,
                  'PROJECT_DIR': Path(tempfile.mkdtemp(prefix='cs781-phase2-checks-')),
                  'DEVICE': namespace['torch'].device('cpu'), 'AMP_DTYPE': namespace['torch'].float16})
namespace['torch'].set_num_threads(2)
execute(6)
# Test schema normalization without downloading protected data.
tree = ast.parse(source(8))
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(ast.Module(body=selected, type_ignores=[]), 'schema definitions', 'exec'), namespace)
labels = ['APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT', 'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO']
bio = ['O'] + [p+'-'+l for l in labels for p in ['B','I']]
namespace.update({'ENTITY_LABELS': labels, 'BIO_LABELS': bio, 'NUM_LABELS': len(bio),
                  'LABEL2ID': {l:i for i,l in enumerate(bio)}, 'ID2LABEL': dict(enumerate(bio)),
                  'DATA_FINGERPRINT': 'synthetic', 'SPLIT_INFO': {'synthetic': True},
                  'train_ids': ['train'], 'val_ids': ['val'],
                  'all_text': {'train': 'SUPREME COURT of India', 'val': 'SUPREME COURT of India.'},
                  'all_gold': {'train': [(0,13,'COURT')], 'val': [(0,13,'COURT')]},
                  'test_text': {'test': 'SUPREME COURT of India.'}})
execute(10)
for index in [12,13,14,15,16,18,20,22,24]:
    execute(index)
assert namespace['normalize_gold']({'x': [[0,3,'APP','Ann']]}, {'x': 'Ann'}) == {'x': [(0,3,'APP')]}
try:
    namespace['normalize_gold']({'x': [[0,2,'APP','Ann']]}, {'x': 'Ann'})
    raise AssertionError('Mismatching span text accepted')
except ValueError:
    pass
perfect = namespace['strict_metrics']({'val': [[0,13,'COURT']]}, namespace['all_gold'], ['val'])
assert perfect['per_label']['COURT']['f1'] == 1
assert perfect['macro_f1'] == 1 / len(labels)
print('Schema validation and strict metrics checked.', flush=True)
# Real local checkpoint compatibility and token head outputs, with backward passes.
torch = namespace['torch']
for name in ['decoder_full','encoder_partial','hybrid_lora','lightweight_lorma','decoder_bitfit','lightweight_kd']:
    spec = namespace['EXPERIMENTS'][name]
    model, meta = namespace['make_model'](spec)
    batch = namespace['collate_windows']([namespace['train_ds'][0]])
    model.eval()
    with torch.no_grad():
        solo = model(batch['input_ids'], batch['attention_mask'])
        padded = namespace['collate_windows']([namespace['train_ds'][0],
            {'input_ids': namespace['np'].concatenate([batch['input_ids'][0].numpy(), [5, 6, 7]]),
             'labels': namespace['np'].zeros(batch['input_ids'].shape[1]+3, dtype=namespace['np'].int64)}])
        grouped = model(padded['input_ids'], padded['attention_mask'])[0:1, :solo.size(1)]
        torch.testing.assert_close(solo, grouped, atol=2e-5, rtol=2e-5)
    model.train()
    logits = model(batch['input_ids'], batch['attention_mask'])
    loss = namespace['F'].cross_entropy(logits.reshape(-1,len(bio)), batch['labels'].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    if spec['method'] in {'lora', 'lorma'}:
        adapted = [m for m in model.modules() if isinstance(m,namespace['LowRankLinear'])]
        assert adapted and any(m.B.weight.grad.abs().sum() > 0 for m in adapted)
    if spec['method'] == 'kd':
        assert len(model.backbone.blocks) == 6
    print(f'{name}: strict checkpoint load, finite forward/backward; parameters={meta["total_parameters"]}', flush=True)
    del model, logits, loss
    namespace['gc'].collect()

# Verify constrained decoding against independent brute-force path scoring.
import itertools
np = namespace['np']
active_tags = [0, namespace['LABEL2ID']['B-APP'], namespace['LABEL2ID']['I-APP'], namespace['LABEL2ID']['B-RESP']]
emissions = np.full((3, len(bio)), -1000, dtype=np.float32)
emissions[:, active_tags] = np.array([[1, 2, 9, 0], [0, 1, 6, 2], [1, 0, 7, 0]], dtype=np.float32)
def legal_path(path):
    if bio[path[0]].startswith('I-'):
        return False
    for prev, current in zip(path, path[1:]):
        if bio[current].startswith('I-') and bio[prev] not in ['B-'+bio[current][2:], 'I-'+bio[current][2:]]:
            return False
    return True
paths = [p for p in itertools.product(active_tags, repeat=3) if legal_path(p)]
expected = max(paths, key=lambda p: sum(emissions[t, label] for t, label in enumerate(p)))
actual = namespace['constrained_bio_tags'](emissions)
assert actual == list(expected)
print('BIO Viterbi agrees with independent exhaustive path search.', flush=True)
# A crafted validation case exercises selecting an improved decoder, not only fallback.
truth = namespace['encoded_docs']['val']['labels']
court_begin = namespace['LABEL2ID']['B-COURT']
court_inside = namespace['LABEL2ID']['I-COURT']
assert np.count_nonzero(truth == court_inside) > 0
scores = np.full((len(truth), len(bio)), -20.0, dtype=np.float32)
scores[:, 0] = 5
scores[truth == court_begin, court_begin] = 4
scores[truth == court_inside, court_inside] = 10
class ScriptedScores(torch.nn.Module):
    def forward(self, ids, mask):
        return torch.tensor(scores[:ids.shape[1]])[None, :, :].expand(ids.shape[0], -1, -1)
fake_model = ScriptedScores()
val_docs = {'val': namespace['all_text']['val']}
base_predictions = namespace['predict_documents'](fake_model, val_docs, namespace['encoded_docs'])
base_metric = namespace['strict_metrics'](base_predictions, namespace['all_gold'], ['val'])
selection_path = namespace['PROJECT_DIR'] / 'decoder_selection_check'
selection_path.mkdir()
namespace['save_json'](base_metric, selection_path / 'best_validation_metrics.json')
improved, decoder, report = namespace['verify_and_select_decoder'](fake_model, selection_path)
assert improved['macro_f1'] > base_metric['macro_f1']
assert improved['per_label']['COURT']['f1'] == 1.0 and decoder['method'] == 'viterbi'
print('Validation decoder search selects an improved exact-span path.', flush=True)

# Tiny models reuse the exact classes; replace only disk loading for fast pipeline tests.
original_loader = namespace['load_phase1']
def tiny_loader(architecture):
    config = dict(vocab_size=namespace['VOCAB_SIZE_ACTUAL'])
    if architecture == 'decoder_only':
        config.update(n_layer=2,n_embd=16,n_head=2,d_ff=32,max_len=384)
    elif architecture == 'encoder_decoder':
        config.update(n_enc_layer=2,n_dec_layer=2,d_model=16,n_head=2,d_ff=32)
    elif architecture == 'hybrid_prefixlm':
        config.update(n_layer=2,d_model=16,n_head=2,d_ff=32,max_len=384)
    else:
        config.update(n_layer=4,d_model=16,n_head=2,n_kv_head=1,d_ff=32,max_len=384)
    b = namespace['MODEL_CLASSES'][architecture](**config)
    total = sum(p.numel() for p in b.parameters())
    del b.head
    if architecture == 'encoder_decoder':
        del b.dec_layers,b.dec_rel_bias,b.ln_dec_f
    return b,config,total
namespace['load_phase1'] = tiny_loader
namespace.update({'EPOCHS': 1, 'GRAD_ACCUM_STEPS': 3, 'SAVE_EVERY_STEPS': 1,
                  'BATCH_SIZE': 1, 'RUN_TAG': 'test', 'TEACHER_EXPERIMENT': 'decoder_full'})
# Four windows exercise a short final accumulation group.
namespace['train_ds'].rows *= 4
metrics = namespace['run_experiment']('decoder_full')
assert metrics['optimizer_steps'] == 2 and metrics['training_complete']
# Completed resume must not take further optimizer steps.
resumed = namespace['run_experiment']('decoder_full')
assert resumed['optimizer_steps'] == 2
# Use a larger teacher (same vocabulary embedding, dimension=16, four layers vs two).
namespace['TEACHER_EXPERIMENT'] = 'lightweight_lorma'
namespace['run_experiment']('lightweight_lorma')
kd = namespace['run_experiment']('lightweight_kd')
assert kd['training_complete'] and kd['teacher_experiment'] == 'lightweight_lorma'
# Interruption and exact continuation at an optimizer boundary.
namespace['MAX_OPTIMIZER_STEPS'] = 1
namespace['RUN_TAG'] = 'capped'
capped = namespace['run_experiment']('decoder_full')
assert not capped['training_complete'] and capped['optimizer_steps'] == 1
state = namespace['torch_load'](namespace['run_path']('decoder_full') / 'last.pt')
assert state['next_batch'] == 3 and state['epoch'] == 0
# Emulate a runtime interruption checkpoint with the uncapped config, then continue.
namespace['MAX_OPTIMIZER_STEPS'] = None
config = namespace['make_run_config']('decoder_full')
state['run_config'] = config
namespace['save_torch'](state, namespace['run_path']('decoder_full') / 'last.pt')
namespace['save_json'](config, namespace['run_path']('decoder_full') / 'run_config.json')
continued = namespace['run_experiment']('decoder_full')
assert continued['training_complete'] and continued['optimizer_steps'] == 2
uninterrupted_state = namespace['torch_load'](namespace['PROJECT_DIR'] / 'decoder_full_test' / 'last.pt')
continued_state = namespace['torch_load'](namespace['run_path']('decoder_full') / 'last.pt')
for key, value in uninterrupted_state['model'].items():
    torch.testing.assert_close(value, continued_state['model'][key], rtol=0, atol=0)
print('Interrupted training exactly matches uninterrupted final weights.', flush=True)
namespace['selected_experiments'] = ['decoder_full']
execute(28)
execute(30)
print('PASS: real checkpoints, all adaptations, training, saved state reload, KD, resume, ZIP validation.', flush=True)
