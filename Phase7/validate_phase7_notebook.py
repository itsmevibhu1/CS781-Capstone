"""Offline PCR verification with synthetic compatible Phase 1 states and six split files."""
import ast
import json
import math
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg'); os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs781-phase7-mpl')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'Phase7' / 'CS781_Phase7_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
ns = {'__name__': 'phase7_offline_verification'}
def get(prefix):
    return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))
def execute(prefix): exec(compile(get(prefix), prefix, 'exec'), ns)
for cell in notebook['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and not source.startswith('%'): compile(source, cell['id'], 'exec')
setup = ast.parse(get('import os, json'))
exec(compile(ast.Module(body=[node for node in setup.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[]), 'setup', 'exec'), ns)
temporary = Path(tempfile.mkdtemp(prefix='cs781-phase7-verify-'))
ns.update(BASE_DIR=temporary, PHASE1_DIR=temporary / 'phase1', PROJECT_DIR=temporary / 'results', IN_COLAB=False,
          DEVICE=ns['torch'].device('cpu'), AMP_DTYPE=ns['torch'].float16)
ns['PHASE1_DIR'].mkdir(); ns['PROJECT_DIR'].mkdir()
torch = ns['torch']; np = ns['np']; torch.set_num_threads(2)
execute('SEED =')
files = {}; calls = []; corpus = []; mock = temporary / 'data'; mock.mkdir()
for split, base in [('train', 0), ('dev', 100), ('test', 200)]:
    candidates = [{'id': f'{base + i + 1:010d}', 'text': [f'The {split} court considers statute {i}.',
                   ('The appeal concerns jurisdiction and legal precedent. ' * (i % 3 + 1))]} for i in range(12)]
    queries = []
    for i in range(9 if split == 'train' else 3):
        relevant = [base + j + 1 for j in range(9)] if i == 0 else [f'{base + i % 10 + 1:010d}', f'{base + (i + 1) % 10 + 1:010d}']
        if i == (8 if split == 'train' else 2): relevant = []
        queries.append({'id': f'{9000 + base + i:010d}', 'text': [f'The {split} query concerns statute {i}.', 'Prior cases and jurisdiction are relevant.'],
                        'relevant_candidates': relevant if split != 'test' else 'PRIVATE_LABELS_IGNORED'})
    for role, records in [('queries', queries), ('candidates', candidates)]:
        filename = split + '.json' if role == 'queries' else split + '_candidates.json'
        official = 'Phase_7_PCR/' + split.capitalize() + '/' + filename
        path = mock / (split + '_' + role + '.json')
        if split == 'test' and role == 'candidates': path.write_text('\n'.join(json.dumps(row) for row in records))
        else: path.write_text(json.dumps({'data': records} if split == 'dev' else records))
        files[official] = path
        corpus.extend(' '.join(row['text']) for row in records)
tokenizer_dir = ns['PHASE1_DIR'] / 'bpe_tokenizer'; tokenizer_dir.mkdir()
tokenizer = ns['ByteLevelBPETokenizer'](); tokenizer.train_from_iterator(corpus, vocab_size=340, min_frequency=1,
    special_tokens=['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']); tokenizer.save_model(str(tokenizer_dir))
def provider(filename, **kwargs):
    assert filename in files; calls.append(filename); return files[filename]
ns['HF_TOKEN'] = 'hf_offline_test'; ns['hf_hub_download'] = provider
execute('def parse_case_file'); assert len(calls) == 6
assert all(row['relevant'] is None for row in ns['QUERIES']['test'].values())
assert len(ns['TRAIN_KEYS']) == 8
assert ns['QUERIES']['train']['0000009000']['relevant'][0] == '0000000001'
assert len(ns['QUERIES']['dev']['0000009100']['relevant']) == 9
execute('def validate_predictions'); execute('from scipy import sparse')
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention', 'class GQAAttention',
               'def torch_load', 'class ProjectionAdapter', '@torch.no_grad()', 'def capture_rng', 'def restore_rng', 'def base_config']:
    execute(prefix)
ns['PHASE1_HEADS'] = {key: 2 for key in ns['MODEL_CLASSES']}
for architecture, model_class in ns['MODEL_CLASSES'].items():
    cfg = {'vocab_size': ns['VOCAB_SIZE_ACTUAL']}
    if architecture == 'decoder_only': cfg.update(n_layer=4, n_embd=16, n_head=2, d_ff=32, max_len=160)
    elif architecture == 'encoder_decoder': cfg.update(n_enc_layer=3, n_dec_layer=3, d_model=16, n_head=2, d_ff=32)
    elif architecture == 'hybrid_prefixlm': cfg.update(n_layer=4, d_model=16, n_head=2, d_ff=32, max_len=160)
    else: cfg.update(n_layer=4, d_model=16, n_head=2, n_kv_head=1, d_ff=32, max_len=160)
    folder = ns['PHASE1_DIR'] / ('run_' + architecture); folder.mkdir(); torch.save(model_class(**cfg).state_dict(), folder / 'final_model.pt')
ns.update(EPOCHS=1, QUERY_BATCH_SIZE=2, GRAD_ACCUM_STEPS=3, POSITIVES_PER_QUERY=2, NEGATIVES_PER_QUERY=2,
          CHUNK_LENGTH=32, CHUNK_OVERLAP=6, MAX_CHUNKS=2, CHUNK_BATCH_SIZE=3, ENCODE_DOCUMENT_BATCH_SIZE=3,
          EMBEDDING_DIM=16, PROMPT_TOKENS=3, PARTIAL_LAYERS=1, RANK=2, ALPHA=4., SAVE_EVERY_STEPS=1, RUN_TAG='tiny')
execute('assert len(FILES)')
print('PASS: six-file schema, split pools, sentence lists, leading zeros, full gold sets and private-test isolation.', flush=True)
# Independent set-counting implements complete gold micro F1, including >7 and empty gold.
predictions = ns['BM25_BASELINE']['dev']; metric = ns['retrieval_metrics'](predictions)
pairs_p = {(q, value) for q, values in predictions.items() for value in values}
pairs_g = {(q, value) for q, row in ns['QUERIES']['dev'].items() for value in row['relevant']}
assert metric['gold_count'] == 11 and metric['predicted_count'] == 21
assert math.isclose(metric['micro_f1_at_7'], 2 * len(pairs_p & pairs_g) / (len(pairs_p) + len(pairs_g)))
for bad_kind in ['duplicate', 'too_few', 'foreign_pool', 'integer', 'missing_query']:
    bad = {q: list(values) for q, values in ns['BM25_BASELINE']['test'].items()}; first = next(iter(bad))
    if bad_kind == 'duplicate': bad[first][1] = bad[first][0]
    elif bad_kind == 'too_few': bad[first].pop()
    elif bad_kind == 'foreign_pool': bad[first][0] = next(iter(ns['CANDIDATES']['train']))
    elif bad_kind == 'integer': bad[first][0] = int(bad[first][0])
    else: bad.pop(first)
    try: ns['validate_predictions'](bad, 'test')
    except ValueError: pass
    else: raise AssertionError('Malformed output accepted: ' + bad_kind)
# BM25 sparse weights must match its explicit per-term formula for one document/query.
bm = ns['BM25']['dev']; document = ns['CANDIDATES']['dev'][bm.ids[0]]['text']
counts = bm.vectorizer.transform([ns['CANDIDATES']['dev'][key]['text'] for key in bm.ids]).toarray()
query = bm.vectorizer.transform([document]).toarray()[0]; dl = counts.sum(1); df = (counts > 0).sum(0)
idf = np.log1p((len(counts) - df + 0.5) / (df + 0.5))
formula = ((counts * (ns['BM25_K1'] + 1)) / (counts + ns['BM25_K1'] * (1 - ns['BM25_B'] + ns['BM25_B'] * dl[:, None] / dl.mean())) * idf * (query > 0)).sum(1)
np.testing.assert_allclose(bm.scores(document), formula, atol=1e-5, rtol=1e-5)
print('PASS: independent Micro-F1@7/gold-size arithmetic, strict submissions and sparse BM25 formula.', flush=True)
assert ns['stable_order']([-0.2, 0.7, 0.7], ['c', 'b', 'a']).tolist() == [2, 1, 0]
assert ns['citation_signal']('See https://indiankanoon.org/doc/123/ for the precedent.', ['0000000123', '0000000124']).tolist() == [1.0, 0.0]
for name in ns['EXPERIMENTS']:
    cfg = ns['base_config'](name); model, metadata = ns['make_model'](cfg)
    keys = list(ns['QUERIES']['train'])[:2]; batch = ns['make_chunks']('train', 'queries', keys, cfg)
    model.eval()
    with torch.no_grad():
        grouped = model(batch, 'queries'); single = model(ns['make_chunks']('train', 'queries', keys[:1], cfg), 'queries')
        torch.testing.assert_close(single, grouped[:1], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(grouped.norm(dim=-1), torch.ones(len(keys)), atol=1e-5, rtol=1e-5)
    candidates, positive, allowed = ns['sample_training_batch'](keys, cfg, 0, 0)
    for i, query_id in enumerate(keys):
        for j, candidate in enumerate(candidates): assert bool(positive[i, j]) == (candidate in ns['QUERIES']['train'][query_id]['relevant'])
    model.train(); q = model(batch, 'queries'); c = model(ns['make_chunks']('train', 'candidates', candidates, cfg), 'candidates')
    loss = ns['retrieval_loss']((q @ c.T).float(), positive, allowed, cfg); loss.backward(); assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    if cfg['spec']['method'] == 'prompt': assert model.features.soft_prompt.grad is not None
    if cfg['spec']['method'] == 'ia3':
        assert any(m.scale.grad is not None for m in model.modules() if isinstance(m, ns['IA3Projection']))
    print('PASS:', name, 'strict load, padding invariance, normalized vectors and finite gradients.', flush=True)
for objective in ['multi_positive', 'pairwise', 'combined']:
    cfg = ns['base_config']('encoder_full', {'loss_name': objective})
    scores = torch.randn(2, 4, requires_grad=True); positive = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
    allowed = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    loss = ns['retrieval_loss'](scores, positive, allowed, cfg); loss.backward(); assert torch.isfinite(loss) and scores.grad[0, 3] == 0
    kd = ns['distillation_loss'](scores, torch.randn_like(scores), allowed, cfg, cfg); assert torch.isfinite(kd)
for pooling in ['attention', 'mean', 'max', 'mean_max']:
    cfg = ns['base_config']('encoder_full', {'pooling': pooling, 'share_projection': False})
    model, _ = ns['make_model'](cfg); batch = ns['make_chunks']('train', 'queries', list(ns['QUERIES']['train'])[:1], cfg)
    model(batch, 'queries').sum().backward()
print('PASS: all fourteen profiles, all pooling choices, multi-positive masks, objectives and KD masks.', flush=True)
baseline = ns['run_experiment']('encoder_full'); assert baseline['training_complete'] and baseline['optimizer_steps'] == 2
saved_vectors = ns['torch_load'](ns['run_path']('encoder_full') / 'dev_embeddings.pt')
components = ns['ranking_components'](saved_vectors['queries'], saved_vectors['candidates'], 'dev', ns['base_config']('encoder_full'))
pure = {'neural_mode': 'dense', 'dense_weight': 1.0, 'rrf_k': 60, 'rrf_depth': 200, 'citation_weight': 0.0}
pure_predictions = ns['rank_with_policy'](components, 'dev', pure)
for query, row in components.items():
    assert pure_predictions[query] == [list(ns['CANDIDATES']['dev'])[i] for i in row['dense'][:7]]
for i, query in enumerate(saved_vectors['queries']['ids']):
    q = saved_vectors['queries']['chunks'][saved_vectors['queries']['owners'] == i]
    manual = []
    for j in range(len(saved_vectors['candidates']['ids'])):
        c = saved_vectors['candidates']['chunks'][saved_vectors['candidates']['owners'] == j]
        manual.append(float((q @ c.T).max(1).values.mean()))
    assert components[query]['late'].tolist() == ns['stable_order'](manual, list(ns['CANDIDATES']['dev'])).tolist()
assert ns['run_experiment']('encoder_full')['optimizer_steps'] == 2
for name in ['decoder_partial', 'hybrid_lorma', 'lightweight_prompt', 'lightweight_ia3']:
    assert ns['run_experiment'](name)['training_complete']
kd = ns['run_experiment']('lightweight_kd'); assert kd['training_complete']
assert kd['teacher']['teacher_parameters'] > kd['total_parameters']
ns.update(RUN_TAG='interrupted', MAX_OPTIMIZER_STEPS=1)
assert not ns['run_experiment']('encoder_full')['training_complete']
state = ns['torch_load'](ns['run_path']('encoder_full') / 'last.pt'); assert state['next_batch'] == 3
ns['MAX_OPTIMIZER_STEPS'] = None; state['config'] = ns['base_config']('encoder_full')
ns['save_json'](state['config'], ns['run_path']('encoder_full') / 'run_config.json'); ns['save_torch'](state, ns['run_path']('encoder_full') / 'last.pt')
assert ns['run_experiment']('encoder_full')['training_complete']
uninterrupted = ns['torch_load'](ns['PROJECT_DIR'] / 'encoder_full_tiny' / 'last.pt')
resumed = ns['torch_load'](ns['run_path']('encoder_full') / 'last.pt')
for key, value in uninterrupted['model'].items(): torch.testing.assert_close(value, resumed['model'][key], atol=0, rtol=0)
ns.update(RUN_TAG='mining', EPOCHS=2)
assert ns['run_experiment']('encoder_full', {'mine_dense_negatives': True}, 'encoder_mining')['training_complete']
saved = ns['torch_load'](ns['run_path']('encoder_mining') / 'last.pt'); assert saved['mined']
ns.update(RUN_TAG='citation', EPOCHS=1)
assert ns['run_experiment']('encoder_full', {'citation_enabled': True}, 'encoder_citation')['training_complete']
execute('import csv\nrecords')
coverage = json.loads((ns['PROJECT_DIR'] / 'coverage.json').read_text())
assert coverage['all_four_phase1_origins'] and coverage['compulsory_lorma'] and coverage['compulsory_kd'] and coverage['four_listed_techniques']
print('PASS: complete core training, larger-teacher KD, exact resume, train-only dense mining, dev selection, ranks/metrics reload, JSON and ZIP verification.', flush=True)
print('Synthetic artifacts:', temporary, flush=True)
