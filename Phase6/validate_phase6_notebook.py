"""Offline generation verification with synthetic Phase 1 checkpoints and a tiny metric model.

No original weights remain locally following the user's disk cleanup. Synthetic compatible weights
test exact architecture/checkpoint behavior; they are not substitutes for Colab task performance.
"""
import ast
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg'); os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs781-phase6-mpl')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'Phase6' / 'CS781_Phase6_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
ns = {'__name__': 'phase6_offline_checks'}
def get(prefix):
    return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))
def execute(prefix): exec(compile(get(prefix), prefix, 'exec'), ns)
for c in notebook['cells']:
    source = ''.join(c['source'])
    if c['cell_type'] == 'code' and not source.startswith('%'): compile(source, c['id'], 'exec')
setup = ast.parse(get('import os, json'))
exec(compile(ast.Module(body=[n for n in setup.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[]), 'setup', 'exec'), ns)
ns['shutil'] = __import__('shutil')
temporary = Path(tempfile.mkdtemp(prefix='cs781-phase6-verify-'))
ns.update(BASE_DIR=temporary, PHASE1_DIR=temporary / 'phase1', PROJECT_DIR=temporary / 'results', IN_COLAB=False,
          DEVICE=ns['torch'].device('cpu'), AMP_DTYPE=ns['torch'].float16)
ns['PHASE1_DIR'].mkdir(); ns['PROJECT_DIR'].mkdir()
torch = ns['torch']; np = ns['np']; torch.set_num_threads(2)
execute('SEED =')
# Two files only. Duplicate training texts must never be divided between fit and validation.
rows = [{'id': f'doc_{i}', 'document': f'The court considered appeal number {i}. The claim concerns a legal statute. ' * 3,
         'summary': 'The court considered the claim and allowed the appeal.'} for i in range(18)]
rows.append({**rows[0], 'id': 'duplicate_source_different_summary', 'summary': 'The court dismissed the legal claim.'})
test = [{'id': 101, 'document': 'The judge considered a private appeal and decided the claim.', 'summary': 'HIDDEN_DO_NOT_USE'},
        {'id': 'test_empty', 'document': ''}]
rows.append({'id': 'public_test_copy', 'document': test[0]['document'], 'summary': 'Training copy must be excluded.'})
valid_rows = len(rows)
# Reproduce the official-data failure without inventing a summary target.
rows.append({'id': 299, 'document': 'A legal document whose summary is missing.', 'summary': None})
corpus = [row['document'] + ' ' + (row.get('summary') or '') for row in rows]
tokenizer_dir = ns['PHASE1_DIR'] / 'bpe_tokenizer'; tokenizer_dir.mkdir()
tokenizer = ns['ByteLevelBPETokenizer']()
tokenizer.train_from_iterator(corpus, vocab_size=340, min_frequency=1, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>', '<sep>'])
tokenizer.save_model(str(tokenizer_dir))
mock = temporary / 'data'; mock.mkdir()
(mock / 'train.json').write_text(json.dumps(rows)); (mock / 'test.json').write_text(json.dumps(test))
ns['HF_TOKEN'] = 'hf_offline_test'
def provider(filename, **kwargs):
    assert filename in ['Phase_6_Summ/Train/train.json', 'Phase_6_Summ/Test/test.json']
    return mock / ('train.json' if '/Train/' in filename else 'test.json')
ns['hf_hub_download'] = provider
execute('def parse_case_file')
assert len(ns['DATA']['train']) + len(ns['DATA']['dev']) == valid_rows - 1
assert all(row['summary'] is None for row in ns['DATA']['test'].values())
manifest = json.loads((ns['PROJECT_DIR'] / 'split_manifest.json').read_text())
assert manifest['excluded_train_test_matches'] == 1
assert manifest['raw_training_instances'] == len(rows) and manifest['excluded_invalid_training_instances'] == 1
quality = json.loads((ns['PROJECT_DIR'] / 'data_quality_report.json').read_text())
assert quality['train']['reason_counts'] == {'missing_summary': 1}
assert quality['train']['invalid_examples'][0]['id'] == 299
assert len(ns['DATA']['test']) == len(test)
# Both document and summary paragraph lists are normalized in order.
normalized = ns['normalize_summaries']([{'id': 'paragraphs', 'document': ['A court.', 'An appeal.'],
                                       'summary': ['The court', 'allowed the appeal.']}], True)
assert normalized['public:row:0']['summary'] == 'The court\n\nallowed the appeal.'
assert normalized['public:row:0']['text'] == 'A court.\n\nAn appeal.'
incomplete = [
    {'id': 'ok', 'document': 'Source', 'summary': 'Reference'},
    {'id': 'blank', 'document': 'Source', 'summary': '  '},
    {'id': 'missing', 'document': 'Source'},
    {'id': 'list_empty', 'document': 'Source', 'summary': []},
    {'id': 'bad_document', 'document': None, 'summary': 'Reference'},
    {'id': 'bad_summary_type', 'document': 'Source', 'summary': 7},
]
retained = ns['normalize_summaries'](incomplete, True, max_invalid_fraction=1)
assert len(retained) == 1 and retained['public:row:0']['id'] == 'ok'
for options in [{'invalid_policy': 'error'}, {'max_invalid_fraction': 0.05}]:
    try: ns['normalize_summaries'](incomplete, True, **options)
    except ValueError: pass
    else: raise AssertionError('Strict policy/schema-rate guard did not stop')
try: ns['normalize_summaries']([{'id': 'bad_test', 'document': None}], False)
except ValueError as exc: assert 'test rows are never skipped' in str(exc)
else: raise AssertionError('Unusable test source silently skipped')
assert ns['normalize_summaries']([{'id': 456, 'document': '', 'summary': None}], False)['456']['summary'] is None
# Restore full-data audits after isolated helper regression checks.
ns['normalize_summaries'](rows, True); ns['normalize_summaries'](test, False)
print('PASS: missing/blank training summaries audited and excluded, paragraph summaries retained, fraction guard and test-ID preservation.', flush=True)
hashes = [{ns['document_hash'](r) for r in ns['DATA'][s].values()} for s in ['train', 'dev']]
assert hashes[0].isdisjoint(hashes[1])
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention', 'class GQAAttention',
               'def torch_load', 'class LoRAProjection', 'def clean_summary', 'def capture_rng', 'def restore_rng', 'def metric_config', 'def base_config']:
    execute(prefix)
ns['PHASE1_HEADS'] = {key: 2 for key in ns['MODEL_CLASSES']}
for architecture, model_class in ns['MODEL_CLASSES'].items():
    cfg = {'vocab_size': ns['VOCAB_SIZE_ACTUAL']}
    if architecture == 'decoder_only': cfg.update(n_layer=4, n_embd=16, n_head=2, d_ff=32, max_len=160)
    elif architecture == 'encoder_decoder': cfg.update(n_enc_layer=3, n_dec_layer=3, d_model=16, n_head=2, d_ff=32)
    elif architecture == 'hybrid_prefixlm': cfg.update(n_layer=4, d_model=16, n_head=2, d_ff=32, max_len=160)
    else: cfg.update(n_layer=4, d_model=16, n_head=2, n_kv_head=1, d_ff=32, max_len=160)
    path = ns['PHASE1_DIR'] / ('run_' + architecture); path.mkdir()
    torch.save(model_class(**cfg).state_dict(), path / 'final_model.pt')
ns.update(EPOCHS=1, BATCH_SIZE=2, GRAD_ACCUM_STEPS=3, MAX_SOURCE_TOKENS=24, MAX_TARGET_TOKENS=10,
          INSTRUCTION='Document: ', SUMMARY_PREFIX=' Summary: ', PROMPT_TOKENS=3, RANK=2, ALPHA=4.0,
          BOTTLENECK_DIM=8, VALIDATION_FRACTION=0.05, EPOCH_VALIDATION_LIMIT=0, FINAL_VALIDATION_LIMIT=0,
          EVAL_LOSS_LIMIT=0, SAVE_EVERY_STEPS=1, RUN_TAG='tiny')
ns['GENERATION'].update(num_beams=1, max_new_tokens=6, min_new_tokens=2, no_repeat_ngram_size=2)
# Real BERTScore API with a tiny randomly initialized local model/tokenizer, not a performance metric.
from transformers import BertConfig, BertModel, BertTokenizer
metric = temporary / 'tiny_metric'; metric.mkdir()
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'the', 'court', 'claim', 'appeal', 'legal', 'allowed', 'considered', 'and', '.', 'a', 'statute']
(metric / 'vocab.txt').write_text('\n'.join(vocab))
metric_tok = BertTokenizer(vocab_file=str(metric / 'vocab.txt'), model_max_length=128)
metric_tok.save_pretrained(metric)
BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=128)).save_pretrained(metric)
ns.update(BERTSCORE_MODEL=str(metric), BERTSCORE_LAYERS=2, BERTSCORE_BATCH_SIZE=2)
execute('assert set(FILES)')
print('PASS: two-file schema, test-reference isolation, grouped held-out split, exact-overlap filtering and token caches.', flush=True)

# Native Phase 1 computations and all adaptation sites must agree before adaptation updates.
for name, spec in ns['EXPERIMENTS'].items():
    cfg = ns['base_config'](name); model, meta = ns['make_model'](cfg)
    keys = list(ns['DATA']['train'])[:2]; batch = ns['pack_batch']('train', keys, cfg)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        assert logits.shape[:2] == batch['labels'].shape
        solo = model(ns['pack_batch']('train', keys[:1], cfg))
        torch.testing.assert_close(solo, logits[:1, :solo.shape[1]], atol=3e-5, rtol=3e-5)
    model.train(); logits = model(batch); loss = ns['target_loss'](logits, batch['labels'], 0.1)
    loss.backward(); assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    if spec['method'] == 'prompt':
        assert model.soft_prompt.grad is not None and model.soft_prompt.grad.abs().sum() > 0
        assert not any(p.requires_grad for p in model.backbone.parameters())
    model.eval()
    # Cached decoding must match a complete forward on the exact same context and target prefix.
    single = ns['pack_batch']('train', keys[:1], cfg)
    encoder = model.encode(single['source'], single['source_mask']) if model.architecture == 'encoder_decoder' else None
    ids = single['input_ids']; length = ids.shape[1]; cut = max(int(single['prefix_lengths'][0]), length - 2)
    if cut >= length: cut = length - 1
    with torch.no_grad():
        dense, _ = model.hidden(ids, single['input_mask'], single['prefix_lengths'], encoder=encoder)
        _, cache = model.hidden(ids[:, :cut], single['input_mask'][:, :cut], single['prefix_lengths'], encoder=encoder, use_cache=True)
        for position in range(cut, length):
            keep = torch.ones(1, cache[0][0].shape[2] + 1, dtype=torch.bool)
            step, cache = model.hidden(ids[:, position:position + 1], keep, past=cache, encoder=encoder, use_cache=True)
            shift = len(model.soft_prompt) if model.soft_prompt is not None else 0
            torch.testing.assert_close(step[:, -1], dense[:, position + shift], atol=3e-5, rtol=3e-5)
    print('PASS:', name, 'strict load, masked target gradients, padding and cached/full equivalence.', flush=True)

for architecture, name in [('decoder_only', 'decoder_full'), ('encoder_decoder', 'encoder_full'),
                            ('hybrid_prefixlm', 'hybrid_lora'), ('lightweight', 'lightweight_full')]:
    cfg = ns['base_config'](name); model, _ = ns['make_model'](cfg); model.eval()
    batch = ns['pack_batch']('train', list(ns['DATA']['train'])[:1], cfg)
    with torch.no_grad():
        if architecture == 'encoder_decoder':
            _, expected = model.backbone(batch['source'], batch['source_mask'], batch['input_ids'])
        elif architecture == 'hybrid_prefixlm':
            _, expected = model.backbone(batch['input_ids'], int(batch['prefix_lengths'][0]))
        else:
            _, expected = model.backbone(batch['input_ids'])
        indices = batch['target_positions'][:, :, None].expand(-1, -1, expected.shape[-1])
        torch.testing.assert_close(model(batch), expected.gather(1, indices), atol=3e-5, rtol=3e-5)
print('PASS: wrapper logits match all four original Phase 1 architectures before adaptation.', flush=True)

cfg = ns['base_config']('encoder_full', {'encoder_chunk_size': 8}); model, _ = ns['make_model'](cfg)
batch = ns['pack_batch']('train', list(ns['DATA']['train'])[:2], cfg); ns['target_loss'](model(batch), batch['labels']).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
for policy in ['head', 'tail', 'head_tail', 'spread', 'sentence_mmr']:
    selected, coverage = ns['select_source']('The court allowed. An appeal was heard.', list(range(100)), 17, policy)
    assert 0 < len(selected) <= 17 and 0 <= coverage <= 1
labels = torch.tensor([[1, -100]])
s = torch.randn(1, 2, 9, requires_grad=True); t = torch.randn(1, 2, 9)
masked = ns['kd_loss'](s, t, labels, 2.0); masked.backward()
assert s.grad[0, 1].abs().sum() == 0
print('PASS: hierarchical encoder backward, context selection and KD padding mask.', flush=True)

# Confirm metric identity and actual ROUGE/BERTScore API behavior, and no scorer-induced training RNG drift.
keys = list(ns['DATA']['dev']); perfect = [{'id': ns['DATA']['dev'][key]['id'], 'summary': ns['DATA']['dev'][key]['summary']} for key in keys]
rng_state = torch.get_rng_state().clone(); measured = ns['score_summaries'](perfect, 'dev', keys)
assert torch.equal(torch.get_rng_state(), rng_state)
assert measured['rouge1_f1'] == 1.0 and measured['rougeL_f1'] == 1.0 and measured['bertscore_f1'] > 0.999
full = ns['run_experiment']('encoder_full'); assert full['training_complete'] and full['full_validation']
again = ns['run_experiment']('encoder_full'); assert again['optimizer_steps'] == full['optimizer_steps']
for name in ['decoder_lora', 'hybrid_lorma', 'encoder_adapters', 'lightweight_prompt', 'lightweight_kd']:
    assert ns['run_experiment'](name)['training_complete']
ns['RUN_TAG'] = 'interrupted'; ns['MAX_OPTIMIZER_STEPS'] = 1
assert not ns['run_experiment']('encoder_full')['training_complete']
state = ns['torch_load'](ns['run_path']('encoder_full') / 'last.pt')
ns['MAX_OPTIMIZER_STEPS'] = None; state['config'] = ns['base_config']('encoder_full')
ns['save_json'](state['config'], ns['run_path']('encoder_full') / 'run_config.json')
ns['save_torch'](state, ns['run_path']('encoder_full') / 'last.pt')
assert ns['run_experiment']('encoder_full')['training_complete']
a = ns['torch_load'](ns['PROJECT_DIR'] / 'encoder_full_tiny' / 'last.pt')
b = ns['torch_load'](ns['run_path']('encoder_full') / 'last.pt')
for key, value in a['model'].items(): torch.testing.assert_close(value, b['model'][key], atol=0, rtol=0)
cfg = ns['base_config']('encoder_full'); model, _ = ns['make_model'](cfg); model.eval()
for beams in [1, 3]:
    policy = {**cfg['generation'], 'num_beams': beams}
    generated = ns['generate_rows'](model, 'test', list(ns['DATA']['test']), cfg, policy)
    ns['validate_rows'](generated, 'test', list(ns['DATA']['test']))
policy = {**cfg['generation'], 'long_document_mode': 'map_reduce', 'map_reduce_max_chunks': 2}
assert ns['generate_summary'](model, 'train', list(ns['DATA']['train'])[0], cfg, policy)
ns['DECODING_CANDIDATES'] = [{'name': 'beam2', 'overrides': {'num_beams': 2}}]
ns['RUN_TAG'] = 'tiny'; ns['search_decoding']('encoder_full')
execute('import csv\nrecords')
coverage = json.loads((ns['PROJECT_DIR'] / 'coverage.json').read_text())
assert coverage['all_four_phase1_origins'] and coverage['compulsory_lorma'] and coverage['compulsory_kd']
assert coverage['four_techniques_listed_in_phase6']
print('PASS: real metric APIs, six core training/export runs, exact resume, beam/map-reduce generation, decoding search and requirement coverage.', flush=True)
print('Synthetic validation artifacts:', temporary, flush=True)
