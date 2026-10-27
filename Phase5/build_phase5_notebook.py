"""Assemble a standalone LSI Colab notebook from tested Phase 1/4 components.

The resulting notebook contains all source; no local Phase4/Phase5 imports are needed in Colab.
The Phase 4 notebook is read only and is never regenerated or overwritten.
"""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
template = json.loads((ROOT / 'Phase4' / 'CS781_Phase4_Solution_Colab.ipynb').read_text())


def source(prefix):
    return next(''.join(c['source']) for c in template['cells']
                if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))


def replace(s, old, new):
    if old not in s:
        raise ValueError('Phase 4 template changed; missing anchor: ' + old[:120])
    return s.replace(old, new)


cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip().splitlines(True)})


def code(s):
    s = s.strip()
    if not s.startswith('%'):
        ast.parse(s)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(True),
                  'execution_count': None, 'outputs': []})


md('''# CS781 Phase 5 — Legal Statute Identification (LSI)

Phase 1 legal backbones → long-document **multi-label** statute identification.
Follows §11 of `Capstone Document_edited.pdf` and the professor's starter workflow:
authenticate → configure → load official data → train/adapt → evaluate → reload/verify → ZIP.
Only your Phase 1 weights, tokenizer and supplied competition data are used.

**Train fits models; dev selects checkpoints/thresholds/hyperparameters; test produces predictions.**
Outputs save automatically under **`MyDrive/cs781_phase5/`**.
Primary: **Macro-F1**. Secondary: **Micro-F1 and Subset Accuracy**.
Predictions are a list of `{"id": original_id, "pred": [integer_statute_ids]}` objects.

Six core runs cover all four Phase 1 origins, full/partial/adapters/compulsory LoRMA and compulsory KD.
Seven optional profiles and thirteen bounded tuning trials broaden comparisons. More trials consume
more GPU time and may overfit dev; no leaderboard improvement is guaranteed.
''')
md('## 1. Install dependencies')
code(source('%pip'))
md('## 2. Mount Drive, authenticate with an HF token, and check GPU')
setup = source('import os, json').replace('cs781_phase4', 'cs781_phase5').replace('Phase 4 output', 'Phase 5 output')
# Separate runtimes must not share a fixed atomic temporary filename.
setup = setup.replace("temporary = path.with_name(path.name + '.tmp')",
                      "temporary = path.with_name(path.name + '.' + __import__('uuid').uuid4().hex + '.tmp')")
code(setup)
md('''## 3. All experiment, hyperparameter and calibration controls — edit here

First verify `decoder_full` with `RUN_ALL=False`. Then use `RUN_ALL=True` for the six core profiles.
For independent Colab runtimes choose one `EXPERIMENT`, leave all sweep flags false, and keep
`BUILD_ENSEMBLE=False`. Finish Section 5 in one runtime before others prepare shared caches.
Run `lightweight_kd` **after a larger completed Phase 5 teacher exists**. A Phase 4 teacher is not valid.
Use a new `RUN_TAG` whenever changing training settings. Matching tag/settings resume automatically.
Calibration is automatic on dev and does not retrain models within an epoch.
''')
code((HERE / 'controls.py').read_text())
md('''## 4. Download all official splits, validate formats and build the statute catalog

Supports JSON/JSONL, explicit-column CSV and safe literal exports despite `.json` filenames.
One forced fresh download follows a parsing failure. `dataset_loading_report.json` names its split.
Train/dev repeated IDs retain all instances; test IDs/types remain unchanged and must be unique.
Exact nonempty training copies of dev/test documents are excluded under the default
`SPLIT_OVERLAP_POLICY='exclude_train'`. Every dev/test case remains. Counts and ID/hash provenance
save to `split_audit.json` and `excluded_train_instances.json`; weights/priors are recomputed afterward.
The label catalog comes from train+dev (or `STATUTE_IDS`), never private test labels. IDs can have gaps.
Labels absent from train are reported rather than assigned invented training targets.
''')
data_prefix = source('FILES =').split('def case_key')[0]
data_prefix = data_prefix.replace('Phase_4_CJPE', 'Phase_5_LSI').replace('CJPE', 'LSI')
code(data_prefix + '\n' + (HERE / 'data.py').read_text())
md('''## 5. Shared Phase 1 BPE, Drive token caches and overlapping document windows

No external tokenizer is downloaded. Default: four spread windows including the beginning/end,
not full document coverage. `chunk_coverage.json` reports actual coverage. Compare `MAX_CHUNKS=8`
or `0` (all windows), head/tail selection, random training windows and hierarchical pooling.
Each case still receives exactly one multi-label prediction. Empty texts receive an UNK chunk.
''')
tokens = source('TOKENIZER_DIR=').replace('cs781_phase4_token_cache', 'cs781_phase5_token_cache')
tokens = replace(tokens, "'labels':torch.tensor([r['label'] if r['label'] is not None else -100 for r in rows]),",
                 "'labels':torch.tensor(np.stack([encode_labels(r['label']) for r in rows]),dtype=torch.float32),")
# Avoid same-name temp files if sessions are inadvertently started together.
tokens = tokens.replace("temporary=binary_path.with_suffix('.bin.tmp')", "temporary=binary_path.with_name(binary_path.name+'.'+__import__('uuid').uuid4().hex+'.tmp')")
tokens = tokens.replace("temporary=active_path.with_suffix('.tmp')", "temporary=active_path.with_name(active_path.name+'.'+__import__('uuid').uuid4().hex+'.tmp')")
code(tokens)
md('## 6. Strictly checkpoint-compatible Phase 1 architectures — no pretraining rerun')
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention', 'class GQAAttention']:
    code(source(prefix))
md('''## 7. Hierarchical multi-label head and adaptation methods

Encoder–decoder experiments use its encoder as the document backbone. Other backbones retain their
Phase 1 attention behavior. All create a new trainable statute head with independent sigmoid outputs.
KD keeps alternating Phase 1 lightweight layers, giving a smaller Phase 1-initialized student.
LoRMA+ implements `(I + alpha/r BA)W`; adapters use one post-block residual bottleneck.
''')
code(source('def torch_load'))
code(source('class CaseFeatures'))
model = replace(source('class LowRankMultiplicative'), 'nn.Linear(hidden,2))', 'nn.Linear(hidden,cfg[\'num_labels\']))')
# Optional train-only label co-occurrence propagation; no dev labels enter this graph.
model = replace(model, '    def forward(self,input_ids,attention_mask,target_mask,chunk_positions,owners,stats):',
'''        train_y=torch.tensor(gold_matrix('train'),dtype=torch.float32)
        co=train_y.T@train_y
        co.fill_diagonal_(0)
        graph=co/co.sum(0,keepdim=True).clamp_min(1)
        self.register_buffer('label_graph',graph if cfg['label_graph_weight'] else torch.zeros_like(graph))
    def forward(self,input_ids,attention_mask,target_mask,chunk_positions,owners,stats):''')
model = replace(model, '        return self.classifier(torch.stack(documents))',
'''        logits=self.classifier(torch.stack(documents))
        return logits+self.cfg['label_graph_weight']*(torch.sigmoid(logits)@self.label_graph)''')
model = replace(model, '    model=LegalCase(features,cfg)', '''    model=LegalCase(features,cfg)
    if cfg['initialize_label_priors']:
        priors=torch.tensor(np.clip(TRAIN_COUNTS/max(1,len(DATA['train'])),0.005,0.995),dtype=torch.float32)
        with torch.no_grad(): model.classifier[-1].bias.copy_(torch.logit(priors))''')
code(model)
md('''## 8. Multi-label metrics, global/per-label thresholds and output verification

Macro-F1 averages **per-statute positive-class F1**; Micro-F1 aggregates all statute decisions;
Subset Accuracy requires the entire predicted label set to match. Zero-denominator F1 is zero.
The PDF does not supply evaluator code and says label dimension is derived from references.
Default `METRIC_LABEL_SCOPE='reference_range'` assumes IDs `0..max(gold ID)`; known-catalog and
reference-present variants are also reported. Match this control to any published official scorer.

Dev compares 0.5, a bounded global grid, and per-label F1 thresholds shrunk toward the global threshold.
Rare labels below `MIN_DEV_POSITIVES` retain the global threshold. Calibration keeps a global/0.5
fallback if per-label changes hurt Macro-F1. Test labels never calibrate any decision.
''')
metrics = (HERE / 'metrics.py').read_text()
# Prediction loop is shared but uses the new multi-label probability conversion.
old_metrics = source('def binary_metrics')
predict = old_metrics[old_metrics.index('@torch.no_grad()'):old_metrics.index('def probability_vector')]
code(metrics + '\n' + predict)
md('## 9. Full configuration fingerprints, larger-teacher cache and deterministic resume')
code(source('def capture_rng'))
code(source('def restore_rng'))
config = source('def base_config').replace('Phase 4 teacher', 'Phase 5 teacher')
config = replace(config, "'data_fingerprint':DATA_FINGERPRINT}", """'data_fingerprint':DATA_FINGERPRINT,
        'num_labels':NUM_LABELS,'label_ids':LABEL_IDS,'metric_label_scope':METRIC_LABEL_SCOPE,
        'loss_name':LOSS_NAME,'pos_weight_cap':POS_WEIGHT_CAP,'focal_gamma':FOCAL_GAMMA,
        'asl_gamma_pos':ASL_GAMMA_POS,'asl_gamma_neg':ASL_GAMMA_NEG,'asl_clip':ASL_CLIP,
        'label_smoothing':LABEL_SMOOTHING,'initialize_label_priors':INITIALIZE_LABEL_PRIORS,
        'label_graph_weight':0.0,'threshold_mode':THRESHOLD_MODE,'threshold_grid':THRESHOLD_GRID,
        'min_dev_positives':MIN_DEV_POSITIVES,'threshold_shrinkage':THRESHOLD_SHRINKAGE,
        'allow_empty':ALLOW_EMPTY_PREDICTIONS}""")
config = replace(config, '    return cfg', '''    assert cfg['loss_name'] in ['bce','focal','asymmetric']
    assert cfg['threshold_mode'] in ['global','per_label','auto']
    assert 0<=cfg['label_smoothing']<1 and cfg['pos_weight_cap']>=1
    assert cfg['threshold_shrinkage']>=0 and cfg['min_dev_positives']>=1
    assert cfg['focal_gamma']>=0 and cfg['asl_gamma_pos']>=0 and cfg['asl_gamma_neg']>=0 and 0<=cfg['asl_clip']<1
    assert cfg['num_labels']==NUM_LABELS and cfg['label_ids']==LABEL_IDS
    assert cfg['metric_label_scope']==METRIC_LABEL_SCOPE
    return cfg''')
config = replace(config, 'logits.shape!=(len(keys),2)', 'logits.shape!=(len(keys),NUM_LABELS)')
code(config)
md('''## 10. Preflight checks: multi-label behavior, thresholds, windows and LoRMA

These checks establish implementation behavior, not expected leaderboard performance.
Full post-training checks independently reload weights and recompute every reported dev metric.
''')
code('''assert NUM_LABELS>0 and all(type(label) is int for label in LABEL_IDS)
y=np.eye(NUM_LABELS,dtype=np.int64)
assert multilabel_metrics(y,y)['micro_f1']==1.0
assert multilabel_metrics(y,y)['subset_accuracy']==1.0
assert multilabel_metrics(y,np.zeros_like(y))['micro_f1']==0.0
p=0.1+0.8*y
policy,selected,baseline=optimal_threshold(p,y)
assert selected['macro_f1']+1e-12>=baseline['macro_f1']
if NUM_LABELS>1:
    assert decisions(np.full((1,NUM_LABELS),0.9),{'thresholds':[0.5]*NUM_LABELS,'allow_empty':True}).sum()==NUM_LABELS
for length in [0,1,500,2000]:
    ranges=window_ranges(length,384,64)
    assert covered_tokens(ranges)==length
base=nn.Linear(7,11); layer=LowRankMultiplicative(copy.deepcopy(base),3,6)
x=torch.randn(2,5,7); torch.testing.assert_close(layer(x),base(x))
layer(x).sum().backward(); assert layer.B.weight.grad.abs().sum()>0 and layer.base.weight.grad is None
print('PASS: multi-label metric invariants, threshold baseline, chunk coverage and LoRMA identity/gradients.')''')
md('''## 11. Train, calibrate on dev, reload the best checkpoint and verify every export

Independent multi-label BCE, focal or asymmetric loss with train-only positive weights.
KD uses temperature-scaled independent Bernoulli targets, not mutually exclusive softmax targets.
The best dev-calibrated checkpoint is reloaded and its full metrics must match. JSON predictions
are decoded back to multi-hot vectors and scored again. Every test ID and ZIP entry is verified.
Progress bars cover tokenization, coverage, epochs, teacher prediction, dev/test and sweeps.
''')
code((HERE / 'losses.py').read_text())
train = source('def optimizer_groups')
start = train.index('def save_probabilities'); stop = train.index('def run_experiment')
train = train[:start] + train[stop:]
start = train.index("    counts=np.bincount"); stop = train.index('    dataset=CaseDataset', start)
train = train[:start] + '''    weights=positive_weights(cfg)
    save_json({'label_ids':LABEL_IDS,'counts':TRAIN_COUNTS.tolist(),
               'weights':weights.tolist() if weights is not None else [1.0]*NUM_LABELS},output/'class_weights.json')
''' + train[stop:]
train = replace(train, "                    per_case=F.cross_entropy(logits.float(),labels,weight=weights,reduction='none')\n                    loss=per_case.mean()  # Do not cancel class weights when BATCH_SIZE=1.",
                "                    loss=multilabel_loss(logits,labels,weights,cfg)")
start = train.index('                        kd=F.kl_div'); stop = train.index("                        loss=(1-cfg['kd_weight'])", start)
train = train[:start] + '                        kd=distillation_loss(logits,teacher_logits,temperature)\n' + train[stop:]
train = replace(train, "gold=[DATA['dev'][key]['label'] for key in keys]", "gold=gold_matrix('dev',keys)")
train = replace(train, "optimal_threshold(p,gold,cfg['tune_threshold'])", "optimal_threshold(p,gold,cfg['tune_threshold'],cfg)")
train = replace(train, 'threshold={threshold:.6f}', 'threshold_strategy={threshold[\'mode\']}')
train = replace(train, "    dev_predictions=[{'id':DATA['dev'][key]['id'],'pred':int(value>=threshold)} for key,value in zip(keys,p)]",
                "    dev_predictions=make_prediction_rows('dev',keys,p,threshold)")
train = replace(train, "    assert binary_metrics(gold,[reloaded_map[key] for key in keys])==metrics",
                "    assert multilabel_metrics(gold,np.stack([encode_labels(reloaded_map[key]) for key in keys]))==metrics")
start = train.index('    errors=['); stop = train.index('    metrics.update', start)
train = train[:start] + '    save_error_analysis(output,keys,p,threshold,metrics)\n' + train[stop:]
train = replace(train, "'rule':'p(class 1) >= threshold'", "'rule':'independent statute probabilities >= per-label thresholds'")
start = train.index('    fig,ax=plt.subplots(); ax.imshow'); stop = train.index('    test_keys,test_logits', start)
train = train[:start] + train[stop:]
train = replace(train, "'integer_binary_predictions':True", "'unique_valid_integer_statute_sets':True")
train = replace(train, "'train_dev_test_separate':True", """'train_dev_test_separate':True,
        'overlap_policy':SPLIT_OVERLAP_POLICY,
        'remaining_train_instances_matching_heldout_text':split_audit['remaining_train_instances_matching_heldout_text']""")
code(train)
md('## 12. Run one experiment, the core set, extras and/or bounded tuning')
code(source('selected=list(CORE_EXPERIMENTS)').replace('CJPE approaches', 'LSI approaches'))
md('''## 13. Optional saved-probability ensemble and export verification

Enable `BUILD_ENSEMBLE` after runs finish. Uses compatible completed runs only, aligns case instances
and label columns, and compares weighted probability pairs to the best single model using dev.
Calibration is repeated on blended probabilities. Test labels are never used. Ensemble is an extra
inference artifact and cannot replace the four architecture-specific adaptation submissions.
''')
ensemble = source('def load_probability_file')
ensemble = ensemble[ensemble.index('def build_ensemble'):]
ensemble = replace(ensemble, "gold=[d['label'] for d in DATA['dev'].values()]", "gold=gold_matrix('dev')")
code(ensemble)
md('## 14. Where to change what in Colab')
md((HERE / 'TUNING.md').read_text())
md('## 15. Compare compatible runs and select the best run per Phase 1 origin')
code((HERE / 'summary.py').read_text())
md('''References and provenance: local `Capstone Document_edited.pdf` §11, professor's
`CS781_Phase1_Starter_Code.ipynb` for workflow, your Phase 1 compatible definitions and Phase 4's
checkpoint/chunking/resume implementation. This notebook contains all runtime code.

LoRMA+ uses identity-plus-low-rank multiplication of frozen attention projection weights.
The residual adapter has one post-block bottleneck. KD matches independent Bernoulli statute targets.
Keep the course-required LLM declaration and base your analysis on measured runs, not synthetic tests.
''')
for index, cell in enumerate(cells):
    cell['id'] = f'phase5-{index:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
             'language_info': {'name': 'python'}, 'colab': {'name': 'CS781_Phase5_Solution_Colab.ipynb', 'provenance': []}},
            'nbformat': 4, 'nbformat_minor': 5}
output = ROOT / 'Phase5' / 'CS781_Phase5_Solution_Colab.ipynb'
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Created', output, 'cells:', len(cells))
