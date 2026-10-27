"""Offline CJPE verification: real Phase 1 weights and synthetic complete pipelines."""
import ast,json,os,math,tempfile,itertools,csv,io
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg'); os.environ.setdefault('MPLCONFIGDIR','/tmp/cs781-phase4-mpl')
ROOT=Path(__file__).resolve().parents[1]
notebook=json.loads((ROOT/'Phase4'/'CS781_Phase4_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(notebook))
ns={'__name__':'phase4_offline_tests'}
def text(cell): return ''.join(cell['source'])
def get(prefix): return next(text(c) for c in notebook['cells'] if c['cell_type']=='code' and text(c).startswith(prefix))
def execute(prefix): exec(compile(get(prefix),prefix,'exec'),ns)
for i,c in enumerate(notebook['cells']):
    if c['cell_type']=='code' and not text(c).startswith('%'): compile(text(c),f'cell {i}','exec')
setup=ast.parse(get('import os, json'))
selected=[x for x in setup.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
exec(compile(ast.Module(body=selected,type_ignores=[]),'setup','exec'),ns)
ns.update(BASE_DIR=ROOT,PHASE1_DIR=ROOT,PROJECT_DIR=Path(tempfile.mkdtemp(prefix='cs781-phase4-tests-')),
          DEVICE=ns['torch'].device('cpu'),AMP_DTYPE=ns['torch'].float16,IN_COLAB=False)
ns['torch'].set_num_threads(2)
execute('SEED =')
# Offline provider exercises the actual three-file load/schema/audit cell.
mock=ns['PROJECT_DIR']/'mock'; mock.mkdir(); paths={}
for split in ['train','dev','test']:
    records=[]
    count=8 if split=='train' else 4
    for i in range(count):
        record={'id':f'{split}_{i}','text':f'The {split} court considers case {i}. '+('The appeal concerns the statute and jurisdiction. '*12)}
        if split!='test': record['label']=i%2
        # Reproduce multi-instance labeled data: same ID, distinct text/labels.
        if split!='test' and i==1: record['id']=f'{split}_0'
        if split=='test' and i==0: record['text']=''
        if i==1: record['text']=[record['text'],'The final paragraph states the decision.']
        records.append(record)
    filename=f'Phase_4_CJPE/{"Test" if split=="test" else "Train"}/{split}.json'
    path=mock/f'{split}.json'
    if split=='dev':
        # A .json suffix does not guarantee JSON contents.
        buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=['id','text','label'])
        writer.writeheader()
        for record in records:
            row=dict(record)
            if isinstance(row['text'],list): row['text']='\n'.join(row['text'])
            writer.writerow(row)
        path.write_text(buffer.getvalue())
    elif split=='test':
        path.write_text('\n'.join(json.dumps(row) for row in records))
    else: path.write_text(json.dumps(records))
    paths[filename]=str(path)
ns['HF_TOKEN']='hf_offline_test'; ns['hf_hub_download']=lambda filename,**kw: paths[filename]
execute('FILES =')
assert all(d['label'] is None for d in ns['DATA']['test'].values())
report=json.loads((ns['PROJECT_DIR']/'dataset_loading_report.json').read_text())
assert [report[s][0]['format'] for s in ['train','dev','test']]==['json','csv','jsonl']
# Isolate parsing failures and single forced-download recovery without network access.
fixture=mock/'parser_fixture.json'
row={'id':7,'text':'Quoted "court", Unicode न्याय\nsecond line','label':1}
for contents,encoding,expected in [(json.dumps([row]),'utf-8-sig','json'),
                                    (json.dumps([row]),'utf-16','json'),
                                    (repr([row]),'utf-8','python_literal')]:
    fixture.write_text(contents,encoding=encoding)
    parsed,fmt=ns['parse_case_file'](fixture)
    assert parsed==[row] and fmt==expected
for contents in ['', ' \n ', '<!DOCTYPE html><html>Unauthorized</html>',
                 'version https://git-lfs.github.com/spec/v1\noid sha256:abc',
                 '[{"id": 1,', '{"id":1}\nTHIS IS BROKEN',
                 'id,text,label\nx,good,1\ny,missing',
                 '__import__("os").system("false")']:
    fixture.write_text(contents)
    try: ns['parse_case_file'](fixture)
    except ValueError: pass
    else: raise AssertionError('Malformed case data accepted: '+repr(contents))
provider=ns['hf_hub_download']; original=ns['local_files']['train']
fixture.write_text('')
ns['local_files']['train']=fixture; calls=[]
def recovered_provider(**kwargs):
    calls.append(kwargs)
    return original
ns['hf_hub_download']=recovered_provider
assert len(ns['load_case_split']('train'))==8
assert len(calls)==1 and calls[0]['force_download'] is True
fixture.write_text('<html>Unavailable</html>'); ns['local_files']['train']=fixture
ns['hf_hub_download']=lambda **kwargs:fixture
try: ns['load_case_split']('train')
except ValueError as exc:
    assert 'train (Phase_4_CJPE/Train/train.json)' in str(exc) and 'after a fresh download' in str(exc)
else: raise AssertionError('Repeated invalid downloads accepted')
ns['local_files']['train']=original; ns['hf_hub_download']=provider
ns['load_case_split']('train')
print('PASS: JSON/BOM/UTF-16, JSONL, CSV, safe literals, malformed-file rejection and one forced-download recovery.',flush=True)
execute('TOKENIZER_DIR=')
for prefix in ['class RMSNorm','class CausalSelfAttention','class RelativePositionBias','class RoPEAttention','class GQAAttention',
               'def torch_load','class CaseFeatures','class LowRankMultiplicative','def binary_metrics',
               'def capture_rng','def restore_rng','def base_config','# SignatureOptionalNotPresent']:
    if prefix!='# SignatureOptionalNotPresent': execute(prefix)
execute('assert binary_metrics')
execute('def optimizer_groups')
# Extract ensemble functions without running the result/report display loop.
tree=ast.parse(get('def load_probability_file'))
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'ensemble helpers','exec'),ns)
np=ns['np']; torch=ns['torch']
print('PASS: notebook syntax/schema, official-file paths, split audit, tokenizer cache, preflight checks.',flush=True)
# Schema rejects booleans/unknown labels; preserves original integer IDs.
assert ns['normalize_cases']([{'id':7,'text':'x','label':'1'}],True)['7']['id']==7
duplicates=[{'id':'2019_890','text':'first appeal','label':0},
            {'id':'2019_890','text':'second appeal','label':1},
            {'id':'train:row:0','text':'literal row-like ID','label':0}]
for split in ['train','dev']:
    docs=ns['normalize_cases'](duplicates,True,split=split)
    assert len(docs)==3 and [d['id'] for d in docs.values()]==['2019_890','2019_890','train:row:0']
    assert [d['label'] for d in docs.values()]==[0,1,0]
    predictions=[{'id':d['id'],'pred':d['label']} for d in docs.values()]
    matched=ns['validate_submission'](json.loads(json.dumps(predictions)),docs)
    assert list(matched.values())==[0,1,0]
    directory=ns['PROJECT_DIR']/('duplicate_check_'+split); directory.mkdir()
    original=ns['DATA'][split]; ns['DATA'][split]=docs
    try:
        ns['save_probabilities'](directory,split,list(docs),[.1,.9,.2])
        np.testing.assert_array_equal(ns['load_probability_file'](directory,split),[.1,.9,.2])
    finally: ns['DATA'][split]=original
try: ns['normalize_cases'](duplicates,False,split='test')
except ValueError as exc: assert 'duplicate' in str(exc)
else: raise AssertionError('Ambiguous test IDs accepted')
audit=json.loads((ns['PROJECT_DIR']/'case_instance_audit.json').read_text())
assert audit['train']['instances']==8 and audit['train']['extra_instances']==1
assert audit['dev']['instances']==4 and audit['dev']['extra_instances']==1
print('PASS: repeated train/dev IDs retain all texts/labels; probability and prediction round trips preserve each instance.',flush=True)
for value in [True,2,'yes',None]:
    try: ns['normalize_cases']([{'id':'x','text':'x','label':value}],True)
    except ValueError: pass
    else: raise AssertionError('Invalid binary label accepted')
# Efficient exact threshold search must agree with brute force, including ties and all-one/all-zero cases.
rng=np.random.default_rng(12)
for n in range(1,12):
    for _ in range(15):
        p=np.round(rng.uniform(0,1,n),2); y=rng.integers(0,2,n)
        threshold,metrics,baseline=ns['optimal_threshold'](p,y)
        candidates=[0.5,float(np.nextafter(max(p),np.inf)),*p.tolist()]
        best=max(ns['binary_metrics'](y,(p>=t).astype(int))['macro_f1'] for t in candidates)
        assert math.isclose(best,metrics['macro_f1'],abs_tol=1e-12)
# Independent set-based F1 matches the confusion-matrix formulas.
gold=[0,0,0,1,1]; pred=[0,1,0,1,0]
m=ns['binary_metrics'](gold,pred)
assert m['confusion_matrix']==[[2,1],[1,1]]
assert math.isclose(m['macro_f1'],(4/6+2/4)/2)
# Sampling covers endpoints, all-range union is exact, random sampling is resume-deterministic.
for n in [0,1,29,30,31,512,10000]:
    ranges=ns['window_ranges'](n,32,6)
    assert ns['covered_tokens'](ranges)==n
    for policy in ['head','tail','head_tail','spread','random_mix','all']:
        a=ns['select_ranges'](ranges,4,policy,'id',3)
        b=ns['select_ranges'](ranges,4,policy,'id',3)
        assert a==b and a==sorted(a) and len(set(a))==len(a)
# Nonzero LoRMA update agrees with (I+sBA)W, with frozen base weight/bias.
base=torch.nn.Linear(7,11); wrapper=ns['LowRankMultiplicative'](ns['copy'].deepcopy(base),3,6)
with torch.no_grad():wrapper.B.weight.normal_(std=0.05)
x=torch.randn(2,4,7); weight=base.weight+wrapper.scale*(wrapper.B.weight@wrapper.A.weight)@base.weight
torch.testing.assert_close(wrapper(x),ns['F'].linear(x,weight,base.bias),atol=1e-5,rtol=1e-5)
print('PASS: exact threshold search vs brute force, metric formulas, chunk policies, LoRMA formula.',flush=True)
# Six core real checkpoint tests, including gradients and case-batch padding independence.
for name in ns['CORE_EXPERIMENTS']:
    cfg=ns['base_config'](name,{'chunk_length':32,'chunk_overlap':6,'max_chunks':2,'chunk_batch_size':2,'doc_hidden':16})
    model,meta=ns['make_model'](cfg)
    rows=ns['CaseDataset']('train',cfg)
    one=ns['collate_cases']([rows[0]])
    model.eval()
    with torch.no_grad():
        solo=ns['forward_batch'](model,one)
        grouped=ns['forward_batch'](model,ns['collate_cases']([rows[0],rows[1]]))[:1]
        torch.testing.assert_close(solo,grouped,atol=3e-5,rtol=3e-5)
    model.train(); logits=ns['forward_batch'](model,one)
    loss=ns['F'].cross_entropy(logits,one['labels']); loss.backward()
    assert torch.isfinite(loss) and model.classifier[-1].weight.grad is not None
    if cfg['spec']['method']=='lorma':
        modules=[m for m in model.modules() if isinstance(m,ns['LowRankMultiplicative'])]
        assert any(m.B.weight.grad.abs().sum()>0 for m in modules)
    if cfg['spec']['method']=='adapters':
        modules=[m for m in model.modules() if isinstance(m,ns['ResidualAdapter'])]
        assert any(m.up.weight.grad.abs().sum()>0 for m in modules)
    print(name,'real Phase 1 load, finite forward/backward, padding invariance PASS',flush=True)
    del model,logits,loss;ns['gc'].collect()
# Small models exercise all optional paths and entire training/checkpoint/export workflows.
def tiny_loader(architecture):
    cfg={'vocab_size':ns['VOCAB_SIZE_ACTUAL']}
    if architecture=='decoder_only':cfg.update(n_layer=6,n_embd=16,n_head=2,d_ff=32,max_len=384)
    elif architecture=='encoder_decoder':cfg.update(n_enc_layer=3,n_dec_layer=2,d_model=16,n_head=2,d_ff=32)
    elif architecture=='hybrid_prefixlm':cfg.update(n_layer=3,d_model=16,n_head=2,d_ff=32,max_len=384)
    else:cfg.update(n_layer=4,d_model=16,n_head=2,n_kv_head=1,d_ff=32,max_len=384)
    b=ns['MODEL_CLASSES'][architecture](**cfg); count=sum(p.numel() for p in b.parameters()); del b.head
    if architecture=='encoder_decoder':del b.dec_layers,b.dec_rel_bias,b.ln_dec_f
    return b,cfg,count
ns['load_phase1']=tiny_loader
ns.update(EPOCHS=1,BATCH_SIZE=2,GRAD_ACCUM_STEPS=3,CHUNK_LENGTH=32,CHUNK_OVERLAP=6,MAX_CHUNKS=2,
          CHUNK_BATCH_SIZE=2,DOC_HIDDEN=16,SAVE_EVERY_STEPS=1,RUN_TAG='tiny')
for name in ns['EXTRA_EXPERIMENTS']:
    cfg=ns['base_config'](name)
    model,_=ns['make_model'](cfg)
    if cfg['spec']['method']=='gradual':
        before=sum(p.numel() for p in model.parameters() if p.requires_grad)
        ns['unfreeze_last_layers'](model,2)
        after=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert after>before
    batch=ns['collate_cases']([ns['CaseDataset']('train',cfg)[0]])
    loss=ns['F'].cross_entropy(ns['forward_batch'](model,batch),batch['labels']);loss.backward()
    assert torch.isfinite(loss)
for pooling in ['attention','mean','max','mean_max','transformer']:
    cfg=ns['base_config']('decoder_full',{'pooling':pooling})
    model,_=ns['make_model'](cfg)
    batch=ns['collate_cases']([ns['CaseDataset']('train',cfg)[0]])
    output=ns['forward_batch'](model,batch)
    assert output.shape==(1,2) and torch.isfinite(output).all()
    output.sum().backward()
# Coverage audit and short final accumulation group (4 microbatches / groups of 3).
full=ns['run_experiment']('decoder_full')
assert full['training_complete'] and full['optimizer_steps']==2
resumed=ns['run_experiment']('decoder_full')
assert resumed['optimizer_steps']==2
# A cross-tag KD student with a DIFFERENT chunk budget reuses the same teacher cache.
ns.update(RUN_TAG='student_other_tag',MAX_CHUNKS=3)
kd=ns['run_experiment']('lightweight_kd')
assert kd['training_complete'] and kd['teacher_run'].endswith('decoder_full_tiny')
assert (ns['PROJECT_DIR']/'decoder_full_tiny'/'teacher_train_logits.pt').is_file()
# Warm and cold teacher caches must produce exactly the same student training.
ns['RUN_TAG']='student_repeat_tag'
repeat=ns['run_experiment']('lightweight_kd')
cold=ns['torch_load'](ns['PROJECT_DIR']/'lightweight_kd_student_other_tag'/'last.pt')
warm=ns['torch_load'](ns['run_path']('lightweight_kd')/'last.pt')
for key,value in cold['model'].items():torch.testing.assert_close(value,warm['model'][key],atol=0,rtol=0)
# A second student tuning trial must reuse the cache without reloading teacher weights.
mtime=(ns['PROJECT_DIR']/'decoder_full_tiny'/'teacher_train_logits.pt').stat().st_mtime_ns
ns['run_experiment']('lightweight_kd',{'lr':2e-5},'kd_lr_test')
assert (ns['PROJECT_DIR']/'decoder_full_tiny'/'teacher_train_logits.pt').stat().st_mtime_ns==mtime
# Mid-epoch interruption must exactly match uninterrupted final weights, including dropout.
ns.update(RUN_TAG='interrupted',MAX_CHUNKS=2,MAX_OPTIMIZER_STEPS=1)
capped=ns['run_experiment']('decoder_full'); assert not capped['training_complete']
state=ns['torch_load'](ns['run_path']('decoder_full')/'last.pt')
assert state['epoch']==0 and state['next_batch']==3
ns['MAX_OPTIMIZER_STEPS']=None
cfg=ns['base_config']('decoder_full');state['run_config']=cfg
ns['save_json'](cfg,ns['run_path']('decoder_full')/'run_config.json')
ns['save_torch'](state,ns['run_path']('decoder_full')/'last.pt')
continued=ns['run_experiment']('decoder_full');assert continued['optimizer_steps']==2
final=ns['torch_load'](ns['run_path']('decoder_full')/'last.pt')
baseline=ns['torch_load'](ns['PROJECT_DIR']/'decoder_full_tiny'/'last.pt')
for key,value in final['model'].items():torch.testing.assert_close(value,baseline['model'][key],atol=0,rtol=0)
# Gradual-unfreezing optimizer must contain still-frozen future parameters.
gradual=ns['run_experiment']('decoder_gradual')
assert gradual['trainable_parameters_final']>gradual['trainable_parameters_initial']
# Early stopping persists completion across restarts.
ns.update(RUN_TAG='early',EPOCHS=6,EARLY_STOPPING_PATIENCE=1)
old_lr=ns['EXPERIMENTS']['decoder_full']['lr'];ns['EXPERIMENTS']['decoder_full']['lr']=0.0
stopped=ns['run_experiment']('decoder_full')
assert stopped['training_complete'] and stopped['optimizer_steps']==4 and stopped['best_epoch']==1
again=ns['run_experiment']('decoder_full'); assert again['optimizer_steps']==4
ns['EXPERIMENTS']['decoder_full']['lr']=old_lr
# Saved probability blending must not underperform its best single-model dev baseline.
ensemble=ns['build_ensemble']();assert ensemble is not None
# Intentionally malformed outputs must be rejected, even with correct IDs/count.
bad_rows=[{'id':d['id'],'pred':0} for d in ns['DATA']['test'].values()]
bad_rows[0]['pred']=True
try:ns['validate_submission'](bad_rows,ns['DATA']['test'])
except ValueError:pass
else:raise AssertionError('Invalid submission accepted')
execute('summary=[]')
print('PASS: all six core real checkpoints, six optional approaches, all five pooling paths, training/resume/KD/cache, threshold/ensemble, reports and ZIPs.',flush=True)
