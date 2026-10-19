"""Offline verification with real Phase 1 checkpoints and tiny synthetic RR training."""
import ast,json,os,itertools,tempfile
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/cs781-phase3-mpl')
ROOT=Path(__file__).resolve().parents[1]
n=json.loads((ROOT/'Phase3'/'CS781_Phase3_Solution_Colab.ipynb').read_text())
import nbformat
nbformat.validate(nbformat.from_dict(n))
ns={'__name__':'phase3_offline_validation'}
source=lambda c:''.join(c['source'])
def cell(prefix):
    return next(c for c in n['cells'] if c['cell_type']=='code' and source(c).startswith(prefix))
def execute(prefix):exec(compile(source(cell(prefix)),prefix,'exec'),ns)
for i,c in enumerate(n['cells']):
    if c['cell_type']=='code' and not source(c).startswith('%'):compile(source(c),f'cell {i}','exec')
setup=ast.parse(source(cell('import os, json')))
selected=[x for x in setup.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
exec(compile(ast.Module(body=selected,type_ignores=[]),'setup','exec'),ns)
ns.update(BASE_DIR=ROOT,PHASE1_DIR=ROOT,PROJECT_DIR=Path(tempfile.mkdtemp(prefix='cs781-phase3-verify-')),
          DEVICE=ns['torch'].device('cpu'),AMP_DTYPE=ns['torch'].float16)
ns['torch'].set_num_threads(2)
execute('SEED =')
# Extract pure schema helpers without downloading/authenticating the protected dataset.
data_cell=ast.parse(source(cell('FILES =')))
selected=[x for x in data_cell.body if isinstance(x,ast.FunctionDef)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'schema helpers','exec'),ns)
expected=next(x for x in data_cell.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED_LABELS' for t in x.targets))
labels=ast.literal_eval(expected.value)
ns.update(EXPECTED_LABELS=labels,LABELS=labels,LABEL2ID={v:i for i,v in enumerate(labels)},
          ID2LABEL=dict(enumerate(labels)),NUM_LABELS=len(labels),DOMAIN2ID={'CL':0,'IT':1},DATA_FINGERPRINT='synthetic')
for vocab in [labels,{l:i for i,l in enumerate(labels)},{str(i):l for i,l in enumerate(labels)},{'label2id':{l:i for i,l in enumerate(labels)}}]:
    assert ns['parse_label_vocab'](vocab)==labels
DATA={s:{} for s in ['train','dev','test']}
for domain in ['CL','IT']:
    for split in DATA:
        texts=[f'{domain} {split} the petitioner filed an appeal.',f'{domain} {split} the issue is jurisdiction.',
               f'{domain} {split} section 12 applies.',f'{domain} {split} the appeal is dismissed.']
        targets=['Fact','Issue','Statute','RulingByPresentCourt']
        doc={'id':domain+'_'+split,'text':texts}
        if split!='test':doc['labels']=targets
        else:doc['text'][2]=''  # Empty sentences still require a prediction.
        DATA[split].update(ns['normalize_rr']([doc],domain,split!='test'))
ns['DATA']=DATA
# Execute the real download/schema/split audit cell with an offline seven-file provider.
mock_dir=ns['PROJECT_DIR']/'mock_dataset';mock_dir.mkdir()
file_map={}
label_path=mock_dir/'label_vocab.json';label_path.write_text(json.dumps(labels))
file_map['Phase_3_RR/Train/label_vocab.json']=str(label_path)
for domain in ['CL','IT']:
    for split in ['train','dev','test']:
        folder='Test' if split=='test' else 'Train'
        filename=f'Phase_3_RR/{folder}/{domain}_{split}.json'
        path=mock_dir/f'{domain}_{split}.json'
        docs=[d for d in DATA[split].values() if d['domain']==domain]
        path.write_text(json.dumps(docs))
        file_map[filename]=str(path)
ns['HF_TOKEN']='hf_offline_test'
ns['hf_hub_download']=lambda filename,**kwargs:file_map[filename]
execute('FILES =')
assert len(ns['local_files'])==7
assert all(d['labels'] is None for d in ns['DATA']['test'].values())
execute('TOKENIZER_DIR =')
for prefix in ['class RMSNorm','class CausalSelfAttention','class RelativePositionBias','class RoPEAttention','class GQAAttention',
               'def torch_load','class LowRankLinear','def validate_predictions','def fitted_priors',
               '# Tokenization','def capture_rng','def restore_rng','def run_path']:
    execute(prefix)
print('PASS: notebook syntax/format, label formats, target/context tokenization, metrics, adapters.',flush=True)
# Malformed labels/counts must fail.
try:ns['normalize_rr']([{'id':'x','text':['s'],'labels':[]}],'CL',True)
except ValueError:pass
else:raise AssertionError('Invalid label count accepted')
for budget in [1,2,3]:assert len(ns['head_tail'](list(range(100)),budget))==budget
long_doc={'id':'long','domain':'CL','text':['before','target','after'],'labels':['Fact','Issue','Statute']}
long_row=ns['sentence_example'](long_doc,1,[list(range(1000))]*3)
assert len(long_row['input_ids'])==ns['MAX_LENGTH']
assert sum(long_row['target_mask'])==ns['MAX_TARGET_TOKENS'] and long_row['target_truncated']
# Compare train-fitted transition decoder with brute force on a short sequence.
np=ns['np'];torch=ns['torch']
logits=np.full((3,len(labels)),-1000.0,dtype=np.float32)
logits[:,:3]=np.array([[1,2,0],[0,2,1],[2,0,1]])
config={'transition_strength':0.3,'prior_correction':0.25}
prior=ns['PRIORS']['CL']
v=logits.astype(np.float64);v-=v.max(1,keepdims=True);em=v-np.log(np.exp(v).sum(1,keepdims=True))-config['prior_correction']*prior['class']
def score(p):return sum(em[i,t] for i,t in enumerate(p))+config['transition_strength']*(prior['start'][p[0]]+sum(prior['transition'][a,b] for a,b in zip(p,p[1:])))
expected=max(itertools.product(range(3),repeat=3),key=score)
assert ns['sequence_tags'](logits,prior,config)==list(expected)
# Check Low-rank multiplicative formula after nonzero updates, not only identity initialization.
for method in ['lora','rslora','lorma']:
    base=torch.nn.Linear(7,11)
    wrapped=ns['LowRankLinear'](ns['copy'].deepcopy(base),method,rank=3,alpha=5)
    with torch.no_grad():wrapped.B.weight.normal_(std=0.1)
    delta=wrapped.B.weight@wrapped.A.weight
    weight=base.weight+wrapped.scale*delta if method!='lorma' else base.weight+wrapped.scale*delta@base.weight
    x=torch.randn(2,4,7)
    torch.testing.assert_close(wrapped(x),ns['F'].linear(x,weight,base.bias),atol=1e-5,rtol=1e-5)
print('PASS: decoder agrees with exhaustive search; LoRA/rsLoRA/LoRMA formulas verified.',flush=True)
# Real saved checkpoints: six forward/backward adaptation paths plus padding invariance.
for name,spec in ns['EXPERIMENTS'].items():
    model,meta=ns['make_model'](spec)
    row=ns['SETS']['train'][0]
    batch=ns['collate_rr']([row])
    model.eval()
    with torch.no_grad():
        solo=ns['forward_batch'](model,batch)
        other=dict(row,input_ids=row['input_ids']+[5,6,7],target_mask=row['target_mask']+[False]*3)
        grouped=ns['forward_batch'](model,ns['collate_rr']([row,other]))[:1]
        torch.testing.assert_close(solo,grouped,atol=3e-5,rtol=3e-5)
    model.train();logits=ns['forward_batch'](model,batch)
    loss=ns['F'].cross_entropy(logits,batch['labels']);loss.backward()
    assert torch.isfinite(loss) and model.classifier[-1].weight.grad is not None
    if spec['method'] in ['lora','rslora','lorma']:
        adapters=[m for m in model.modules() if isinstance(m,ns['LowRankLinear'])]
        assert any(m.B.weight.grad.abs().sum()>0 for m in adapters)
    if spec['method']=='ia3':
        adapters=[m for m in model.modules() if isinstance(m,ns['IA3Linear'])]
        assert any(m.scale.grad.abs().sum()>0 for m in adapters)
    print(name,'real checkpoint load + forward/backward + padding invariance PASS',flush=True)
    del model,loss,logits;ns['gc'].collect()
# Fast synthetic end-to-end tests preserve model/training logic and replace only checkpoint loader.
def tiny_loader(architecture):
    config={'vocab_size':ns['VOCAB_SIZE_ACTUAL']}
    if architecture=='decoder_only':config.update(n_layer=6,n_embd=16,n_head=2,d_ff=32,max_len=384)
    elif architecture=='encoder_decoder':config.update(n_enc_layer=3,n_dec_layer=2,d_model=16,n_head=2,d_ff=32)
    elif architecture=='hybrid_prefixlm':config.update(n_layer=3,d_model=16,n_head=2,d_ff=32,max_len=384)
    else:config.update(n_layer=4,d_model=16,n_head=2,n_kv_head=1,d_ff=32,max_len=384)
    b=ns['MODEL_CLASSES'][architecture](**config);total=sum(p.numel() for p in b.parameters())
    del b.head
    if architecture=='encoder_decoder':del b.dec_layers,b.dec_rel_bias,b.ln_dec_f
    return b,config,total
ns['load_phase1']=tiny_loader
ns.update(EPOCHS=1,BATCH_SIZE=2,GRAD_ACCUM_STEPS=3,SAVE_EVERY_STEPS=1,RUN_TAG='tiny',TEACHER_EXPERIMENT='decoder_full')
full=ns['run_experiment']('decoder_full')
assert full['training_complete'] and full['optimizer_steps']==2
again=ns['run_experiment']('decoder_full')
assert again['optimizer_steps']==2
kd=ns['run_experiment']('lightweight_kd')
assert kd['teacher_experiment']=='decoder_full' and kd['training_complete']
# Resume from a true mid-epoch optimizer boundary; final short group is one microbatch.
ns.update(RUN_TAG='interrupted',MAX_OPTIMIZER_STEPS=1)
capped=ns['run_experiment']('decoder_full')
assert not capped['training_complete']
state=ns['torch_load'](ns['run_path']('decoder_full')/'last.pt')
assert state['next_batch']==3 and state['epoch']==0
ns['MAX_OPTIMIZER_STEPS']=None
config=ns['run_config']('decoder_full');state['run_config']=config
ns['save_json'](config,ns['run_path']('decoder_full')/'run_config.json')
ns['save_torch'](state,ns['run_path']('decoder_full')/'last.pt')
continued=ns['run_experiment']('decoder_full')
assert continued['training_complete'] and continued['optimizer_steps']==2
final_state=ns['torch_load'](ns['run_path']('decoder_full')/'last.pt')
uninterrupted=ns['torch_load'](ns['PROJECT_DIR']/'decoder_full_tiny'/'last.pt')
for name,weight in final_state['model'].items():torch.testing.assert_close(weight,uninterrupted['model'][name],atol=0,rtol=0)
# Order reconstruction must be invariant to the order examples are presented for prediction.
model,_=ns['make_model'](ns['EXPERIMENTS']['decoder_full'])
model.load_state_dict(final_state['model'],strict=True)
before=ns['predict_logits'](model,'test')
ns['SETS']['test'].rows.reverse()
after=ns['predict_logits'](model,'test')
for key in before:np.testing.assert_allclose(before[key],after[key],atol=1e-5,rtol=1e-5)
ns['SETS']['test'].rows.reverse()
# Early stopping completion must persist across runtime restarts.
ns.update(EPOCHS=6,EARLY_STOPPING_PATIENCE=1,RUN_TAG='early_stop')
old_lr=ns['EXPERIMENTS']['decoder_full']['lr']
ns['EXPERIMENTS']['decoder_full']['lr']=0.0
stopped=ns['run_experiment']('decoder_full')
assert stopped['training_complete'] and stopped['optimizer_steps']==4 and stopped['best_epoch']==1
stopped_again=ns['run_experiment']('decoder_full')
assert stopped_again['optimizer_steps']==4
ns['EXPERIMENTS']['decoder_full']['lr']=old_lr
execute('summary=[]')
print('PASS: training/dev/test, KD, completed reuse, deterministic interruption/resume, ordering, reports and submission ZIP.',flush=True)
