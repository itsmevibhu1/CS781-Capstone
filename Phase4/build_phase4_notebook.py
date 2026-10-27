"""Build a standalone CJPE notebook from compatible Phase 1/3 definitions."""
import ast,json,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p3=json.loads((ROOT/'Phase3'/'CS781_Phase3_Solution_Colab.ipynb').read_text())
def source(prefix):
    return next(''.join(c['source']) for c in p3['cells'] if c['cell_type']=='code' and ''.join(c.get('source',[])).startswith(prefix))
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip().splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(s).strip().splitlines(True)))
md('''
# CS781 Phase 4 — Court Judgment Prediction (CJPE)

From-scratch Phase 1 legal backbones → whole-case binary prediction. This notebook follows the
professor's starter workflow: HF authentication → official data → model/training → dev metrics → saved
weights/predictions → verified `submission.zip`. It is standalone; no Phase 2/3 training is rerun.

It uses all three official files: **train for fitting**, **dev for checkpoint/hyperparameter/threshold
selection**, and **unlabeled test for predictions**. The required output is a **list** of
`{"id": original_case_id, "pred": 0_or_1}` objects, not a Phase 3 document-label mapping.

Six core experiments cover all four Phase 1 origins and full tuning, partial tuning, residual adapters,
LORMA+, BitFit and knowledge distillation. Optional approaches and a bounded tuning sweep broaden
long-document/pooling/adaptation comparisons. A small dev-selected ensemble is optional.

Select a Colab GPU, add an `HF` secret, and run all cells. Phase 1 inputs live in
`MyDrive/cs781_phase1`; new artifacts live in **`MyDrive/cs781_phase4`**. Section 3 contains controls;
section 14 explains exactly how to change them. No external pretrained model or LLM API is used.
Scores must come from actual runs; implementation tests cannot establish a leaderboard score.
''')
md('## 1. Install dependencies')
code(source('%pip'))
md('## 2. Mount Drive, authenticate, and check device')
setup=source('import os, json').replace("'cs781_phase3'","'cs781_phase4'").replace("print('Phase 3 output:'","print('Phase 4 output:'")
code(setup)
md('''
## 3. Experiment and tuning controls — edit here

First run: `EXPERIMENT='decoder_full'`, `RUN_ALL=False`. Then set `RUN_ALL=True` for the six core
methods. `RUN_EXTRA_APPROACHES=True` adds six optional approaches; this is extra compute, not necessary
for minimum coverage. `RUN_TUNING=True` executes the explicit bounded trials below. `BUILD_ENSEMBLE`
uses saved probabilities from compatible completed runs without retraining.

A new `RUN_TAG` starts fresh Phase 1-based experiments. Changing settings under an existing run name
raises a clear error. Resume uses matching configs and restores optimizer/scheduler/scaler/RNG/position.
The KD teacher can be selected from **any compatible completed Phase 4 run**, including another tag;
changing student chunking/LR does not force you to retrain a teacher. Teacher logits are generated on
TRAIN cases only, saved once and reused. Dev labels never enter a student minibatch.
''')
code('''
SEED = 42
EXPERIMENT = 'decoder_full'
RUN_ALL = False
RUN_EXTRA_APPROACHES = False
RUN_TUNING = False
BUILD_ENSEMBLE = True
RUN_TAG = 'v1'
RESUME = True
HF_DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
EPOCHS = 6
BATCH_SIZE = 2                 # Cases per minibatch, not chunks.
GRAD_ACCUM_STEPS = 4
CHUNK_LENGTH = 384             # Includes BOS/EOS; <= Phase 1 learned position limit.
CHUNK_OVERLAP = 64
MAX_CHUNKS = 4                # Per case; 0 = use EVERY sliding window (potentially expensive).
CHUNK_BATCH_SIZE = 4           # Backbone processes this many chunks at once.
TRAIN_CHUNK_POLICY = 'spread'  # spread, head, tail, head_tail, random_mix, all
EVAL_CHUNK_POLICY = 'spread'   # Deterministic inference; random_mix is training-only.
DOC_POOLING = 'attention'     # attention, mean, max, mean_max, transformer
DOC_HIDDEN = 128               # Divisible by 4 for transformer pooling.
HEAD_DROPOUT = 0.15
HEAD_LR_MULTIPLIER = 5.0
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 0.5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CHECKPOINTING = True
ADAPTER_RANK = 8
ADAPTER_ALPHA = 16.0
BOTTLENECK_DIM = 64
PARTIAL_LAYERS = 2
UNFREEZE_LAYERS_PER_EPOCH = 2
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0
TEACHER_RUN_DIR = None  # e.g. str(PROJECT_DIR/'lightweight_lorma_v1'); None = best compatible teacher.
SAVE_EVERY_STEPS = 100
MAX_OPTIMIZER_STEPS = None  # Positive integer for smoke tests; leave None for real training.
NUM_WORKERS = 0
TUNE_THRESHOLD = True        # Exact finite dev threshold search, always includes 0.5 baseline.
PHASE1_HEADS = {'decoder_only':12,'encoder_decoder':8,'hybrid_prefixlm':10,'lightweight':12}
NUM_LABELS = 2

CORE_EXPERIMENTS = {
    'decoder_full': {'architecture':'decoder_only','method':'full','lr':2e-5},
    'encoder_partial': {'architecture':'encoder_decoder','method':'partial','lr':5e-5},
    'hybrid_adapters': {'architecture':'hybrid_prefixlm','method':'adapters','lr':2e-4},
    'lightweight_lorma': {'architecture':'lightweight','method':'lorma','lr':2e-4},
    'decoder_bitfit': {'architecture':'decoder_only','method':'bitfit','lr':2e-4},
    'lightweight_kd': {'architecture':'lightweight','method':'kd','lr':5e-5},
}
EXTRA_EXPERIMENTS = {
    'lightweight_full': {'architecture':'lightweight','method':'full','lr':2e-5},
    'encoder_full': {'architecture':'encoder_decoder','method':'full','lr':2e-5},
    'hybrid_lorma': {'architecture':'hybrid_prefixlm','method':'lorma','lr':2e-4},
    'lightweight_adapters': {'architecture':'lightweight','method':'adapters','lr':2e-4},
    'decoder_gradual': {'architecture':'decoder_only','method':'gradual','lr':2e-5},
    'decoder_linear_probe': {'architecture':'decoder_only','method':'linear_probe','lr':2e-4},
}
EXPERIMENTS = {**CORE_EXPERIMENTS, **EXTRA_EXPERIMENTS}
# Automatic finite sweep. One factor per trial relative to TUNE_BASE_EXPERIMENT.
TUNE_BASE_EXPERIMENT = 'lightweight_lorma'
TUNING_TRIALS = [
    {'name':'lr_low','overrides':{'lr':1e-4}},
    {'name':'more_chunks','overrides':{'max_chunks':8}},
    {'name':'longer_chunks','overrides':{'chunk_length':512}},
    {'name':'mean_max','overrides':{'pooling':'mean_max'}},
    {'name':'doc_transformer','overrides':{'pooling':'transformer'}},
    {'name':'head_tail','overrides':{'train_chunk_policy':'head_tail','eval_chunk_policy':'head_tail'}},
    {'name':'random_training_chunks','overrides':{'train_chunk_policy':'random_mix'}},
    {'name':'unweighted_loss','overrides':{'use_class_weights':False}},
]
ENSEMBLE_TOP_K = 3
ENSEMBLE_PAIR_WEIGHTS = [0.25,0.5,0.75]  # Each pair plus the best single model baseline.
assert EXPERIMENT in EXPERIMENTS and TUNE_BASE_EXPERIMENT in EXPERIMENTS
assert EPOCHS>0 and BATCH_SIZE>0 and GRAD_ACCUM_STEPS>0 and SAVE_EVERY_STEPS>0
assert 0<=KD_WEIGHT<=1 and KD_TEMPERATURE>0 and 0<=CLASS_WEIGHT_POWER<=1
assert MAX_OPTIMIZER_STEPS is None or MAX_OPTIMIZER_STEPS>0
seed_all(SEED)
''')
md('''
## 4. Download train, dev, test and verify binary case schemas

Train/dev may contain several instances with the same original case ID. Every row is retained
under a separate internal key, including rows with different labels. Original IDs are preserved
in exports. Counts and repeated-ID examples are saved to `case_instance_audit.json` in the
Phase 4 Drive folder. Cross-split checks use original case IDs and texts; test IDs must be unique.
File contents are checked before parsing: JSON, JSON Lines, Python literal containers and CSV with
explicit case columns are supported. A failed parse triggers one fresh download, then a split-specific
error. `dataset_loading_report.json` records format/size/status without printing case contents.
''')
code('''
FILES = {'train':'Phase_4_CJPE/Train/train.json','dev':'Phase_4_CJPE/Train/dev.json',
         'test':'Phase_4_CJPE/Test/test.json'}
local_files={}
for split,filename in tqdm(FILES.items(),desc='Downloading CJPE splits',unit='file'):
    local_files[split]=Path(hf_hub_download(repo_id=HF_DATASET_REPO,filename=filename,
                                         repo_type='dataset',token=HF_TOKEN))

def parse_case_file(path):
    import ast, csv, io
    path=Path(path)
    with path.open('rb') as stream:
        signature=stream.read(4)
    encoding='utf-16' if signature.startswith((b'\\xff\\xfe',b'\\xfe\\xff')) else 'utf-8-sig'
    try:
        content=path.read_text(encoding=encoding)
    except UnicodeError as exc:
        raise ValueError('File is not UTF-8/UTF-16 text; possibly an incomplete or non-text download.') from exc
    stripped=content.strip()
    if not stripped:
        raise ValueError('Downloaded file is empty or whitespace-only.')
    if stripped.startswith('version https://git-lfs.github.com/spec/v1'):
        raise ValueError('Downloaded file is a Git LFS pointer, not the dataset contents.')
    if stripped.lower().startswith(('<!doctype html','<html','<?xml')):
        raise ValueError('Downloaded file is an HTML/XML response, not case data.')
    try:
        return json.loads(content), 'json'
    except json.JSONDecodeError as exc:
        json_error=f'JSON parse failed at line {exc.lineno}, column {exc.colno}.'
    # Newline-delimited objects are accepted only if every nonblank line is valid.
    lines=io.StringIO(content)
    first=next((line.strip() for line in lines if line.strip()),'')
    if first.startswith('{'):
        try:
            initial=json.loads(first)
            if isinstance(initial,dict):
                rows=[initial]
                for number,line in enumerate(lines,2):
                    if not line.strip(): continue
                    row=json.loads(line)
                    if not isinstance(row,dict): raise ValueError('JSON Lines must contain objects.')
                    rows.append(row)
                return rows,'jsonl'
        except (json.JSONDecodeError,ValueError):
            pass
    # Some exports are CSV despite their .json suffix. Require known columns.
    csv_reader=csv.reader(io.StringIO(content))
    try:
        header=next(csv_reader)
    except (StopIteration,csv.Error):
        header=[]
    if 'id' in header and 'text' in header:
        csv.field_size_limit(min(__import__('sys').maxsize,2**31-1))
        reader=csv.DictReader(io.StringIO(content),strict=True)
        rows=[]
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('CSV row has missing or extra columns.')
                rows.append(row)
        except csv.Error as exc:
            raise ValueError('Malformed CSV case export.') from exc
        return rows,'csv'
    # Safe parsing only; never execute downloaded Python code.
    if stripped.lstrip('[').lstrip().startswith(('{'+chr(39),'{"')):
        try:
            literal=ast.literal_eval(stripped)
            if isinstance(literal,(list,dict)):
                return literal,'python_literal'
        except (SyntaxError,ValueError,TypeError,RecursionError):
            pass
    raise ValueError(json_error+' Unrecognized or malformed case-file format; no records were skipped.')

loading_report={}
def load_case_split(split):
    attempts=[]
    for attempt in range(2):
        path=local_files[split]
        detail={'attempt':attempt+1,'bytes':path.stat().st_size}
        print(f'Parsing {split}: {FILES[split]} ({detail["bytes"]:,} bytes)',flush=True)
        try:
            raw,format_name=parse_case_file(path)
        except ValueError as exc:
            detail.update(status='parse_failed',reason=str(exc)); attempts.append(detail)
            loading_report[split]=attempts
            save_json(loading_report,PROJECT_DIR/'dataset_loading_report.json')
            if attempt==0:
                print(f'{split}: {exc} Retrying with a fresh Hugging Face download.',flush=True)
                local_files[split]=Path(hf_hub_download(repo_id=HF_DATASET_REPO,filename=FILES[split],
                    repo_type='dataset',token=HF_TOKEN,force_download=True))
                continue
            raise ValueError(f'{split} ({FILES[split]}) could not be parsed after a fresh download: {exc} '
                             'See dataset_loading_report.json in the Phase 4 Drive folder.') from exc
        detail.update(status='parsed',format=format_name); attempts.append(detail)
        loading_report[split]=attempts
        save_json(loading_report,PROJECT_DIR/'dataset_loading_report.json')
        print(f'{split}: detected {format_name}',flush=True)
        return normalize_cases(raw,split!='test',split=split)

def case_key(case_id):
    if isinstance(case_id,bool) or not isinstance(case_id,(str,int)):
        raise ValueError('Case IDs must be strings or integers.')
    return str(case_id)

def normalize_cases(raw,labeled,split=None):
    if isinstance(raw,dict) and 'id' in raw and 'text' in raw:
        entries=[(None,raw)]
    elif isinstance(raw,dict) and isinstance(raw.get('data'),list):
        entries=[(None,r) for r in raw['data']]
    elif isinstance(raw,dict):
        entries=list(raw.items())
    elif isinstance(raw,list):
        entries=[(None,r) for r in raw]
    else:
        raise ValueError('Expected a list of case objects or an ID-keyed mapping.')
    result={}
    for row_index,(outer_id,record) in enumerate(entries):
        if isinstance(record,str) and outer_id is not None and not labeled:
            record={'id':outer_id,'text':record}
        if not isinstance(record,dict):
            raise ValueError('Every case must contain id/text and, for train/dev, label.')
        raw_id=record.get('id',outer_id)
        key=case_key(raw_id)
        if outer_id is not None and str(outer_id)!=key:
            raise ValueError('Keyed document ID differs from embedded ID.')
        text=record.get('text')
        # Some multi-instance exports contain paragraphs. Joining is safe for case classification.
        if isinstance(text,list) and all(isinstance(t,str) for t in text):
            text='\n\n'.join(text)
        if not isinstance(text,str):
            raise ValueError(f'{key}: expected case text string or ordered text fragments.')
        label=None
        if labeled:
            value=record.get('label',record.get('labels',record.get('outcome')))
            if isinstance(value,list) and len(value)==1:
                value=value[0]
            if isinstance(value,bool) or value not in [0,1,'0','1']:
                raise ValueError(f'{key}: expected a binary label in field label/labels/outcome.')
            label=int(value)
        # A labeled case can have several instances, even with different labels.
        # Internal row keys preserve every instance; original IDs stay untouched.
        instance_key=f'{split}:row:{row_index}' if split in ('train','dev') else key
        if instance_key in result:
            raise ValueError(f'{split or "input"}: duplicate test/ambiguous case ID {key}; inspect the source rows. No rows were overwritten.')
        result[instance_key]={'id':raw_id,'text':text,'label':label}
    return result

DATA={split:load_case_split(split) for split in FILES}
instance_audit={}
for split,docs in DATA.items():
    if not docs:
        raise ValueError(f'Empty {split} split.')
    print(split,len(docs),'cases; labels:',dict(Counter(d['label'] for d in docs.values())) if split!='test' else 'private')
    print('  Empty texts:',sum(not d['text'].strip() for d in docs.values()))
    counts=Counter(case_key(d['id']) for d in docs.values())
    repeated={key:count for key,count in counts.items() if count>1}
    instance_audit[split]={'instances':len(docs),'unique_original_ids':len(counts),
                         'repeated_ids':len(repeated),'extra_instances':len(docs)-len(counts),
                         'repeated_id_examples':list(repeated.items())[:20]}
    print('  Unique original IDs:',len(counts),'; repeated IDs:',len(repeated),'; all instances retained.')
save_json(instance_audit,PROJECT_DIR/'case_instance_audit.json')
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    if {case_key(d['id']) for d in DATA[a].values()} & {case_key(d['id']) for d in DATA[b].values()}:
        raise ValueError(f'{a}/{b} IDs overlap; inspect the official files before training.')
    ha={hashlib.sha256(d['text'].encode()).hexdigest() for d in DATA[a].values() if d['text'].strip()}
    hb={hashlib.sha256(d['text'].encode()).hexdigest() for d in DATA[b].values() if d['text'].strip()}
    if ha & hb:
        raise ValueError(f'{a}/{b} contain duplicate case texts; investigate leakage.')
if set(d['label'] for d in DATA['train'].values())!={0,1}:
    raise ValueError('Both outcome classes are required in training.')
fingerprint=hashlib.sha256()
for split in sorted(DATA):
    for key,doc in sorted(DATA[split].items()):
        # Hash one case at a time rather than building another entire dataset-sized JSON string.
        fingerprint.update(json.dumps([split,key,doc],sort_keys=True,ensure_ascii=False).encode())
DATA_FINGERPRINT=fingerprint.hexdigest()
save_json({'files':FILES,'fingerprint':DATA_FINGERPRINT,'counts':{s:len(d) for s,d in DATA.items()}},PROJECT_DIR/'dataset_manifest.json')
'''.replace("text='\n\n'.join(text)","text=chr(10).join(text)"))
md('''
## 5. Phase 1 BPE tokenizer, persistent token cache, and document chunking

Token IDs are cached as memory-mapped int32 files rather than loading the entire token corpus into RAM.
On Colab a local disk copy accelerates random reads; the canonical cache remains in Drive.
The default uses up to four evenly spaced overlapping windows, including document beginning and end.
This is a **sampled representation**, not a claim that every token was read. Coverage is reported and
saved. Set `MAX_CHUNKS=0` or the `all` policy to use every window; this can be much slower and larger.
`CHUNK_BATCH_SIZE` bounds the number of chunks sent through the backbone at once. An empty case gets
one UNK-containing chunk, so every official ID still receives a prediction.
''')
code('''
TOKENIZER_DIR=PHASE1_DIR/'bpe_tokenizer'
for filename in ['vocab.json','merges.txt']:
    if not (TOKENIZER_DIR/filename).is_file():
        raise FileNotFoundError(TOKENIZER_DIR/filename)
tok=ByteLevelBPETokenizer(str(TOKENIZER_DIR/'vocab.json'),str(TOKENIZER_DIR/'merges.txt'))
PAD_ID,BOS_ID,EOS_ID,UNK_ID,SEP_ID=[tok.token_to_id(s) for s in ['<pad>','<bos>','<eos>','<unk>','<sep>']]
assert all(v is not None for v in [PAD_ID,BOS_ID,EOS_ID,UNK_ID,SEP_ID])
VOCAB_SIZE_ACTUAL=tok.get_vocab_size()
TOKENIZER_HASH=hashlib.sha256(b''.join((TOKENIZER_DIR/f).read_bytes() for f in ['vocab.json','merges.txt'])).hexdigest()
import shutil
(PROJECT_DIR/'bpe_tokenizer').mkdir(exist_ok=True)
for filename in ['vocab.json','merges.txt']:
    shutil.copy2(TOKENIZER_DIR/filename,PROJECT_DIR/'bpe_tokenizer'/filename)
class TokenCache:
    def __init__(self,binary_path,index):
        self.index=index
        expected=sum(size for start,size in index.values())
        if binary_path.stat().st_size!=4*expected:
            raise ValueError('Truncated token-cache file: remove/rebuild this cache before training.')
        self.tokens=np.memmap(binary_path,dtype='<i4',mode='r') if expected else np.empty(0,dtype=np.int32)
    def __getitem__(self,key):
        start,size=self.index[key]
        return self.tokens[start:start+size]
    def __len__(self):
        return len(self.index)

TOKENS={}
for split,docs in DATA.items():
    stem=f'tokens_{split}_{DATA_FINGERPRINT[:12]}_{TOKENIZER_HASH[:12]}'
    binary_path=PROJECT_DIR/(stem+'.bin'); index_path=PROJECT_DIR/(stem+'.json')
    if binary_path.is_file() and index_path.is_file():
        index=json.loads(index_path.read_text())
        if set(index)!=set(docs): raise ValueError('Token cache IDs disagree with current split.')
        print('Reused disk-backed token cache:',split)
    else:
        index={}; offset=0; temporary=binary_path.with_suffix('.bin.tmp')
        with open(temporary,'wb',buffering=8*1024*1024) as stream:
            for key,doc in tqdm(docs.items(),desc='Tokenizing '+split,unit='case'):
                ids=np.asarray(tok.encode(doc['text'],add_special_tokens=False).ids,dtype='<i4')
                stream.write(ids.tobytes()); index[key]=[offset,len(ids)]; offset+=len(ids)
        os.replace(temporary,binary_path); save_json(index,index_path)
    # Stage once onto Colab's local disk; canonical token caches and all results persist in Drive.
    active_path=binary_path
    if IN_COLAB:
        staging=Path('/content/cs781_phase4_token_cache'); staging.mkdir(exist_ok=True)
        active_path=staging/binary_path.name
        if not active_path.is_file() or active_path.stat().st_size!=binary_path.stat().st_size:
            temporary=active_path.with_suffix('.tmp')
            with open(binary_path,'rb') as source_stream,open(temporary,'wb') as target_stream:
                with tqdm.wrapattr(source_stream,'read',total=binary_path.stat().st_size,desc='Staging '+split,bytes=True) as monitored:
                    shutil.copyfileobj(monitored,target_stream,length=8*1024*1024)
            os.replace(temporary,active_path)
    TOKENS[split]=TokenCache(active_path,index)

def window_ranges(n,length,overlap):
    size=length-2
    if n==0:
        return [(0,0)]
    ranges=[]; start=0
    while True:
        end=min(start+size,n); ranges.append((start,end))
        if end==n:
            return ranges
        start+=size-overlap

def select_ranges(ranges,limit,policy,key='',epoch=0,seed=SEED):
    if policy=='all' or limit==0 or len(ranges)<=limit:
        return list(ranges)
    if policy=='head':
        return ranges[:limit]
    if policy=='tail':
        return ranges[-limit:]
    if policy=='head_tail':
        first=(limit+1)//2
        return ranges[:first]+(ranges[-(limit-first):] if limit-first else [])
    if policy=='random_mix':
        stable=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
        rng=random.Random(seed+epoch*1000003+stable)
        if limit==1:
            return [ranges[rng.randrange(len(ranges))]]
        chosen=[0,len(ranges)-1]+rng.sample(list(range(1,len(ranges)-1)),limit-2)
        return [ranges[i] for i in sorted(chosen)]
    if policy=='spread':
        indices=np.linspace(0,len(ranges)-1,limit).round().astype(int).tolist()
        return [ranges[i] for i in indices]
    raise ValueError('Unknown chunk policy '+policy)

def covered_tokens(ranges):
    total=0; previous=0
    for start,end in sorted(ranges):
        total+=max(0,end-max(start,previous)); previous=max(previous,end)
    return total

class CaseDataset(Dataset):
    def __init__(self,split,cfg,training=False):
        self.split=split; self.cfg=cfg; self.training=training; self.epoch=0
        self.keys=list(DATA[split])
    def __len__(self):
        return len(self.keys)
    def __getitem__(self,index):
        key=self.keys[index]; tokens=TOKENS[self.split][key]
        ranges=window_ranges(len(tokens),self.cfg['chunk_length'],self.cfg['chunk_overlap'])
        policy=self.cfg['train_chunk_policy'] if self.training else self.cfg['eval_chunk_policy']
        selected=select_ranges(ranges,self.cfg['max_chunks'],policy,key,self.epoch,self.cfg['seed'])
        chunks=[[BOS_ID]+(tokens[s:e].tolist() or [UNK_ID])+[EOS_ID] for s,e in selected]
        positions=[[s/max(1,len(tokens)),e/max(1,len(tokens))] for s,e in selected]
        coverage=covered_tokens(selected)/max(1,len(tokens)) if len(tokens) else 1.0
        stats=[math.log1p(len(tokens))/10,math.log1p(len(ranges))/5,coverage,len(selected)/max(1,len(ranges))]
        return {'key':key,'chunks':chunks,'positions':positions,'stats':stats,
                'label':DATA[self.split][key]['label'],'coverage':coverage}

def collate_cases(rows):
    chunks=[(owner,ids,pos) for owner,row in enumerate(rows) for ids,pos in zip(row['chunks'],row['positions'])]
    length=max(len(ids) for _,ids,_ in chunks)
    ids=torch.full((len(chunks),length),PAD_ID,dtype=torch.long)
    mask=torch.zeros_like(ids,dtype=torch.bool)
    content=torch.zeros_like(mask)
    for i,(_,values,_) in enumerate(chunks):
        ids[i,:len(values)]=torch.tensor(values); mask[i,:len(values)]=True
        content[i,1:len(values)-1]=True
    return {'input_ids':ids,'attention_mask':mask,'target_mask':content,
            'chunk_positions':torch.tensor([p for _,_,p in chunks],dtype=torch.float32),
            'owners':torch.tensor([o for o,_,_ in chunks]),
            'stats':torch.tensor([r['stats'] for r in rows],dtype=torch.float32),
            'labels':torch.tensor([r['label'] if r['label'] is not None else -100 for r in rows]),
            'keys':[r['key'] for r in rows]}
''')
md('## 6. Checkpoint-compatible Phase 1 definitions — no pretraining is rerun')
for prefix in ['class RMSNorm','class CausalSelfAttention','class RelativePositionBias','class RoPEAttention','class GQAAttention']:
    code(source(prefix))
md('## 7. Hierarchical case classifier and adaptation methods')
base=source('def torch_load')
base=base[:base.index('class LegalRR')]
code(base)
rr=source('def torch_load')
tree=ast.parse(rr)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LegalRR')
feature=ast.get_source_segment(rr,node).replace('class LegalRR','class CaseFeatures')
start=feature.index('        self.dropout =');end=feature.index('        self.use_checkpointing')
feature=feature[:start]+feature[end:]
feature=feature.replace('self.use_checkpointing = GRADIENT_CHECKPOINTING','self.use_checkpointing = GRADIENT_CHECKPOINTING')
feature=feature.replace('def forward(self, input_ids, attention_mask, target_mask, domains, positions):','def forward(self, input_ids, attention_mask, target_mask):')
start=feature.index('        features = torch.cat')
feature=feature[:start]+'''        return torch.cat([pooled, final], dim=-1)
'''
code(feature)
code('''
class LowRankMultiplicative(nn.Module):
    # LoRMA+, Eq. 8: (I + alpha/r B A) W x + bias.
    def __init__(self,base,rank,alpha):
        super().__init__(); self.base=base; base.requires_grad_(False)
        self.A=nn.Linear(base.out_features,rank,bias=False)
        self.B=nn.Linear(rank,base.out_features,bias=False)
        self.scale=alpha/rank
        nn.init.kaiming_uniform_(self.A.weight,a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
    def forward(self,x):
        wx=F.linear(x,self.base.weight,None)
        y=wx+self.scale*self.B(self.A(wx))
        return y if self.base.bias is None else y+self.base.bias

class ResidualAdapter(nn.Module):
    # One post-block residual bottleneck adapter; not an exact two-site Houlsby reproduction.
    def __init__(self,base,dim,bottleneck):
        super().__init__(); self.base=base; base.requires_grad_(False)
        self.norm=nn.LayerNorm(dim); self.down=nn.Linear(dim,bottleneck); self.up=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,*args):
        x=self.base(*args)
        return x+self.up(F.gelu(self.down(self.norm(x))))

class LegalCase(nn.Module):
    def __init__(self,features,cfg):
        super().__init__(); self.features=features; self.cfg=cfg
        dim=features.backbone.tok_emb.embedding_dim; hidden=cfg['doc_hidden']
        self.chunk_project=nn.Sequential(nn.LayerNorm(2*dim),nn.Linear(2*dim,hidden),nn.GELU())
        self.position_project=nn.Linear(2,hidden)
        self.gate=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.document_encoder=None
        if cfg['pooling']=='transformer':
            layer=nn.TransformerEncoderLayer(hidden,4,4*hidden,dropout=cfg['dropout'],batch_first=True,norm_first=True)
            self.document_encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        n_features=4 if cfg['pooling']=='mean_max' else 3
        self.classifier=nn.Sequential(nn.LayerNorm(n_features*hidden+4),nn.Linear(n_features*hidden+4,hidden),
            nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(hidden,2))
    def forward(self,input_ids,attention_mask,target_mask,chunk_positions,owners,stats):
        chunks=[]
        for first in range(0,len(input_ids),self.cfg['chunk_batch_size']):
            stop=first+self.cfg['chunk_batch_size']
            features=self.features(input_ids[first:stop],attention_mask[first:stop],target_mask[first:stop])
            chunks.append(self.chunk_project(features)+self.position_project(chunk_positions[first:stop]))
        x=torch.cat(chunks,dim=0)
        documents=[]
        for owner in range(len(stats)):
            h=x[owners==owner]
            if not len(h):
                raise ValueError('Missing chunks for a case.')
            pooling=self.cfg['pooling']
            if pooling=='attention':
                pooled=(h*torch.softmax(self.gate(h).squeeze(-1),dim=0)[:,None]).sum(0)
            elif pooling=='mean': pooled=h.mean(0)
            elif pooling=='max': pooled=h.max(0).values
            elif pooling=='mean_max': pooled=torch.cat([h.mean(0),h.max(0).values])
            elif pooling=='transformer': pooled=self.document_encoder(h[None])[0].mean(0)
            else: raise ValueError('Unknown document pooling.')
            documents.append(torch.cat([pooled,h[0],h[-1],stats[owner]]))
        return self.classifier(torch.stack(documents))

def layer_list(model):
    b=model.features.backbone
    return b.enc_layers if model.cfg['spec']['architecture']=='encoder_decoder' else b.blocks

def unfreeze_last_layers(model,count):
    for layer in list(layer_list(model))[-count:] if count else []:
        layer.requires_grad_(True)
    b=model.features.backbone
    norm=b.ln_enc_f if model.cfg['spec']['architecture']=='encoder_decoder' else (
        b.ln_f if model.cfg['spec']['architecture']=='decoder_only' else b.norm_f)
    norm.requires_grad_(True)

def make_model(cfg):
    spec=cfg['spec']; b,phase1_config,original_parameters=load_phase1(spec['architecture'])
    if cfg['chunk_length']>phase1_config.get('max_len',1024):
        raise ValueError('Chunk length exceeds Phase 1 positional embeddings.')
    selected=None
    if spec['method']=='kd':
        selected=list(range(0,len(b.blocks),2)); b.blocks=nn.ModuleList([b.blocks[i] for i in selected])
    features=CaseFeatures(b,spec['architecture']); features.use_checkpointing=cfg['gradient_checkpointing']
    model=LegalCase(features,cfg)
    b.requires_grad_(spec['method'] in ['full','kd'])
    targets=[]; method=spec['method']
    if method=='partial': unfreeze_last_layers(model,min(cfg['partial_layers'],len(layer_list(model))))
    elif method=='bitfit':
        for name,p in b.named_parameters(): p.requires_grad_(name.endswith('.bias'))
    elif method=='lorma':
        for name,child in list(b.named_modules()):
            if isinstance(child,nn.Linear) and '.attn.' in name and name.rsplit('.',1)[-1] in ['qkv','proj','q_proj','k_proj','v_proj','o_proj']:
                parent,leaf=name.rsplit('.',1)
                setattr(b.get_submodule(parent),leaf,LowRankMultiplicative(child,cfg['adapter_rank'],cfg['adapter_alpha']))
                targets.append(name)
        assert targets
    elif method=='adapters':
        layers=layer_list(model)
        for i in range(len(layers)):
            layers[i]=ResidualAdapter(layers[i],b.tok_emb.embedding_dim,cfg['bottleneck_dim'])
            targets.append(str(i))
    elif method not in ['full','kd','gradual','linear_probe']:
        raise ValueError('Unknown adaptation method.')
    metadata={'phase1_config':phase1_config,'phase1_parameters':original_parameters,'selected_phase1_layers':selected,
        'adapter_targets':targets,'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters_initial':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model,metadata

def forward_batch(model,batch):
    return model(*[batch[k].to(DEVICE) for k in ['input_ids','attention_mask','target_mask','chunk_positions','owners','stats']])
''')
md('''
## 8. Exact case-level macro-F1, threshold search, and verification

The primary metric is the mean F1 of outcome classes 0 and 1. The document's secondary Precision/Recall
are reported for class 1; macro precision/recall and both class reports are also saved because §10.4 does
not specify their averaging convention. This notebook does not claim parity with an unseen evaluator.
The dev-only threshold search examines all distinct decisions efficiently, includes 0.5, and favors the
threshold closest to 0.5 on ties. Test labels are never scored or used to select a threshold.
''')
code('''
def binary_metrics(gold,predicted):
    gold=np.asarray(gold,dtype=np.int64); predicted=np.asarray(predicted,dtype=np.int64)
    if gold.shape!=predicted.shape or gold.ndim!=1 or not len(gold) or not np.isin(gold,[0,1]).all() or not np.isin(predicted,[0,1]).all():
        raise ValueError('Expected equally sized, nonempty binary label vectors.')
    cm=np.zeros((2,2),dtype=np.int64); np.add.at(cm,(gold,predicted),1)
    per={}
    for label in [0,1]:
        tp=int(cm[label,label]); fp=int(cm[:,label].sum()-tp); fn=int(cm[label,:].sum()-tp)
        per[str(label)]={'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,'tp':tp,'fp':fp,'fn':fn,'support':tp+fn}
    return {'macro_f1':sum(r['f1'] for r in per.values())/2,'precision':per['1']['precision'],
        'recall':per['1']['recall'],'macro_precision':sum(r['precision'] for r in per.values())/2,
        'macro_recall':sum(r['recall'] for r in per.values())/2,'accuracy':float(np.trace(cm)/len(gold)),
        'per_class':per,'confusion_matrix':cm.tolist(),'cases':len(gold)}

def optimal_threshold(probabilities,gold,tune=True):
    p=np.asarray(probabilities,dtype=np.float64); y=np.asarray(gold,dtype=np.int64)
    if p.shape!=y.shape or not np.isfinite(p).all() or not ((0<=p)&(p<=1)).all():
        raise ValueError('Invalid dev probability vector.')
    baseline=binary_metrics(y,(p>=0.5).astype(int))
    if not tune: return 0.5,baseline,baseline
    order=np.argsort(-p,kind='stable'); ps=p[order]; ys=y[order]
    ends=np.flatnonzero(np.r_[ps[:-1]!=ps[1:],True])
    tp=np.r_[0,np.cumsum(ys)[ends]]; count=np.r_[0,ends+1]
    fp=count-tp; fn=int(y.sum())-tp; tn=len(y)-int(y.sum())-fp
    positive=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    negative=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)>0)
    scores=(positive+negative)/2
    thresholds=np.r_[np.nextafter(ps[0],np.inf),ps[ends]]
    best_score=max(float(scores.max()),baseline['macro_f1'])
    options=[(abs(float(t)-0.5),float(t)) for t,s in zip(thresholds,scores) if math.isclose(float(s),best_score,rel_tol=0.0,abs_tol=1e-12)]
    if math.isclose(baseline['macro_f1'],best_score,rel_tol=0.0,abs_tol=1e-12): options.append((0.0,0.5))
    threshold=min(options)[1]
    selected=binary_metrics(y,(p>=threshold).astype(int))
    assert selected['macro_f1']+1e-12>=baseline['macro_f1']
    return threshold,selected,baseline

def show_scores(metrics,title):
    print(title)
    print(f"Macro-F1 [PRIMARY]: {metrics['macro_f1']:.6f}; Precision (class 1): {metrics['precision']:.6f}; Recall (class 1): {metrics['recall']:.6f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.6f}; Macro-Recall: {metrics['macro_recall']:.6f}; Accuracy: {metrics['accuracy']:.6f}")
    for label,row in metrics['per_class'].items(): print('Class',label,row)
    print('Confusion matrix (gold rows, predicted columns):',metrics['confusion_matrix'])

@torch.no_grad()
def predict_logits(model,split,cfg,description=None):
    dataset=CaseDataset(split,cfg,training=False)
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,collate_fn=collate_cases,
        num_workers=NUM_WORKERS,generator=torch.Generator().manual_seed(cfg['seed']))
    was_training=model.training; model.eval(); result={}
    for batch in tqdm(loader,desc=description or 'Predicting '+split,unit='batch'):
        with amp_context(): logits=forward_batch(model,batch)
        logits=logits.float().cpu()
        if not torch.isfinite(logits).all(): raise ValueError('Non-finite case logits.')
        for i,key in enumerate(batch['keys']):
            if key in result: raise ValueError('Duplicate prediction case.')
            result[key]=logits[i]
    model.train(was_training)
    if set(result)!=set(DATA[split]): raise ValueError('Missing/extra prediction IDs.')
    keys=list(DATA[split]); array=torch.stack([result[k] for k in keys])
    return keys,array

def probability_vector(logits):
    return torch.softmax(logits.double(),dim=-1)[:,1].numpy()

def validate_submission(predictions,docs):
    if not isinstance(predictions,list) or len(predictions)!=len(docs): raise ValueError('Expected one prediction object per case.')
    # Occurrence order distinguishes labeled instances sharing an original ID.
    # Test IDs are unique, as checked during loading. Preserve original ID types.
    expected={}
    for key,doc in docs.items():
        expected.setdefault((type(doc['id']),doc['id']),[]).append(key)
    consumed=Counter(); matched={}
    for item in predictions:
        if set(item)!={'id','pred'}: raise ValueError('Submission objects must contain exactly id and pred.')
        case_key(item['id'])
        identity=(type(item['id']),item['id']); occurrence=consumed[identity]
        if identity not in expected or occurrence>=len(expected[identity]): raise ValueError('Duplicate, unknown or changed case ID.')
        if type(item['pred']) is not int or item['pred'] not in [0,1]: raise ValueError('pred must be an integer 0 or 1.')
        matched[expected[identity][occurrence]]=item['pred']; consumed[identity]+=1
    if set(matched)!=set(docs): raise ValueError('Missing prediction instances.')
    return matched

def save_predictions(output,probabilities,threshold):
    predictions=[{'id':d['id'],'pred':int(p>=threshold)} for d,p in zip(DATA['test'].values(),probabilities)]
    validate_submission(predictions,DATA['test']); save_json(predictions,output/'prediction.json')
    assert json.loads((output/'prediction.json').read_text())==predictions
    temporary=output/'submission.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(output/'prediction.json',arcname='prediction.json')
    os.replace(temporary,output/'submission.zip')
    with zipfile.ZipFile(output/'submission.zip') as archive:
        assert archive.namelist()==['prediction.json'] and json.loads(archive.read('prediction.json'))==predictions
    return predictions
''')
md('## 9. Configuration fingerprints, teacher cache, and deterministic resume state')
for prefix in ['def capture_rng','def restore_rng']:code(source(prefix))
code('''
def base_config(name,overrides=None):
    cfg={'experiment':name,'spec':copy.deepcopy(EXPERIMENTS[name]),'run_tag':RUN_TAG,'seed':SEED,
        'epochs':EPOCHS,'batch_size':BATCH_SIZE,'grad_accum':GRAD_ACCUM_STEPS,'chunk_length':CHUNK_LENGTH,
        'chunk_overlap':CHUNK_OVERLAP,'max_chunks':MAX_CHUNKS,'chunk_batch_size':CHUNK_BATCH_SIZE,
        'train_chunk_policy':TRAIN_CHUNK_POLICY,'eval_chunk_policy':EVAL_CHUNK_POLICY,'pooling':DOC_POOLING,
        'doc_hidden':DOC_HIDDEN,'dropout':HEAD_DROPOUT,'head_lr_multiplier':HEAD_LR_MULTIPLIER,
        'use_class_weights':USE_CLASS_WEIGHTS,'class_weight_power':CLASS_WEIGHT_POWER,'weight_decay':WEIGHT_DECAY,
        'warmup_fraction':WARMUP_FRACTION,'patience':EARLY_STOPPING_PATIENCE,'gradient_checkpointing':GRADIENT_CHECKPOINTING,
        'adapter_rank':ADAPTER_RANK,'adapter_alpha':ADAPTER_ALPHA,'bottleneck_dim':BOTTLENECK_DIM,
        'partial_layers':PARTIAL_LAYERS,'unfreeze_layers_per_epoch':UNFREEZE_LAYERS_PER_EPOCH,
        'kd_weight':KD_WEIGHT,'kd_temperature':KD_TEMPERATURE,'teacher_requested':TEACHER_RUN_DIR,
        'max_optimizer_steps':MAX_OPTIMIZER_STEPS,'tune_threshold':TUNE_THRESHOLD,
        'phase1_heads':PHASE1_HEADS,'tokenizer_hash':TOKENIZER_HASH,'data_fingerprint':DATA_FINGERPRINT}
    for key,value in (overrides or {}).items():
        if key=='lr': cfg['spec']['lr']=value
        elif key in cfg: cfg[key]=value
        else: raise ValueError('Unknown tuning override '+key)
    assert 4<=cfg['chunk_length']<=1024 and 0<=cfg['chunk_overlap']<cfg['chunk_length']-2
    assert cfg['max_chunks']>=0 and cfg['chunk_batch_size']>0 and cfg['doc_hidden']>0
    assert cfg['pooling'] in ['attention','mean','max','mean_max','transformer']
    assert cfg['pooling']!='transformer' or cfg['doc_hidden']%4==0
    assert cfg['train_chunk_policy'] in ['spread','head','tail','head_tail','random_mix','all']
    assert cfg['eval_chunk_policy'] in ['spread','head','tail','head_tail','all']
    assert cfg['partial_layers']>0 and cfg['unfreeze_layers_per_epoch']>0 and cfg['adapter_rank']>0
    return cfg

def run_path(run_name): return PROJECT_DIR/(run_name+'_'+RUN_TAG)

def coverage_report(cfg,output):
    report={}
    for split in DATA:
        dataset=CaseDataset(split,cfg,False); ratios=[]; counts=[]
        for i in tqdm(range(len(dataset)),desc='Auditing chunk coverage: '+split,unit='case'):
            row=dataset[i]; ratios.append(row['coverage']); counts.append(len(row['chunks']))
        report[split]={'cases':len(dataset),'mean_token_coverage':float(np.mean(ratios)),
            'cases_with_full_coverage':sum(r>=1-1e-9 for r in ratios),'mean_chunks':float(np.mean(counts)),
            'max_chunks':max(counts)}
    save_json(report,output/'chunk_coverage.json'); print('Chunk coverage:',report)
    return report

def teacher_logits_for_student(student_parameters,cfg):
    choices=[]
    paths=[Path(cfg['teacher_requested'])] if cfg['teacher_requested'] else [p.parent for p in PROJECT_DIR.glob('*/metrics.json')]
    for path in paths:
        if not (path/'best_model.pt').is_file() or not (path/'run_config.json').is_file(): continue
        metrics=json.loads((path/'metrics.json').read_text()); saved_cfg=json.loads((path/'run_config.json').read_text())
        if saved_cfg['data_fingerprint']!=DATA_FINGERPRINT or saved_cfg['tokenizer_hash']!=TOKENIZER_HASH: continue
        if not metrics['training_complete'] or metrics['total_parameters']<=student_parameters: continue
        if saved_cfg['spec']['method']=='kd': continue
        choices.append((metrics['macro_f1'],str(path)))
    if not choices: raise FileNotFoundError('KD requires a completed, compatible, larger Phase 4 teacher. Run a core non-KD experiment first.')
    _,selected=max(choices); path=Path(selected)
    teacher_cfg=json.loads((path/'run_config.json').read_text())
    checkpoint_stat=(path/'best_model.pt').stat()
    signature={'checkpoint_bytes':checkpoint_stat.st_size,'checkpoint_mtime_ns':checkpoint_stat.st_mtime_ns,
               'data_fingerprint':DATA_FINGERPRINT,'tokenizer_hash':TOKENIZER_HASH}
    cache_path=path/'teacher_train_logits.pt'
    cached=torch_load(cache_path) if cache_path.is_file() else None
    if cached is None or cached.get('signature')!=signature:
        if teacher_cfg['phase1_heads']!=PHASE1_HEADS: raise ValueError('Teacher Phase 1 head counts differ.')
        state=torch_load(path/'best_model.pt')
        teacher,_=make_model(teacher_cfg); teacher.load_state_dict(state['model'],strict=True)
        del state
        teacher.eval().requires_grad_(False); teacher=teacher.to(DEVICE)
        keys,logits=predict_logits(teacher,'train',teacher_cfg,'Caching teacher logits on TRAIN only')
        cached={'signature':signature,'keys':keys,'logits':logits}
        save_torch(cached,cache_path)
        del teacher; gc.collect()
        if DEVICE.type=='cuda': torch.cuda.empty_cache()
    keys=cached['keys']; logits=cached['logits']
    if set(keys)!=set(DATA['train']) or len(keys)!=len(set(keys)) or logits.shape!=(len(keys),2) or not torch.isfinite(logits).all():
        raise ValueError('Invalid teacher train-logit cache.')
    print('KD teacher:',path,'(student may use different chunking/hyperparameters)')
    return {key:logits[i] for i,key in enumerate(keys)},selected,signature
''')
md('## 10. Pre-training checks: chunks, metrics, labels, and adaptation identities')
code('''
assert binary_metrics([0,1],[0,1])['macro_f1']==1.0
assert binary_metrics([0,1],[1,0])['macro_f1']==0.0
for probabilities,labels in [([0.1,0.4,0.6,0.9],[0,1,1,1]),([0.2,0.2,0.7],[0,1,1])]:
    threshold,selected,baseline=optimal_threshold(probabilities,labels)
    assert selected['macro_f1']>=baseline['macro_f1']
for length in [0,1,500,2000]:
    ranges=window_ranges(length,384,64)
    assert covered_tokens(ranges)==length
    selected=select_ranges(ranges,2,'spread')
    assert selected[0]==ranges[0] and selected[-1]==ranges[-1]
base=nn.Linear(7,11); layer=LowRankMultiplicative(copy.deepcopy(base),3,6)
x=torch.randn(2,5,7); torch.testing.assert_close(layer(x),base(x))
layer(x).sum().backward(); assert layer.B.weight.grad.abs().sum()>0 and layer.base.weight.grad is None
print('PASS: coverage, threshold/metric formulas, LoRMA identity and gradients. Full run checks follow after training.')
''')
md('''
## 11. Train, select by dev macro-F1, reload, and verify

Every profile starts by strictly loading your Phase 1 checkpoint. Full/partial/adapters/LORMA/BitFit
are actually applied here; there is no separate manual PEFT step. Case-level CE uses optional train-only
class weights. KD adds temperature-scaled KL from the cached TRAIN teacher logits. Gradual-unfreezing
profiles include all future parameters in the optimizer and unfreeze more blocks each epoch.

Chunk sampling is deterministic by case/epoch. Train and inference loaders use private RNG generators.
The last partial gradient accumulation group is handled correctly. The best checkpoint is selected by
dev **threshold-optimized macro-F1**, then reloaded and independently rescored. A capped smoke test is
marked incomplete and excluded from teacher/ensemble/coverage eligibility.
''')
code('''
def optimizer_groups(model,cfg):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad and cfg['spec']['method']!='gradual': continue
        head=not name.startswith('features.backbone.')
        no_decay=p.ndim<2 or name.endswith('.bias') or 'norm' in name or '.ln' in name
        groups.setdefault((head,no_decay),[]).append(p)
    return [{'params':params,'lr':cfg['spec']['lr']*(cfg['head_lr_multiplier'] if head else 1),
        'weight_decay':0.0 if no_decay else cfg['weight_decay']} for (head,no_decay),params in groups.items()]

def save_probabilities(output,split,keys,p):
    save_json([{'id':DATA[split][key]['id'],'instance_key':key,'p1':float(value)} for key,value in zip(keys,p)],output/(split+'_probabilities.json'))

def run_experiment(name,overrides=None,run_name=None):
    cfg=base_config(name,overrides); seed_all(cfg['seed']); run_name=run_name or name
    output=run_path(run_name); output.mkdir(parents=True,exist_ok=True)
    config_path=output/'run_config.json'
    if config_path.is_file() and json.loads(config_path.read_text())!=cfg:
        raise ValueError('Settings changed for '+str(output)+'. Choose a new RUN_TAG or trial name.')
    if not RESUME and (output/'last.pt').exists(): raise ValueError('Use a fresh tag instead of overwriting a run.')
    save_json(cfg,config_path); coverage_report(cfg,output)
    # Obtain teacher first so its temporary GPU model is released before moving the student to GPU.
    model,metadata=make_model(cfg); teacher_cache=None; teacher_name=None; teacher_signature=None
    if cfg['spec']['method']=='kd':
        student_rng=capture_rng()
        try:
            teacher_cache,teacher_name,teacher_signature=teacher_logits_for_student(metadata['total_parameters'],cfg)
        finally:
            restore_rng(student_rng)
    metadata['teacher_run']=teacher_name; metadata['teacher_signature']=teacher_signature; model=model.to(DEVICE)
    save_json(metadata,output/'model_metadata.json')
    print(run_name,'parameters:',metadata['total_parameters'],'initial trainable:',metadata['trainable_parameters_initial'])
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg))
    counts=np.bincount([d['label'] for d in DATA['train'].values()],minlength=2)
    weights=torch.tensor((counts.mean()/counts)**cfg['class_weight_power'],dtype=torch.float32,device=DEVICE) if cfg['use_class_weights'] else None
    save_json({'counts':counts.tolist(),'weights':weights.tolist() if weights is not None else [1,1]},output/'class_weights.json')
    dataset=CaseDataset('train',cfg,training=True)
    n_batches=math.ceil(len(dataset)/cfg['batch_size']); total_steps=cfg['epochs']*math.ceil(n_batches/cfg['grad_accum'])
    warmup=max(1,round(total_steps*cfg['warmup_fraction']))
    def scale(step):
        return (step+1)/warmup if step<warmup else 0.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda' and AMP_DTYPE==torch.float16)
    history=[]; best_f1=-1.0; bad_epochs=0; step=0; start_epoch=0; start_batch=0; complete=False; elapsed=0.0
    if RESUME and (output/'last.pt').is_file():
        state=torch_load(output/'last.pt')
        if state['run_config']!=cfg or state['metadata']['teacher_run']!=teacher_name or state['metadata'].get('teacher_signature')!=teacher_signature: raise ValueError('Saved settings/teacher differ.')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
        history=state['history']; best_f1=state['best_f1']; bad_epochs=state['bad_epochs']; step=state['step']
        start_epoch=state['epoch']; start_batch=state['next_batch']; complete=state['training_complete']; elapsed=state['training_seconds']
        del state; print('Resume epoch',start_epoch+1,'batch',start_batch,'step',step,'complete',complete)
    started=time.time()
    if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
    def persist(epoch,next_batch):
        save_torch({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'rng':capture_rng(),'run_config':cfg,'metadata':metadata,'history':history,
            'best_f1':best_f1,'bad_epochs':bad_epochs,'step':step,'epoch':epoch,'next_batch':next_batch,
            'training_complete':complete,'training_seconds':elapsed+time.time()-started},output/'last.pt')
        save_json(history,output/'history.json')
    if not complete:
        for epoch in range(start_epoch,cfg['epochs']):
            if cfg['max_optimizer_steps'] is not None and step>=cfg['max_optimizer_steps']: break
            if cfg['spec']['method']=='gradual':
                unfreeze_last_layers(model,min(len(layer_list(model)),(epoch+1)*cfg['unfreeze_layers_per_epoch']))
            dataset.epoch=epoch
            order=list(range(len(dataset))); random.Random(cfg['seed']+epoch).shuffle(order)
            begin=start_batch if epoch==start_epoch else 0
            loader=DataLoader(torch.utils.data.Subset(dataset,order[begin*cfg['batch_size']:]),batch_size=cfg['batch_size'],
                shuffle=False,collate_fn=collate_cases,num_workers=NUM_WORKERS,generator=torch.Generator().manual_seed(cfg['seed']+epoch))
            model.train(); optimizer.zero_grad(set_to_none=True); group_loss=0.0; next_batch=begin; capped=False
            progress=tqdm(loader,desc=f'{run_name} epoch {epoch+1}/{cfg["epochs"]}',unit='case batch')
            for local_i,batch in enumerate(progress):
                i=begin+local_i; next_batch=i+1
                group_start=(i//cfg['grad_accum'])*cfg['grad_accum']
                group_cases=min(cfg['batch_size']*cfg['grad_accum'],len(dataset)-group_start*cfg['batch_size'])
                case_fraction=len(batch['keys'])/group_cases
                labels=batch['labels'].to(DEVICE)
                with amp_context():
                    logits=forward_batch(model,batch)
                    per_case=F.cross_entropy(logits.float(),labels,weight=weights,reduction='none')
                    loss=per_case.mean()  # Do not cancel class weights when BATCH_SIZE=1.
                    if teacher_cache is not None:
                        teacher_logits=torch.stack([teacher_cache[k] for k in batch['keys']]).to(DEVICE)
                        temperature=cfg['kd_temperature']
                        kd=F.kl_div(F.log_softmax(logits.float()/temperature,dim=-1),
                            F.softmax(teacher_logits.float()/temperature,dim=-1),reduction='batchmean')*temperature**2
                        loss=(1-cfg['kd_weight'])*loss+cfg['kd_weight']*kd
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss; latest saved checkpoint is preserved.')
                scaler.scale(loss*case_fraction).backward(); group_loss+=loss.detach().item()*case_fraction
                progress.set_postfix(loss=f'{loss.item():.4f}',step=step,lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
                if next_batch%cfg['grad_accum']==0 or next_batch==n_batches:
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                    previous=scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale()>=previous:
                        scheduler.step(); step+=1
                        history.append({'kind':'train','epoch':epoch+1,'step':step,'loss':group_loss,'lr':optimizer.param_groups[0]['lr']})
                        if step%SAVE_EVERY_STEPS==0: persist(epoch,next_batch)
                    group_loss=0.0
                    if cfg['max_optimizer_steps'] is not None and step>=cfg['max_optimizer_steps']: capped=True; break
            progress.close()
            keys,dev_logits=predict_logits(model,'dev',cfg)
            p=probability_vector(dev_logits); gold=[DATA['dev'][key]['label'] for key in keys]
            threshold,metrics,baseline=optimal_threshold(p,gold,cfg['tune_threshold'])
            history.append({'kind':'dev','epoch':epoch+1,'step':step,'macro_f1':metrics['macro_f1'],
                'precision':metrics['precision'],'recall':metrics['recall'],'baseline_macro_f1':baseline['macro_f1'],'threshold':threshold})
            print(f"Dev macro-F1={metrics['macro_f1']:.6f}; precision={metrics['precision']:.6f}; recall={metrics['recall']:.6f}; threshold={threshold:.6f}")
            if metrics['macro_f1']>best_f1:
                best_f1=metrics['macro_f1']; bad_epochs=0
                save_torch({'model':model.state_dict(),'run_config':cfg,'metadata':metadata,'epoch':epoch+1,'step':step},output/'best_model.pt')
                save_json({'threshold':threshold,'metrics':metrics,'baseline_metrics':baseline},output/'best_dev_metrics.json')
            else: bad_epochs+=1
            finished=next_batch==n_batches
            early=finished and cfg['patience']>0 and bad_epochs>=cfg['patience']
            complete=finished and (epoch+1==cfg['epochs'] or early)
            persist(epoch+1 if finished else epoch,0 if finished else next_batch)
            if complete or capped: break
    if not (output/'best_model.pt').is_file(): raise RuntimeError('No evaluated checkpoint: increase the smoke-test cap or train normally.')
    state=torch_load(output/'best_model.pt'); model.load_state_dict(state['model'],strict=True)
    best_epoch=state['epoch']; del state
    keys,dev_logits=predict_logits(model,'dev',cfg,'Verifying reloaded best checkpoint on DEV')
    p=probability_vector(dev_logits); gold=[DATA['dev'][key]['label'] for key in keys]
    threshold,metrics,baseline=optimal_threshold(p,gold,cfg['tune_threshold'])
    recorded=json.loads((output/'best_dev_metrics.json').read_text())
    assert metrics==recorded['metrics'], 'Reloaded checkpoint metrics differ from the selected epoch.'
    assert baseline==recorded['baseline_metrics'], 'Reloaded 0.5-threshold metrics differ.'
    save_probabilities(output,'dev',keys,p)
    dev_predictions=[{'id':DATA['dev'][key]['id'],'pred':int(value>=threshold)} for key,value in zip(keys,p)]
    validate_submission(dev_predictions,DATA['dev']); save_json(dev_predictions,output/'dev_prediction.json')
    reloaded=json.loads((output/'dev_prediction.json').read_text())
    reloaded_map=validate_submission(reloaded,DATA['dev'])
    assert binary_metrics(gold,[reloaded_map[key] for key in keys])==metrics
    errors=[{'id':DATA['dev'][key]['id'],'instance_key':key,'gold':DATA['dev'][key]['label'],'pred':int(value>=threshold),'p1':float(value),
             'text_start':DATA['dev'][key]['text'][:400],'text_end':DATA['dev'][key]['text'][-400:]} for key,value in zip(keys,p)
             if int(value>=threshold)!=DATA['dev'][key]['label']]
    save_json(errors,output/'dev_errors.json')
    metrics.update({'experiment':name,'run_name':run_name,'architecture':cfg['spec']['architecture'],'method':cfg['spec']['method'],
        'threshold':threshold,'baseline_macro_f1':baseline['macro_f1'],'training_complete':complete,'optimizer_steps':step,
        'best_epoch':best_epoch,'trainable_parameters_final':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_seconds':elapsed+time.time()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else 0,**metadata})
    save_json(metrics,output/'metrics.json'); save_json({'threshold':threshold,'rule':'p(class 1) >= threshold'},output/'threshold.json')
    show_scores(metrics,'Post-training DEV — reloaded best checkpoint')
    tr=[r for r in history if r['kind']=='train']; dv=[r for r in history if r['kind']=='dev']
    fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].plot([r['step'] for r in tr],[r['loss'] for r in tr])
    axes[0].set(xlabel='Optimizer step',ylabel='Training loss')
    axes[1].plot([r['step'] for r in dv],[r['macro_f1'] for r in dv],label='Selected threshold')
    axes[1].plot([r['step'] for r in dv],[r['baseline_macro_f1'] for r in dv],label='Threshold 0.5')
    axes[1].legend(); axes[1].set(xlabel='Optimizer step',ylabel='Dev macro-F1')
    fig.tight_layout(); fig.savefig(output/'training_curves.png',dpi=150); plt.show(); plt.close(fig)
    fig,ax=plt.subplots(); ax.imshow(metrics['confusion_matrix'],cmap='Blues')
    for a in [0,1]:
        for b in [0,1]: ax.text(b,a,str(metrics['confusion_matrix'][a][b]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Gold',title='Dev confusion matrix')
    fig.tight_layout(); fig.savefig(output/'confusion_matrix.png',dpi=150); plt.show(); plt.close(fig)
    test_keys,test_logits=predict_logits(model,'test',cfg); test_p=probability_vector(test_logits)
    save_probabilities(output,'test',test_keys,test_p); predictions=save_predictions(output,test_p,threshold)
    verification={'checkpoint_reload_macro_f1_matches':True,'dev_json_metrics_roundtrip':True,
        'one_prediction_per_test_case':True,'original_ids_and_types':True,'integer_binary_predictions':True,
        'zip_root_and_json_match':True,'train_dev_test_separate':True,'test_cases':len(predictions),
        'training_complete':complete,'official_test_score':None,'score_scope':'official DEV used for model and threshold selection'}
    save_json(verification,output/'verification.json'); print('Verification PASS. Saved:',output/'submission.zip')
    if not complete: print('Capped smoke-test artifact; not eligible as a completed experiment/teacher.')
    del model,optimizer,teacher_cache; gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    return metrics
''')
md('## 12. Run the selected/core/extra experiments and optional bounded tuning sweep')
code('''
selected=list(CORE_EXPERIMENTS) if RUN_ALL else [EXPERIMENT]
if RUN_EXTRA_APPROACHES:
    selected=list(dict.fromkeys(selected+list(EXTRA_EXPERIMENTS)))
results=[]
for name in tqdm(selected,desc='CJPE approaches',unit='run'):
    results.append(run_experiment(name))
if RUN_TUNING:
    for trial in tqdm(TUNING_TRIALS,desc='Bounded hyperparameter trials',unit='trial'):
        results.append(run_experiment(TUNE_BASE_EXPERIMENT,trial['overrides'],TUNE_BASE_EXPERIMENT+'__'+trial['name']))
''')
md('''
## 13. Optional saved-probability ensemble and final verification

Uses only compatible **completed** runs. It compares the best single model with a small number of
weighted pairs among the top models, selecting thresholds/weights on DEV. It retains the best single
model if no pair improves dev macro-F1. This is an extra inference approach, not another PEFT method
for minimum-method counts. Repeated dev selection can overestimate private-test generalization.
''')
code('''
def load_probability_file(path,split):
    rows=json.loads((path/(split+'_probabilities.json')).read_text())
    mapping={r.get('instance_key',case_key(r['id'])):r['p1'] for r in rows}
    if len(mapping)!=len(rows) or set(mapping)!=set(DATA[split]): raise ValueError('Probability cache IDs disagree.')
    for row in rows:
        key=row.get('instance_key',case_key(row['id']))
        original=DATA[split][key]['id']
        if type(row['id'])!=type(original) or row['id']!=original: raise ValueError('Probability cache original ID changed.')
    values=np.asarray([mapping[k] for k in DATA[split]],dtype=float)
    if not np.isfinite(values).all() or not ((0<=values)&(values<=1)).all(): raise ValueError('Invalid cached probabilities.')
    return values

def build_ensemble():
    candidates=[]
    for file in PROJECT_DIR.glob('*/metrics.json'):
        path=file.parent
        if not (path/'run_config.json').is_file() or not (path/'dev_probabilities.json').is_file() or not (path/'test_probabilities.json').is_file(): continue
        cfg=json.loads((path/'run_config.json').read_text()); metrics=json.loads(file.read_text())
        if cfg['data_fingerprint']!=DATA_FINGERPRINT or cfg['tokenizer_hash']!=TOKENIZER_HASH or not metrics['training_complete']: continue
        candidates.append((metrics['macro_f1'],str(path)))
    candidates=sorted(candidates,reverse=True)[:ENSEMBLE_TOP_K]
    if len(candidates)<2:
        print('Ensemble skipped: at least two compatible completed runs are needed.'); return None
    paths=[Path(p) for _,p in candidates]
    dev=[load_probability_file(p,'dev') for p in paths]; test=[load_probability_file(p,'test') for p in paths]
    gold=[d['label'] for d in DATA['dev'].values()]
    threshold,metrics,_=optimal_threshold(dev[0],gold,TUNE_THRESHOLD)
    best=(metrics,[str(paths[0])],[1.0],threshold,test[0]); trials=[]
    for a in range(len(paths)):
        for b in range(a+1,len(paths)):
            for weight in ENSEMBLE_PAIR_WEIGHTS:
                blended=weight*dev[a]+(1-weight)*dev[b]
                t,m,_=optimal_threshold(blended,gold,TUNE_THRESHOLD)
                trials.append({'sources':[str(paths[a]),str(paths[b])],'weights':[weight,1-weight],'threshold':t,'macro_f1':m['macro_f1']})
                if m['macro_f1']>best[0]['macro_f1']+1e-12:
                    best=(m,[str(paths[a]),str(paths[b])],[weight,1-weight],t,weight*test[a]+(1-weight)*test[b])
    metrics,sources,weights,threshold,probabilities=best
    output=PROJECT_DIR/('ensemble_'+RUN_TAG); output.mkdir(exist_ok=True)
    save_json({'sources':sources,'weights':weights,'threshold':threshold,'data_fingerprint':DATA_FINGERPRINT,
               'selected_on':'dev','trials':trials},output/'ensemble_recipe.json')
    save_json(metrics,output/'metrics.json'); save_predictions(output,probabilities,threshold)
    show_scores(metrics,'Selected ensemble/single model on DEV'); print('Saved:',output/'submission.zip')
    return metrics

if BUILD_ENSEMBLE: ensemble_metrics=build_ensemble()
for result in results:
    output=run_path(result['run_name'])
    show_scores(json.loads((output/'metrics.json').read_text()),result['run_name'])
    print('Verification:',json.loads((output/'verification.json').read_text()))
    with zipfile.ZipFile(output/'submission.zip') as archive:
        assert archive.namelist()==['prediction.json']
        predictions=json.loads(archive.read('prediction.json'))
    validate_submission(predictions,DATA['test'])
''')
md('## 14. How to tune in Colab and compare approaches')
md('''
**Run all cells once** with `decoder_full` to confirm actual dataset compatibility, resource use and dev
scores. Then keep the same tag/settings and set `RUN_ALL=True`. These six runs already perform the
required fine-tuning/PEFT/distillation. You do not need a separate manual adaptation step.

For more approaches, set `RUN_EXTRA_APPROACHES=True`. For automatic finite tuning, set
`RUN_TUNING=True` and choose `TUNE_BASE_EXPERIMENT`. Edit `TUNING_TRIALS` to set your compute budget.
This sweep really trains the specified trials; it is not merely a decoder search. No unrestricted
search is claimed. The initial approach list and each trial use distinct folders, preserving comparisons.

| Section 3 control | Meaning / useful experiment |
|---|---|
| `EXPERIMENT`, `RUN_ALL` | One profile vs the six core methods |
| `RUN_EXTRA_APPROACHES` | Add six full/adapter/probing/gradual alternatives |
| `RUN_TUNING`, `TUNING_TRIALS` | Explicit bounded automatic hyperparameter/representation trials |
| `EXPERIMENTS[name]['lr']` | Backbone/adapter learning rate; override after dictionary definition |
| `HEAD_LR_MULTIPLIER` | Relative speed of new chunk/document/classification heads |
| `MAX_CHUNKS` | 4 vs 8/12 sampled windows; 0 for complete window coverage |
| `CHUNK_LENGTH`, `CHUNK_OVERLAP` | Longer context vs overlap; <= positional embedding limit |
| `CHUNK_BATCH_SIZE` | Chunk microbatch memory control; all selected chunks still contribute |
| `TRAIN_CHUNK_POLICY`, `EVAL_CHUNK_POLICY` | Head, tail, head+tail, spread, full-coverage comparison |
| `DOC_POOLING` | Mean, max, mean+max, learned attention, or chunk Transformer |
| `BATCH_SIZE`, `GRAD_ACCUM_STEPS` | Effective case batch = product; do not confuse with chunk batch |
| `EPOCHS`, `EARLY_STOPPING_PATIENCE` | Duration with best-dev selection; more epochs need not help |
| `USE_CLASS_WEIGHTS`, `CLASS_WEIGHT_POWER` | Train-only outcome balancing; compare with unweighted |
| `ADAPTER_RANK`, `ADAPTER_ALPHA`, `BOTTLENECK_DIM` | LORMA rank/scaling or adapter capacity |
| `PARTIAL_LAYERS`, `UNFREEZE_LAYERS_PER_EPOCH` | Partial/gradual adaptation depth |
| `KD_WEIGHT`, `KD_TEMPERATURE` | Hard-label vs soft-teacher supervision; cached teacher can be reused |
| `TEACHER_RUN_DIR` | Explicit larger completed teacher, even from another run tag |
| `TUNE_THRESHOLD`, `BUILD_ENSEMBLE` | Dev-only threshold calibration and bounded pair ensembles |

Example fresh KD context experiment after completing a teacher: set `EXPERIMENT='lightweight_kd'`,
`RUN_ALL=False`, `RUN_TAG='kd_chunks12'`, `MAX_CHUNKS=12`. The teacher may use eight chunks; the cache
maps training case IDs to its logits. Both student and teacher still originate from Phase 1 weights.

For a manual changed setting, use a fresh `RUN_TAG` and rerun Section 3 onward. For resumption, keep
all settings/tag identical. A new epoch count under a fresh tag retrains from Phase 1 with a new schedule;
it does not extend an exhausted optimizer schedule. The trial mechanism writes overrides into each saved
`run_config.json`; direct global changes under an existing run will be rejected.

Every run saves: `run_config.json`, `model_metadata.json`, `chunk_coverage.json`, `class_weights.json`,
`last.pt`, `best_model.pt`, `history.json`, `best_dev_metrics.json`, `metrics.json`, `threshold.json`,
`dev_prediction.json`, `dev_probabilities.json`, `dev_errors.json`, plots, `test_probabilities.json`,
`verification.json`, `prediction.json`, `submission.zip`. Teacher caches live in their source run folder.
Ensemble recipes record all source folders and weights. Copying only an ensemble ZIP is sufficient for
submission, but reproducing it requires those source models or their saved probabilities.

Compare **dev macro-F1** on the same official split and inspect class-level errors/coverage. Never invent
test labels or tune thresholds on test labels. Your actual leaderboard result comes from Codabench.
Keep the course-required LLM declaration, citations and your own experimentally supported conclusions.
''')
md('## 15. Experiment comparison and requirement coverage')
code('''
summary=[]; digests={}
for file in PROJECT_DIR.glob('*/metrics.json'):
    output=file.parent
    if not (output/'run_config.json').is_file(): continue
    cfg=json.loads((output/'run_config.json').read_text())
    if cfg['data_fingerprint']!=DATA_FINGERPRINT: continue
    m=json.loads(file.read_text()); summary.append(m)
    if m['training_complete'] and (output/'prediction.json').is_file():
        canonical=json.dumps(json.loads((output/'prediction.json').read_text()),sort_keys=True)
        digests[str(output)]=hashlib.sha256(canonical.encode()).hexdigest()
completed=[r for r in summary if r['training_complete']]
methods={r['method'] for r in completed}; architectures={r['architecture'] for r in completed}
coverage={'six_completed_runs':len(completed)>=6,'six_distinct_prediction_contents':len(set(digests.values()))>=6,
    'all_four_phase1_origins':architectures==set(MODEL_CLASSES),
    'three_distinct_adaptation_methods':len(methods-{'kd','linear_probe'})>=3,
    'four_listed_approaches':len(methods & {'full','partial','gradual','bitfit','lorma','adapters','kd'})>=4,
    'lorma_completed':'lorma' in methods,'distillation_completed':'kd' in methods}
save_json(coverage,PROJECT_DIR/'coverage.json')
import csv
columns=['run_name','architecture','method','macro_f1','baseline_macro_f1','precision','recall','threshold','best_epoch','training_complete']
with open(PROJECT_DIR/'experiment_summary.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
    for row in sorted(summary,key=lambda r:r['macro_f1'],reverse=True):
        writer.writerow({k:row.get(k) for k in columns})
        print(row['run_name'],f"DEV macro-F1={row['macro_f1']:.6f}", 'completed=',row['training_complete'])
print('Coverage across compatible saved run tags (not proof of Codabench acceptance):',coverage)
''')
md('''
References: `Capstone Document_edited.pdf` §§1/10/19 for methods, data and prediction format;
professor's `CS781_Phase1_Starter_Code.ipynb` for workflow; your Phase 1 solution for compatible model
definitions; verified Phase 3 code for strict checkpoint loading and RNG handling.

- [LoRMA](https://aclanthology.org/2025.findings-acl.527/), additive rank inflation (Eq. 8).
- [Adapters](https://arxiv.org/abs/1902.00751); this notebook uses one post-block residual bottleneck.
- [BitFit](https://arxiv.org/abs/2106.10199).
- [Knowledge distillation](https://arxiv.org/abs/1503.02531).

No performance guarantee is implied by these references. Inspect actual dev scores and error reports.
''')
for i,c in enumerate(cells): c['id']=f'phase4-{i:02d}'
nb={'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
    'language_info':{'name':'python'},'colab':{'name':'CS781_Phase4_Solution_Colab.ipynb','provenance':[]}},'nbformat':4,'nbformat_minor':5}
output=ROOT/'Phase4'/'CS781_Phase4_Solution_Colab.ipynb'; output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Created',output,'cells',len(cells))
