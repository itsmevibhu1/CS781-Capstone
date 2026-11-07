"""Build standalone Phase 6 notebook. Existing executed notebooks are read only."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; HERE = Path(__file__).resolve().parent
template = json.loads((ROOT / 'Phase5' / 'CS781_Phase5_Solution_Colab.ipynb').read_text())
def source(prefix):
    return next(''.join(c['source']) for c in template['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))
cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip().splitlines(True)})
def code(s):
    s = s.strip()
    if not s.startswith('%'): ast.parse(s)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(True), 'outputs': [], 'execution_count': None})
md('''# CS781 Phase 6 — Abstractive Legal Summarization

Phase 1 weights → document-to-summary supervision → dev BERTScore/ROUGE → verified test summaries.
Follows the professor's starter workflow and §12 of `Capstone Document_edited.pdf`.
**Only two official files exist:** train and test. A fixed, grouped held-out split is derived from train.
No test references are inspected. Every test document receives a generated summary and original ID.

All four Phase 1 backbones are used for generation, including the full encoder–decoder's pretrained
decoder and language-model heads. No external pretrained summarizer or API is used.
The RoBERTa model loaded by BERTScore is **evaluation only** and never supplies training targets.
Defaults: six core experiments; ten optional profiles; eight training trials; six decoding comparisons.
Everything saves to **`MyDrive/cs781_phase6`**. All runtime code is embedded in this notebook.
''')
md('## 1. Install dependencies')
code('''%pip -q install "tokenizers>=0.20,<0.24" "huggingface_hub>=0.26,<1" "transformers>=4.45,<5" "bert-score==0.3.13" "rouge-score==0.1.2" "numpy>=1.26" "tqdm>=4.66" matplotlib sentencepiece''')
md('## 2. Drive, HF token and GPU')
setup = source('import os, json').replace('cs781_phase5', 'cs781_phase6').replace('Phase 5 output', 'Phase 6 output')
setup += '\nimport shutil\n'
code(setup)
md('''## 3. All controls — change here

Start with `EXPERIMENT='encoder_full'`, `RUN_ALL=False`. `RUN_ALL=True` runs the six core profiles,
with mandatory smaller-student KD last. Select any profile individually for separate Colab sessions.
Use separate notebook copies, finish shared token preparation once, and never run the same experiment/tag
in two sessions. A new manual setting needs a new `RUN_TAG`; matching settings resume automatically.
Section 12 explains tuning, validation subsets, generation limits and individual experiment names.
''')
code((HERE / 'controls.py').read_text())
md('''## 4. Download train/test, create a fixed held-out split and tokenize

No nonexistent dev file is requested. Duplicate source texts stay in the same split. Optional exact
train/test text overlap filtering removes training copies only; test rows remain. The split manifest
records every retained/excluded ID. Empty string sources get an UNK token. String-list documents and
summaries are joined in their original paragraph order. Missing/blank or unsupported training fields
are audited and excluded by default; no reference is invented. More than 5% unusable training rows
stops for schema inspection. Test rows are never skipped. `data_quality_report.json` and
`excluded_training_rows.json` give IDs, reasons and field types without copying document contents.
Shared binary token caches persist in Drive and stage to Colab local disk for fast reads.
''')
old_data = source('FILES =')
parse_start = old_data.index('def parse_case_file'); parse_stop = old_data.index('loading_report={}')
parser = old_data[parse_start:parse_stop].replace("'id' in header and 'text' in header", "'id' in header and ('document' in header or 'text' in header)")
download = '''FILES={'train':'Phase_6_Summ/Train/train.json','test':'Phase_6_Summ/Test/test.json'}
loading_report={}
def load_download(split):
    filename=FILES[split]
    for attempt in range(2):
        path=Path(hf_hub_download(repo_id=HF_DATASET_REPO,filename=filename,repo_type='dataset',token=HF_TOKEN,force_download=attempt==1))
        print('Reading',split,filename,path.stat().st_size,'bytes')
        try:
            raw,kind=parse_case_file(path)
            loading_report[split]={'format':kind,'bytes':path.stat().st_size,'attempt':attempt+1}
            save_json(loading_report,PROJECT_DIR/'dataset_loading_report.json')
            return raw
        except ValueError as error:
            loading_report[split]={'error':str(error),'bytes':path.stat().st_size,'attempt':attempt+1}
            save_json(loading_report,PROJECT_DIR/'dataset_loading_report.json')
            if attempt: raise ValueError(f'{split}: invalid dataset after fresh download: {error}') from error
            print('Retrying',split,'with a fresh download after:',error)
'''
code(parser + '\n' + download + '\n' + (HERE / 'data.py').read_text())
md('## 5. Phase 1 architecture definitions and generation/adaptation wrappers')
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention', 'class GQAAttention']:
    code(source(prefix))
original = source('def torch_load')
original = original[:original.index('def load_phase1')]
code(original)
code((HERE / 'model.py').read_text())
md('''## 6. Cached beam decoding and submission validation

The source prefix is computed once; each decoder step reuses past attention keys/values. Native
encoder states are reused throughout generation. Greedy/beam search are deterministic. Length,
repetition penalties and n-gram blocking are dev-tunable. Optional map–reduce generates intermediate
chunk summaries before a final generated summary; its extra inference cost is recorded by runtime.
''')
code((HERE / 'generation.py').read_text())
md('''## 7. Primary BERTScore F1 and secondary ROUGE-1/ROUGE-L F1

RoBERTa-large, layer 17, no IDF or baseline rescaling is the explicit default. Metric model/hash/settings
are saved. The PDF does not publish exact evaluator settings, so equality to its hidden scorer is not
claimed. ROUGE uses the specified Porter-stemming flag and mean per-document F1.
BERTScore model/tokenizer downloads are solely for scoring; they are never used by the summarizer.
No fallback silently substitutes ROUGE for the primary metric if BERTScore cannot load.
''')
code(source('def capture_rng')); code(source('def restore_rng'))
code((HERE / 'metrics.py').read_text())
md('''## 8. Train/adapt, select on fixed validation, reload and independently verify

Only summary tokens contribute to supervised loss. Source/padding labels are masked. Native seq2seq
targets and decoder-only concatenated targets are shifted consistently. LoRA/LoRMA/adapter/prompt
parameters actually train. KD matches a larger frozen Phase 6 teacher's next-token distributions on
training references only, with a smaller student retaining alternating Phase 1 lightweight layers.

Epoch selection uses a fixed validation subset by default. Final evaluation scores **all** held-out
documents when `FINAL_VALIDATION_LIMIT=0`; subset and full scores are kept separate. Interrupted
training checkpoints include optimizer/scheduler/scaler/RNG state and the exact next batch.
''')
code((HERE / 'training.py').read_text())
md('## 9. Preflight invariants')
code('''assert set(FILES)=={'train','test'}
assert all(row['summary'] is None for row in DATA['test'].values())
assert set(DATA['train']).isdisjoint(DATA['dev'])
assert not ({document_hash(row) for row in DATA['train'].values()} & {document_hash(row) for row in DATA['dev'].values()})
base=nn.Linear(7,11); x=torch.randn(2,3,7)
for multiplicative in [False,True]:
    adapter=LoRAProjection(copy.deepcopy(base),3,6,multiplicative)
    torch.testing.assert_close(adapter(x),base(x))
    adapter(x).sum().backward()
    assert adapter.B.weight.grad.abs().sum()>0 and adapter.base.weight.grad is None
assert clean_summary('Court allowed. Court allowed.',True)=='Court allowed.'
print('PASS: grouped split, private-test label isolation, LoRA/LoRMA identity and trainable gradients.')''')
md('## 10. Run selected/core/extra profiles and optional training/decoding searches')
code('''selected=list(CORE_EXPERIMENTS) if RUN_ALL else [EXPERIMENT]
if RUN_EXTRA_APPROACHES: selected=list(dict.fromkeys(selected+list(EXTRA_EXPERIMENTS)))
results=[]
for name in tqdm(selected,desc='Phase 6 experiments',unit='run'):
    results.append(run_experiment(name))
if RUN_TUNING:
    for trial in tqdm(TUNING_TRIALS,desc='Training trials',unit='trial'):
        results.append(run_experiment(TUNE_BASE_EXPERIMENT,trial['overrides'],TUNE_BASE_EXPERIMENT+'__'+trial['name']))
if RUN_DECODING_SEARCH:
    decoding_metrics=search_decoding(EXPERIMENT)''')
md('## 11. Compare all compatible runs and required architecture/method coverage')
code((HERE / 'summary.py').read_text())
md('## 12. How to tune and run in Colab')
md((HERE / 'TUNING.md').read_text())
md('''References: capstone PDF §12 and professor's Phase 1 starter for workflow; your Phase 1 model
definitions and checkpoint inference; Phase 5's robust input/Drive/RNG handling.

- [BERTScore implementation and reproducibility settings](https://github.com/Tiiiger/bert_score).
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).
- [LoRA](https://arxiv.org/abs/2106.09685), [LoRMA](https://aclanthology.org/2025.findings-acl.527/).
- [Houlsby adapters](https://arxiv.org/abs/1902.00751), [prompt tuning](https://arxiv.org/abs/2104.08691).

Adapter placement uses bottlenecks at the stated residual sites in the existing pre-norm Phase 1 blocks;
it is not a claim of reproducing every detail of the original papers. Keep the required LLM declaration.
Only actual generated validation/test results support performance conclusions; implementation tests do not.
''')
for i, cell in enumerate(cells): cell['id'] = f'phase6-{i:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
             'language_info': {'name': 'python'}, 'colab': {'name': 'CS781_Phase6_Solution_Colab.ipynb', 'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'Phase6' / 'CS781_Phase6_Solution_Colab.ipynb'; path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Created', path, '; cells:', len(cells))
