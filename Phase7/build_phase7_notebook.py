"""Build a standalone PCR Colab notebook. Prior executed notebooks are read only."""
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; HERE = Path(__file__).resolve().parent
phase5 = json.loads((ROOT / 'Phase5' / 'CS781_Phase5_Solution_Colab.ipynb').read_text())
phase6 = json.loads((ROOT / 'Phase6' / 'CS781_Phase6_Solution_Colab.ipynb').read_text())
def source(notebook, prefix):
    return next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))
cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip().splitlines(True)})
def code(s):
    s = s.strip()
    if not s.startswith('%'): ast.parse(s)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(True), 'execution_count': None, 'outputs': []})
md('''# CS781 Phase 7 — Prior Case Retrieval

Your Phase 1 backbones → relevance training → split-specific full-pool retrieval → verified top seven.
Follows §13 of `Capstone Document_edited.pdf` and the professor's starter progression: authentication,
configuration, data, model/adaptation, training, validation, saved predictions and ZIP.

All **six official files** are used. Train query relevance supervises fitting; dev selects checkpoints
and ranking policies; test supplies queries and its own candidate pool only. No external pretrained
retriever, embedding model or LLM API is used. BM25 is a provided-text lexical baseline.

Primary: **Micro-F1@7**, secondary: **Micro-Precision@7 and Micro-Recall@7**. The full gold set is
never truncated/padded to seven. Output is an object mapping each original query ID to **seven unique
original candidate ID strings**, ordered by predicted relevance and drawn only from that split's pool.

Six core profiles cover all four Phase 1 origins, compulsory LoRMA and smaller-student KD, plus full,
partial, prompt and IA³ adaptation. Eight optional profiles and twelve bounded training trials broaden
comparisons. Actual scores must come from runs; more experiments do not guarantee improvement.
''')
md('## 1. Install dependencies')
code('''%pip -q install "tokenizers>=0.20,<0.24" "huggingface_hub>=0.26,<3" "numpy>=1.26" "scipy>=1.11" "scikit-learn>=1.4" "tqdm>=4.66" matplotlib''')
md('## 2. Mount Drive, authenticate and check GPU')
setup = source(phase6, 'import os, json').replace('cs781_phase6', 'cs781_phase7').replace('Phase 6 output', 'Phase 7 output')
if 'import shutil' not in setup: setup += '\nimport shutil\n'
code(setup)
md('''## 3. All experiment and tuning controls — edit here

Start with `encoder_full`, `RUN_ALL=False`. After verifying the first result, `RUN_ALL=True` runs the
six core profiles, reusing completed matching runs. KD runs last and needs a larger completed Phase 7
teacher. Use a new `RUN_TAG` for changed manual settings. Section 12 contains the controls guide.
''')
code((HERE / 'controls.py').read_text())
md('''## 4. Download six files, preserve IDs, audit relevance and build token caches

Candidate pools remain separate, including when the same ID occurs in several pools. Sentence-list
texts are joined in order. Original leading zeros are preserved; an integer gold ID can resolve to a
zero-padded pool ID only if the numeric match is unambiguous. Every gold ID must belong to its pool.
Test relevance fields are ignored. Train queries with no positives are audited and excluded from
positive-supervised batching; all dev queries, including empty gold sets, remain in evaluation.
''')
old = source(phase6, 'def parse_case_file'); parser = old[:old.index('FILES=')]
download = '''FILES={
 'train_queries':'Phase_7_PCR/Train/train.json',
 'train_candidates':'Phase_7_PCR/Train/train_candidates.json',
 'dev_queries':'Phase_7_PCR/Dev/dev.json',
 'dev_candidates':'Phase_7_PCR/Dev/dev_candidates.json',
 'test_queries':'Phase_7_PCR/Test/test.json',
 'test_candidates':'Phase_7_PCR/Test/test_candidates.json'}
DOWNLOAD_REPORT={}
def load_download(key):
    for attempt in range(2):
        path=Path(hf_hub_download(repo_id=HF_DATASET_REPO,filename=FILES[key],repo_type='dataset',token=HF_TOKEN,force_download=attempt==1))
        print('Reading',key,path.stat().st_size,'bytes')
        try:
            raw,kind=parse_case_file(path)
            DOWNLOAD_REPORT[key]={'format':kind,'bytes':path.stat().st_size,'attempt':attempt+1}
            save_json(DOWNLOAD_REPORT,PROJECT_DIR/'dataset_loading_report.json')
            return raw
        except ValueError as error:
            DOWNLOAD_REPORT[key]={'error':str(error),'attempt':attempt+1}
            save_json(DOWNLOAD_REPORT,PROJECT_DIR/'dataset_loading_report.json')
            if attempt: raise ValueError(f'{key}: parsing failed after a fresh download: {error}') from error
            print('Retrying with fresh download:',key,error)
'''
code(parser + '\n' + download + '\n' + (HERE / 'data.py').read_text())
md('''## 5. Exact @7 metrics and split-specific BM25 baseline/hard negatives

BM25 is fitted separately on each candidate pool without relevance labels. The held-out baseline
helps determine whether learned representations add useful ranking information. It does not count
as a Phase 1 adaptation experiment. Hard negatives always come from TRAIN and exclude every known
positive, not just the sampled positive. Metrics count the complete gold relevance set.
''')
code((HERE / 'metrics.py').read_text()); code((HERE / 'lexical.py').read_text())
md('## 6. Checkpoint-compatible Phase 1 architectures and retrieval adaptation')
for prefix in ['class RMSNorm', 'class CausalSelfAttention', 'class RelativePositionBias', 'class RoPEAttention', 'class GQAAttention']:
    code(source(phase5, prefix))
code(source(phase5, 'def torch_load'))
code((HERE / 'model.py').read_text())
md('''## 7. Full-pool dense retrieval, chunk late interaction and dev-selected fusion

All candidates are encoded and searched, not just a gold-containing shortlist. Shared neural encoders
produce normalized embeddings. Optional asymmetric heads and chunk late interaction can be compared.
Bounded reciprocal-rank fusion compares dense/late-interaction and BM25 ranks on full dev. Selected
trained-model submissions retain a nonzero neural contribution. Optional exact IndianKanoon URL
features use only the supplied query text and are disabled by default; no web lookups are made.
''')
code((HERE / 'retrieval.py').read_text())
md('''## 8. Train, select on full dev, reload and verify

Uniform multi-positive contrastive loss, pairwise hardest-negative loss or their combination. In-batch
negatives are masked against every query's full known gold set. Optional train-only dense mining
refreshes hard negatives between epochs and is included in resume state. Query-weighted accumulation
handles short final batches. KD transfers a frozen larger teacher's candidate-score distribution on
train batches, with shared candidate IDs but potentially different chunking/dimensions.

Checkpoints select by full-dev Micro-F1@7 with dense-only Micro-F1 as a tie-breaker. The best model
is reloaded and rankings/metrics must match. JSON is reread and rescored; test IDs/pool membership,
seven-way uniqueness, ranking order and ZIP layout are checked. Test labels are never scored.
''')
code(source(phase6, 'def capture_rng')); code(source(phase6, 'def restore_rng'))
code((HERE / 'training.py').read_text())
md('## 9. Preflight checks')
code('''assert len(FILES)==6 and TOP_K==7
assert all(row['relevant'] is None for row in QUERIES['test'].values())
validate_predictions(BM25_BASELINE['dev'],'dev')
base=nn.Linear(7,12); x=torch.randn(2,3,7)
for adapter in [ProjectionAdapter(copy.deepcopy(base),3,6),IA3Projection(copy.deepcopy(base)),IA3Projection(copy.deepcopy(base),input_scale=True),IA3Projection(copy.deepcopy(base),fused_kv=True)]:
    torch.testing.assert_close(adapter(x),base(x))
    adapter(x).sum().backward()
    assert any(p.grad is not None for p in adapter.parameters() if p.requires_grad)
print('PASS: six files, split pools, test gold isolation, seven-ID schema and LoRMA/IA3 identity/gradients.')''')
md('## 10. Run selected/core/extra experiments and optional bounded training trials')
code('''selected=list(CORE_EXPERIMENTS) if RUN_ALL else [EXPERIMENT]
if RUN_EXTRA_APPROACHES: selected=list(dict.fromkeys(selected+list(EXTRA_EXPERIMENTS)))
results=[]
for name in tqdm(selected,desc='Phase 7 experiments',unit='run'):
    results.append(run_experiment(name))
if RUN_TUNING:
    for trial in tqdm(TUNING_TRIALS,desc='Bounded training trials',unit='trial'):
        results.append(run_experiment(TUNE_BASE_EXPERIMENT,trial['overrides'],TUNE_BASE_EXPERIMENT+'__'+trial['name']))''')
md('## 11. Experiment comparison and requirement coverage')
code((HERE / 'summary.py').read_text())
md('## 12. Colab controls, tuning and interpretation')
md((HERE / 'TUNING.md').read_text())
md('''References: local capstone PDF §13, professor's Phase 1 starter workflow, your Phase 1 architecture
definitions and strict loading, previous phase Drive/RNG handling. No previous phase's task classifier
is used as pretrained retrieval weights.

- [BM25 reference](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).
- [IA³ reference implementation description](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/ia3.md).
- [LoRMA](https://aclanthology.org/2025.findings-acl.527/), [prompt tuning](https://arxiv.org/abs/2104.08691).

BM25 uses a nonnegative log1p IDF variant and presence-only query terms. All settings are saved.
Keep the required LLM declaration and base conclusions on actual runs. No leaderboard score is promised.
''')
for index, cell in enumerate(cells): cell['id'] = f'phase7-{index:02d}'
nb = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python'}, 'colab': {'name': 'CS781_Phase7_Solution_Colab.ipynb', 'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'Phase7' / 'CS781_Phase7_Solution_Colab.ipynb'; path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('Created', path, 'cells:', len(cells))
