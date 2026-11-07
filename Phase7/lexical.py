from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, split):
        self.ids = list(CANDIDATES[split]); self.position = {key: i for i, key in enumerate(self.ids)}
        self.vectorizer = CountVectorizer(max_features=BM25_MAX_FEATURES, stop_words=BM25_STOP_WORDS,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b|\b\d+\b', dtype=np.float32)
        texts = [CANDIDATES[split][key]['text'] for key in self.ids]
        try: counts = self.vectorizer.fit_transform(texts).tocsr()
        except ValueError as error: raise ValueError(f'{split}: empty lexical vocabulary; inspect candidate texts/stop-word settings.') from error
        lengths = np.asarray(counts.sum(1)).ravel(); mean = max(float(lengths.mean()), 1e-8)
        df = np.asarray((counts > 0).sum(0)).ravel()
        idf = np.log1p((len(self.ids) - df + 0.5) / (df + 0.5))
        weights = counts.tocoo(copy=True)
        tf = weights.data.copy()
        weights.data = (tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[weights.row] / mean))) * idf[weights.col]
        self.weights = weights.tocsr()
    def scores(self, text):
        query = self.vectorizer.transform([text]).tocsr()
        # Presence-only query terms avoid long-query repeated boilerplate dominating scores.
        query.data.fill(1.0)
        return (query @ self.weights.T).toarray().ravel().astype(np.float64)


BM25 = {}; LEXICAL_TOP = {}; BM25_BASELINE = {}; BM25_RANKS = {}
LEXICAL_SIGNATURE = {'k1': BM25_K1, 'b': BM25_B, 'max_features': BM25_MAX_FEATURES,
                     'stop_words': BM25_STOP_WORDS, 'hard_depth': HARD_NEGATIVE_DEPTH}
for split in ['train', 'dev', 'test']:
    print('Building split-specific BM25:', split)
    BM25[split] = BM25Index(split); LEXICAL_TOP[split] = {}; BM25_RANKS[split] = {}; predictions = {}
    for query, row in tqdm(QUERIES[split].items(), desc='BM25 ranking ' + split, unit='query'):
        scores = BM25[split].scores(row['text']); order = stable_order(scores, BM25[split].ids)
        BM25_RANKS[split][query] = order[:max(HARD_NEGATIVE_DEPTH, RRF_DEPTH)].astype(np.int32)
        ranked = [BM25[split].ids[i] for i in order if not EXCLUDE_SELF_MATCH or BM25[split].ids[i] != query]
        LEXICAL_TOP[split][query] = ranked[:HARD_NEGATIVE_DEPTH]
        predictions[query] = ranked[:TOP_K]
    BM25_BASELINE[split] = predictions
baseline_folder = PROJECT_DIR / 'bm25_baseline'; baseline_folder.mkdir(exist_ok=True)
bm25_metrics = retrieval_metrics(BM25_BASELINE['dev'])
save_json(bm25_metrics, baseline_folder / 'metrics.json'); save_json(BM25_BASELINE['dev'], baseline_folder / 'dev_prediction.json')
save_submission(baseline_folder, BM25_BASELINE['test'])
show_scores(bm25_metrics, 'BM25 baseline — no pretrained external model; does not count as a neural adaptation experiment')
