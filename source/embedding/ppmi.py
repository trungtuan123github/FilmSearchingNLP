import os
import sys
import time
import pickle
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Union, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def l2_normalize(matrix: np.ndarray, eps=1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, eps, norms)
    return matrix / norms


class TruncatedSVD:
    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.fitted = False

    def fit(self, X: sparse.csr_matrix) -> None:
        if not sparse.isspmatrix_csr(X):
            raise ValueError("Input must be sparse csr_matrix")

        U, S, VT = sparse.linalg.svds(X, k=self.n_components)
        order = np.argsort(-S)
        S, VT = S[order], VT[order]
        self.components_ = VT
        self.singular_values_ = S
        total_var = np.sum(S ** 2)
        comp_var = S ** 2
        self.explained_variance_ratio_ = comp_var / total_var
        self.fitted = True

    def transform(self, X: sparse.csr_matrix) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TruncatedSVD not fit.")
        return X @ self.components_.T

    def fit_transform(self, X: sparse.csr_matrix) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def choose_n_components(self, threshold: float = 0.95) -> int:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum_var, threshold) + 1)
        self.n_components = n_comp
        self.components_ = self.components_[:n_comp]
        self.singular_values_ = self.singular_values_[:n_comp]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_comp]
        return n_comp


class PPMIEmbedder:
    def __init__(self, window_size=6, max_features=5000, n_components=300, min_count=2, n_jobs=4):
        self.window_size = window_size
        self.max_features = max_features
        self.n_components = n_components
        self.min_count = min_count
        self.n_jobs = n_jobs

        self.vocab = {}
        self.idf = {}
        self.svd = TruncatedSVD(n_components)
        self.embeddings = None
        self.ppmi_sparse = None

    def _tokenize(self, doc: Union[str, List[str]]) -> List[str]:
        return doc.split() if isinstance(doc, str) else doc

    def _build_vocab(self, docs: List[Union[str, List[str]]]) -> None:
        counter = Counter()
        for doc in docs:
            counter.update(self._tokenize(doc))

        if self.min_count > 1:
            counter = Counter({w: c for w, c in counter.items() if c >= self.min_count})

        most_common = counter.most_common(self.max_features)
        self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        self._build_idf(docs)

    def _build_idf(self, docs: List[Union[str, List[str]]]) -> None:
        N = len(docs)
        df = Counter()
        for doc in docs:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                if token in self.vocab:
                    df[token] += 1
        self.idf = {w: np.log((N + 1) / (df[w] + 1)) + 1 for w in self.vocab}

    def _build_cooc_worker(self, docs_chunk):
        cooc = defaultdict(float)
        for doc in docs_chunk:
            tokens = self._tokenize(doc)
            token_ids = [self.vocab[t] for t in tokens if t in self.vocab]
            for i, center in enumerate(token_ids):
                start = max(0, i - self.window_size)
                end = min(len(token_ids), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context = token_ids[j]
                        weight = 1.0
                        cooc[(center, context)] += weight
        return cooc

    def _merge_cooc(self, cooc_list):
        merged = defaultdict(float)
        for cooc in cooc_list:
            for key, value in cooc.items():
                merged[key] += value
        return merged

    def _build_cooc_sparse_parallel(self, docs):
        chunks = np.array_split(docs, self.n_jobs)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._build_cooc_worker, chunks), total=len(chunks), desc="Building Cooc"))
        merged = self._merge_cooc(results)
        rows, cols, data = zip(*[(i, j, v) for (i, j), v in merged.items()])
        size = len(self.vocab)
        return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    def _calculate_ppmi(self, cooc_matrix: sparse.csr_matrix, eps=1e-12):
        total_sum = cooc_matrix.sum()
        row_sums = np.array(cooc_matrix.sum(axis=1)).flatten()
        col_sums = np.array(cooc_matrix.sum(axis=0)).flatten()

        coo = cooc_matrix.tocoo()
        p_ij = coo.data / total_sum
        p_i = row_sums[coo.row] / total_sum
        p_j = col_sums[coo.col] / total_sum
        pmi = np.log2((p_ij + eps) / (p_i * p_j + eps))

        ppmi = np.maximum(0, pmi)
        mask = ppmi > 0
        return sparse.coo_matrix(
            (ppmi[mask], (coo.row[mask], coo.col[mask])),
            shape=cooc_matrix.shape
        ).tocsr()

    def fit(self, docs: List[str]):
        self._build_vocab(docs)
        logging.info(f"Vocab size: {len(self.vocab)}")
        self.cooc_matrix = self._build_cooc_sparse_parallel(docs)
        self.ppmi_sparse = self._calculate_ppmi(self.cooc_matrix)
        self.embeddings = l2_normalize(self.svd.fit_transform(self.ppmi_sparse))

    def transform_docs(self, docs: List[str]) -> np.ndarray:
        dim = self.embeddings.shape[1]
        doc_vectors = np.zeros((len(docs), dim), dtype=np.float32)

        for idx, doc in enumerate(docs):
            tokens = self._tokenize(doc)
            weighted_sum = np.zeros(dim, dtype=np.float32)
            total_weight = 0.0
            for t in tokens:
                if t in self.vocab:
                    vec = self.embeddings[self.vocab[t]]
                    idf_weight = self.idf[t]
                    weighted_sum += vec * idf_weight
                    total_weight += idf_weight

            if total_weight > 0:
                doc_vectors[idx] = weighted_sum / total_weight

        return l2_normalize(doc_vectors)

    def transform(self, doc: Union[str, List[str]]) -> np.ndarray:
        if isinstance(doc, str):
            return self.transform_docs([doc])[0]
        return self.transform_docs(doc)


def train_ppmi(
    docs: List[str],
    max_features: int = 2000,
    window_size: int = 6,
    min_count: int = 2,
    n_components: int = 300,
    n_jobs: int = 4,
    save_path: Optional[str] = None
) -> PPMIEmbedder:

    logging.info(f"Starting PPMI training on {len(docs)} documents...")
    embedder = PPMIEmbedder(
        window_size=window_size,
        max_features=max_features,
        n_components=n_components,
        min_count=min_count,
        n_jobs=n_jobs
    )

    start_time = time.time()
    embedder.fit(docs)
    logging.info(f"PPMI shape: {embedder.ppmi_sparse.shape}")
    logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    save_path = save_path or "./embedding/trained_models/ppmi.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(embedder, f)
        logging.info(f"Embedder saved to: {save_path}")

    return embedder