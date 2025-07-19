import numpy as np
from typing import List, Optional
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle

class TruncatedSVD:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        if X.size == 0:
            raise ValueError("Input matrix X is empty.")
        if self.n_components > X.shape[1]:
            raise ValueError(f"n_components ({self.n_components}) cannot be larger than number of features ({X.shape[1]}).")

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        if self.n_components is None:
            self.n_components = X.shape[1]

        self.components_ = VT[:self.n_components, :]
        self.singular_values_ = S[:self.n_components]
        total_var = np.sum(S ** 2)
        if total_var == 0:
            raise ValueError("Total variance is zero, cannot compute explained variance ratio.")
        comp_var = S[:self.n_components] ** 2
        self.explained_variance_ratio_ = comp_var / total_var
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(f"Input X has {X.shape[1]} features, expected {self.components_.shape[1]}.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def choose_n_components(self, threshold=0.95):
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        cum_var_ratio = np.cumsum(self.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var_ratio, threshold) + 1
        if n_components == 0:
            n_components = 1
        self.n_components = n_components
        self.components_ = self.components_[:n_components]
        self.singular_values_ = self.singular_values_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        return n_components

    def plot_cumulative_variance(self, threshold=0.95):
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Model must be fitted first!")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var)+1), cum_var, linestyle='-', marker='o', label='Cumulative Variance')
        if threshold is not None:
            n_component = self.choose_n_components(threshold)
            plt.axvline(x=n_component, color='blue', linestyle='--', label=f'Selected Components = {n_component}')
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold*100}%')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Explained Variance by SVD Components")
        plt.grid(True)
        plt.legend()
        plt.show()

class FastText:
    def __init__(self, vector_size=50, window_size=2, epochs=5, lr=0.01, min_count=1, neg_samples=5):
        self.vector_size = vector_size
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = []
        self.input_vectors = None
        self.output_vectors = None

    def build_vocab(self, corpus: List[List[str]]):
        if not corpus or not any(corpus):
            raise ValueError("Corpus is empty or contains no valid sentences.")
        word_freq = Counter(w for sentence in corpus for w in sentence)
        self.vocab = [w for w, c in word_freq.items() if c >= self.min_count]
        if not self.vocab:
            raise ValueError("Vocabulary is empty after applying min_count.")
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def generate_training_pairs(self, corpus: List[List[str]]):
        pairs = []
        for sentence in corpus:
            for i, center in enumerate(sentence):
                if center not in self.word2idx:
                    continue
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if i != j and sentence[j] in self.word2idx:
                        pairs.append((self.word2idx[center], self.word2idx[sentence[j]]))
        if not pairs:
            raise ValueError("No valid training pairs generated.")
        return pairs

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return np.clip(1 / (1 + np.exp(-x)), 1e-15, 1 - 1e-15)

    def train(self, corpus: List[List[str]]):
        self.build_vocab(corpus)
        vocab_size = len(self.vocab)
        self.input_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, self.vector_size))
        self.output_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, self.vector_size))
        training_pairs = self.generate_training_pairs(corpus)
        print(f"Generated {len(training_pairs):,} training pairs.")
        losses = []
        for epoch in tqdm(range(self.epochs), desc="Training FastText"):
            np.random.shuffle(training_pairs)
            loss = 0
            for center, context in training_pairs:
                v_in = self.input_vectors[center]
                v_out = self.output_vectors[context]
                score = self.sigmoid(np.dot(v_in, v_out))
                grad = self.lr * (1 - score)
                self.input_vectors[center] += grad * v_out
                self.output_vectors[context] += grad * v_in
                # Normalize vectors
                self.input_vectors[center] /= np.linalg.norm(self.input_vectors[center]) + 1e-10
                self.output_vectors[context] /= np.linalg.norm(self.output_vectors[context]) + 1e-10
                # Negative sampling
                neg_indices = np.random.choice(vocab_size, self.neg_samples)
                for neg in neg_indices:
                    v_neg = self.output_vectors[neg]
                    score_neg = self.sigmoid(np.dot(v_in, v_neg))
                    grad_neg = self.lr * score_neg
                    self.input_vectors[center] -= grad_neg * v_neg
                    self.output_vectors[neg] -= grad_neg * v_in
                    self.input_vectors[center] /= np.linalg.norm(self.input_vectors[center]) + 1e-10
                    self.output_vectors[neg] /= np.linalg.norm(self.output_vectors[neg]) + 1e-10
                    loss += -np.log(1 - score_neg + 1e-15)
                loss += -np.log(score + 1e-15)
            avg_loss = loss / len(training_pairs)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return losses

    def get_vector(self, word: str) -> np.ndarray:
        if word in self.word2idx:
            return self.input_vectors[self.word2idx[word]]
        print(f"Warning: Word '{word}' not in vocabulary, returning zero vector.")
        return np.zeros(self.vector_size)

class FastTextLSAEmbedder:
    def __init__(
        self,
        n_components: Optional[int] = 2,
        vector_size: int = 50,
        window: int = 2,
        epochs: int = 5,
        lr: float = 0.01,
        min_count: int = 1,
        neg_samples: int = 5,
        use_lsa: bool = True  # New flag to enable/disable LSA
    ):
        self.model = FastText(
            vector_size=vector_size,
            window_size=window,
            epochs=epochs,
            lr=lr,
            min_count=min_count,
            neg_samples=neg_samples
        )
        self.use_lsa = use_lsa
        self.lsa = TruncatedSVD(n_components) if use_lsa else None
        self.config = {
            "n_components": n_components if use_lsa else None,
            "vector_size": vector_size,
            "window": window,
            "epochs": epochs,
            "lr": lr,
            "min_count": min_count,
            "neg_samples": neg_samples,
            "use_lsa": use_lsa
        }

    def _preprocess(self, doc: str) -> List[str]:
        if not doc.strip():
            return []
        return doc.lower().split()

    def _embed_doc(self, tokens: List[str]) -> np.ndarray:
        vectors = np.array([self.model.get_vector(w) for w in tokens if w in self.model.word2idx])
        if vectors.size == 0:
            print(f"Warning: No valid words in document: {tokens}")
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def fit(self, documents: List[str], plot: bool = False):
        import time
        import numpy as np
        start_time = time.time()
        if not documents:
            raise ValueError("Documents list is empty.")
        tokenized = [self._preprocess(doc) for doc in documents]
        if not any(tokenized):
            raise ValueError("No valid tokens found in documents.")
        print("Training FastText model...")
        losses = self.model.train(tokenized)
        print("Generating document embeddings...")
        X = np.array([self._embed_doc(doc) for doc in tokenized])
        if X.shape[0] == 0 or np.all(X == 0):
            raise ValueError("Document embeddings are all zeros, cannot fit SVD.")
        
        training_time = time.time() - start_time
        if self.use_lsa:
            print("Fitting SVD...")
            self.lsa.fit(X)
            explained_variance = np.sum(self.lsa.explained_variance_ratio_)
        else:
            explained_variance = None

        # Print training statistics
        print("\nTraining Statistics:")
        print(f"Model Configuration: {self.config}")
        print(f"Number of Documents: {len(documents)}")
        print(f"Vocabulary Size: {len(self.model.vocab)}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Final FastText Loss: {losses[-1]:.4f}")
        if self.use_lsa:
            print(f"SVD Components: {self.lsa.n_components}")
            print(f"Explained Variance Ratio: {explained_variance:.4f}")
        else:
            print("LSA is disabled, using raw FastText embeddings.")

        if plot:
            print("Plotting results...")
            if self.use_lsa:
                self.lsa.plot_cumulative_variance()
                if self.lsa.n_components >= 2:
                    self.plot_document_vectors(documents)
            else:
                print("Skipping SVD cumulative variance plot (LSA disabled).")
                if self.model.vector_size >= 2:
                    self.plot_document_vectors(documents)
                else:
                    print("Cannot plot document vectors: vector_size must be at least 2.")
            self.plot_loss_curve(losses)
        return losses

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized = [self._preprocess(doc) for doc in documents]
        X = np.array([self._embed_doc(doc) for doc in tokenized])
        if self.use_lsa:
            if not self.lsa.fitted:
                raise ValueError("Model must be fitted first!")
            return self.lsa.transform(X)
        return X  # Return raw FastText embeddings if LSA is disabled

    def plot_document_vectors(self, documents: List[str]):
        if self.use_lsa and self.lsa.n_components < 2:
            print("Cannot plot document vectors: n_components must be at least 2.")
            return
        if not self.use_lsa and self.model.vector_size < 2:
            print("Cannot plot document vectors: vector_size must be at least 2.")
            return
        import matplotlib.pyplot as plt
        X_transformed = self.transform(documents)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='blue', alpha=0.6)
        for i, doc in enumerate(documents):
            plt.annotate(doc[:20], (X_transformed[i, 0], X_transformed[i, 1]), fontsize=9)
        plt.xlabel("Component 1" if self.use_lsa else "Dimension 1")
        plt.ylabel("Component 2" if self.use_lsa else "Dimension 2")
        plt.title("Document Vectors in 2D (SVD)" if self.use_lsa else "Document Vectors in 2D (FastText)")
        plt.grid(True)
        plt.show()

    def plot_loss_curve(self, losses):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses)+1), losses, marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("FastText Training Loss Curve")
        plt.grid(True)
        plt.show()

    def save_model(self, filename: str):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename: str):
        import pickle
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
