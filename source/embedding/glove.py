import os
import numpy as np
from collections import defaultdict
from pymongo import MongoClient
from collections import Counter, defaultdict
import pickle


def get_sentences():
    client = MongoClient(os.getenv('MONGO_URI'))
    collection = client[os.getenv('DATABASE_NAME')][os.getenv('WEMB_COLLECTION_NAME')]
    data_list = list(collection.find({}, {"cleaned_description": 1}))
    return [doc['cleaned_description'] for doc in data_list if 'cleaned_description' in doc]


class GloVe:
    def __init__(self, vocab_size, embedding_dim=50, x_max=100, alpha=0.75, learning_rate=0.05, epochs=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        self.lr = learning_rate
        self.epochs = epochs
        self.word2id = {}
        self.id2word = {}

        # Embedding vectors and biases
        self.W = np.random.randn(vocab_size, embedding_dim) / np.sqrt(vocab_size)
        self.W_context = np.random.randn(vocab_size, embedding_dim) / np.sqrt(vocab_size)
        self.b = np.zeros(vocab_size)
        self.b_context = np.zeros(vocab_size)

        # For AdaGrad
        self.gradsq_W = np.ones((vocab_size, embedding_dim))
        self.gradsq_W_context = np.ones((vocab_size, embedding_dim))
        self.gradsq_b = np.ones(vocab_size)
        self.gradsq_b_context = np.ones(vocab_size)

    def weighting_func(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1

    def fit(self, cooccur_data):
        # cooccur_data: list of tuples (i, j, x_ij)
        for epoch in range(self.epochs):
            total_loss = 0
            for i, j, x_ij in cooccur_data:
                w_i = self.W[i]
                w_j = self.W_context[j]
                b_i = self.b[i]
                b_j = self.b_context[j]

                weight = self.weighting_func(x_ij)
                inner_prod = np.dot(w_i, w_j)
                cost = inner_prod + b_i + b_j - np.log(x_ij)
                loss = weight * (cost ** 2)
                total_loss += 0.5 * loss

                grad = weight * cost

                # Gradients
                grad_w_i = grad * w_j
                grad_w_j = grad * w_i
                grad_b_i = grad
                grad_b_j = grad

                # AdaGrad update
                self.W[i] -= (self.lr / np.sqrt(self.gradsq_W[i])) * grad_w_i
                self.W_context[j] -= (self.lr / np.sqrt(self.gradsq_W_context[j])) * grad_w_j
                self.b[i] -= (self.lr / np.sqrt(self.gradsq_b[i])) * grad_b_i
                self.b_context[j] -= (self.lr / np.sqrt(self.gradsq_b_context[j])) * grad_b_j

                # Update gradsq
                self.gradsq_W[i] += grad_w_i ** 2
                self.gradsq_W_context[j] += grad_w_j ** 2
                self.gradsq_b[i] += grad_b_i ** 2
                self.gradsq_b_context[j] += grad_b_j ** 2

            print(f"Epoch {epoch+1}/{self.epochs} loss: {total_loss:.4f}")

    def get_embeddings(self):
        # Trả về tổng embeddings của word và context word
        return self.W + self.W_context
    
    @staticmethod
    def from_pretrained(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        word2id = data["word2id"]
        id2word = data["id2word"]
        embeddings = data["embeddings"]
        vocab_size = len(word2id)

        model = GloVe(vocab_size=vocab_size, embedding_dim=embeddings.shape[1])
        model.W = embeddings / 2  # assume W + W_context = embeddings
        model.W_context = embeddings / 2
        model.word2id = word2id
        model.id2word = id2word
        model.embeddings = embeddings
        return model

    def encode(self, text):
        tokens = [w.lower() for w in text if isinstance(w, str)]
        vectors = [self.embeddings[self.word2id[w]] for w in tokens if w in self.word2id]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embeddings.shape[1])


def build_vocab(sentences, min_count=1):
    # Đếm tần suất từ
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    # Lọc từ có tần suất nhỏ hơn min_count
    vocab = {w for w, c in word_counts.items() if c >= min_count}
    word2id = {w: i for i, w in enumerate(sorted(vocab))}
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def build_cooccur_matrix(sentences, word2id, window_size=5):
    cooccur = defaultdict(float)
    for sent in sentences:
        sent_ids = [word2id[w] for w in sent if w in word2id]
        for center_i, center_id in enumerate(sent_ids):
            start = max(0, center_i - window_size)
            end = min(len(sent_ids), center_i + window_size + 1)
            for context_i in range(start, end):
                if context_i != center_i:
                    context_id = sent_ids[context_i]
                    distance = abs(context_i - center_i)
                    # Weight by inverse distance
                    cooccur[(center_id, context_id)] += 1.0 / distance
    # Chuyển sang list (i, j, x_ij)
    cooccur_data = [(i, j, x_ij) for (i, j), x_ij in cooccur.items()]
    return cooccur_data

if __name__ == "__main__":
    sentences = get_sentences()

    print("Building vocabulary...")
    word2id, id2word = build_vocab(sentences, min_count=1)
    print(f"Vocabulary size: {len(word2id)}")

    print("Building co-occurrence matrix...")
    cooccur_data = build_cooccur_matrix(sentences, word2id, window_size=5)
    print(f"Co-occurrence pairs: {len(cooccur_data)}")

    print("Training GloVe model...")
    model = GloVe(vocab_size=len(word2id), embedding_dim=50, epochs=100)
    model.fit(cooccur_data)

    embeddings = model.get_embeddings()
    print("Saving embeddings to glove_embeddings.pkl...")
    with open("./embedding/trained_models/glove.pkl", "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "word2id": word2id,
            "id2word": id2word
        }, f)

    print("Done.")