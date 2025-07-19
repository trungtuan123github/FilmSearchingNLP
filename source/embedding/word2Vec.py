# word2vec_pipeline.py
import numpy as np
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import random
from database_connector.qdrant_connector import connect_to_qdrant
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ---  Load sentences from MongoDB ---
def get_sentences():
    client = MongoClient(MONGO_URI)
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    data_list = list(collection.find({}, {"cleaned_description": 1}))
    return [doc['cleaned_description'] for doc in data_list if 'cleaned_description' in doc]

# --- Train Word2Vec and save model ---
def train_word2vec(sentences, embedding_dim=500, window_size=2, learning_rate=0.01, epochs=1, save_path='word2vec_embedding.pkl'):
    vocab = set(word for sent in sentences for word in sent)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    W = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))

    for _ in range(epochs):
        for sent in sentences:
            for idx, target in enumerate(sent):
                if target not in word2idx:
                    continue
                target_idx = word2idx[target]
                start = max(0, idx - window_size)
                end = min(len(sent), idx + window_size + 1)
                for context_pos in range(start, end):
                    if context_pos == idx:
                        continue
                    context_word = sent[context_pos]
                    if context_word not in word2idx:
                        continue
                    context_idx = word2idx[context_word]
                    error = W[target_idx] - W[context_idx]
                    W[target_idx] -= learning_rate * error
                    W[context_idx] += learning_rate * error

    with open(save_path, 'wb') as f:
        pickle.dump({'embedding': W, 'word2idx': word2idx, 'idx2word': idx2word}, f)

# --- Vector helpers ---
def get_vector(tokens, embedding_matrix, word2idx):
    vectors = [embedding_matrix[word2idx[t]] for t in tokens if t in word2idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_matrix.shape[1])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ---  Similarity matching ---
def find_similar_films(new_description, top_k=10, model_path='word2vec_embedding.pkl'):
    # Tải model embedding
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    embedding_matrix = model_data['embedding']
    word2idx = model_data['word2idx']

    # Tiền xử lý và tính embedding
    tokens = new_description.lower().split()
    query_vector = get_vector(tokens, embedding_matrix, word2idx).tolist()

    # Kết nối Qdrant
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_KEY")
    client = connect_to_qdrant(url, key)

    # Truy vấn Qdrant
    hits = client.search(
        collection_name="word2Vec",
        query_vector=query_vector,
        limit=top_k
    )

    return hits

# ---  Evaluation with Silhouette Score ---
def choose_k(n_samples):
    return max(2, int(np.sqrt(n_samples)))

if __name__ == '__main__':
    #  Load cleaned descriptions
    sentences = get_sentences()
    train_word2vec(sentences, embedding_dim=500, window_size=2, learning_rate=0.01, epochs=5, save_path='word2vec.pkl')
    #  Test similarity search
    print("\n Testing Similarity Search")
    query = "Thomas Brainerd, Sr., as a prospector, is a dutiful and loving husband and father. Two children, Gertrude and Thomas, Jr., are born while the Brainerds live in a log cabin in the mountains"
    find_similar_films(query, model_path='word2vec.pkl')  
