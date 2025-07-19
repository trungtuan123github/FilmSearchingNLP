# search_engine/search_logic.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from dotenv import load_dotenv
from database_connector.qdrant_connector import connect_to_qdrant, search_points
from preprocessing.preprocessing import WordEmbeddingPipeline, LSASVDPipeline
from embedding.FastText import FastTextLSAEmbedder, FastText, TruncatedSVD
from embedding.word2Vec import find_similar_films
from embedding.glove import GloVe



load_dotenv()

# Kết nối Qdrant và nạp mô hình + pipeline
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_KEY")
client = connect_to_qdrant(url, key)

import numpy as np

def embed_with_glove(tokens, embedder):
    vectors = []

    for token in tokens:
        if token in embedder['word2id']:
            idx = embedder['word2id'][token]
            vec = embedder['embeddings'][idx]
            vectors.append(vec)

    if not vectors:
        return np.zeros(embedder['embeddings'].shape[1])  # fallback

    return np.mean(vectors, axis=0)

import numpy as np

def embed_with_word2vec(tokens, embedder):
    embedding_matrix = embedder['embedding']
    word2idx = embedder['word2idx']

    vectors = []
    for token in tokens:
        if token in word2idx:
            idx = word2idx[token]
            vectors.append(embedding_matrix[idx])
    if not vectors:
        return np.zeros(embedding_matrix.shape[1])
    return np.mean(vectors, axis=0)



def search_query(query_text, model_name):
    with open(f"./trained_models/{model_name}.pkl", 'rb') as f:
        print(model_name)
        embedder = pickle.load(f)
        
    # Chọn pipeline tương ứng
    if model_name == "tfidf":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)
        embedded_query = embedder.transform_doc([processed_query])[0] # add [0] to make sure shape is (n, )
        results = search_points(client, model_name, embedded_query)

    elif model_name == "ppmi":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)  
        embedded_query = embedder.transform_docs([processed_query])[0]  
        results = search_points(client, model_name, embedded_query)
    elif model_name == "bow":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)
        embedded_query = embedder.transform([processed_query])[0] # add [0] to make sure shape is (n, )
        results = search_points(client, model_name, embedded_query)

    elif model_name == "hellinger_pca":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)
        embedded_query = embedder.transform_doc([processed_query])[0] # add [0] to make sure shape is (n, )
        results = search_points(client, model_name, embedded_query)
        
    elif model_name == "glove":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)  # list of tokens
        glove = GloVe.from_pretrained('./trained_models/glove.pkl')
        embedded_query = glove.encode(processed_query)  # vectorized query
        collection_name = "glove"
        results = search_points(client, collection_name, embedded_query)

    elif model_name == "fasttext":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)  # -> list các từ
        embedded_query = embedder.transform([" ".join(processed_query)])[0]  # sửa ở đây
        collection_name = "fastText"
        results = search_points(client, collection_name, embedded_query)    
    elif model_name == "word2vec":
        results = find_similar_films(query_text, model_path='./trained_models/word2vec.pkl')
    else:
        raise ValueError(f"Model chưa được hỗ trợ.")

    return results
def main():
    query = "A Billionaire discovers his true destiny after stumbling upon a haunted castle and fights to protect the castle's legacy."
    model_name = "ppmi"
    
    results = search_query(query, model_name)

    print("Kết quả tìm kiếm:")
    # print(results)
    for i, hit in enumerate(results, 1):
        film_name = hit.payload.get('metadata', {}).get('film_name', 'N/A')
        score = hit.score
        print(f"{i}. Tên phim: {film_name} | Score: {score:.4f}")
if __name__ == "__main__":
    main()
