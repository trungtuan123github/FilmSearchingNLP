
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from search_engine.search_logic import search_query
from embedding.FastText import FastTextLSAEmbedder, FastText, TruncatedSVD

load_dotenv()

app = Flask(__name__)

# Kết nối MongoDB
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["Film"]
collection = db["Data"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    model = data.get("model")
    hits = search_query(query, model)

    results = [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in hits
    ]

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000)