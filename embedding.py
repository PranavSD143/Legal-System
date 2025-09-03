import json
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# --- Model and Global Setup ---
# Initialize the models for embedding and generation.
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed = AutoModel.from_pretrained("law-ai/InLegalBERT")

# --- Indexing Functions ---
def build_and_save_indices(articles_json_path="articles.json", semantic_index_path="semantic_index.pkl", bm25_index_path="bm25_index.pkl"):
    """
    Builds both the semantic and BM25 indices from the articles.json file
    and saves them as separate pickle files.
    """
    print("🔄 Building both semantic and BM25 indices from articles.json...")
    try:
        with open(articles_json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error: {articles_json_path} not found. Please ensure it exists.")
        return None, None

    # 1. Build Semantic Index (Vectors)
    semantic_index = []
    def get_embedding(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embed(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return emb / np.linalg.norm(emb)

    for article in tqdm(articles, desc="Building Semantic Index"):
        text = article.get("text", "")
        tags = " ".join(article.get("tags", []))
        full_text = f"{text} {tags}"
        vec = get_embedding(full_text)
        semantic_index.append((vec, article))
    
    with open(semantic_index_path, "wb") as f:
        pickle.dump(semantic_index, f)
    print(f"✅ Semantic index saved to {semantic_index_path}.")

    # 2. Build BM25 Index
    corpus = [f"{article.get('text', '')} {' '.join(article.get('tags', []))}" for article in articles]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(bm25_index_path, 'wb') as f:
        pickle.dump((bm25, articles), f)
    print(f"✅ BM25 index saved to {bm25_index_path}.")

    return semantic_index, (bm25, articles)

# --- Loading Functions ---
def load_indices(semantic_index_path="semantic_index.pkl", bm25_index_path="bm25_index.pkl"):
    """
    Loads both indices. If not found, it calls the build function to create them.
    """
    try:
        with open(semantic_index_path, "rb") as f:
            semantic_index = pickle.load(f)
        with open(bm25_index_path, "rb") as f:
            bm25_index = pickle.load(f)
        print("✅ Indices loaded successfully.")
        return semantic_index, bm25_index
    except FileNotFoundError:
        print("Warning: Indices not found. Building them now...")
        return build_and_save_indices()
    

