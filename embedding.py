import json
import pickle 
import numpy as np

def save_tag_index(tag_index, path="tag_index.pkl"):
    with open(path, "wb") as f:
        pickle.dump(tag_index, f)
    print(f"Tag index saved to {path}")

def load_articles(file_path = 'articles.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles

def load_tag_index(path="tag_index.pkl"):
    with open(path, "rb") as f:
        tag_index = pickle.load(f)
    print(f"Tag index loaded from {path}")
    return tag_index