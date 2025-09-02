import json
import pickle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")


# 2. Load your articles.json
with open("articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# 3. Create embeddings
article_index = []

def get_embedding(text: str):
    """Generate embeddings using mean pooling."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling across tokens
    emb = outputs.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / np.linalg.norm(emb)
    return emb

for article in tqdm(articles, desc="Embedding Articles"):
    # Text + tags = better context for embeddings
    text = article["text"]
    tags = " ".join(article.get("tags", []))
    full_text = text + " " + tags

    # Vectorize
    vec = get_embedding(full_text)
    # Store (vector, metadata)
    article_index.append((vec, article))

# 4. Save as pickle for fast retrieval later
with open("tag_index.pkl", "wb") as f:
    pickle.dump(article_index, f)

print(f"✅ Indexed {len(article_index)} articles with Legal-BERT")
