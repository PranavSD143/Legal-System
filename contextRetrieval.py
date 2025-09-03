import json
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from embedding import load_indices
from sentence_transformers import CrossEncoder

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed = AutoModel.from_pretrained("law-ai/InLegalBERT")
model = OllamaLLM(model="openchat:latest")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def extract_query_components(query):
    """
    Uses an LLM to extract tags, article numbers, and part names from a query.
    """
    prompt = f"""
        You are a legal information extractor AI specializing in the Constitution of India.
        Your task is to process a user's natural language query and return ONLY a valid JSON object with the following structure:
        1. "tags": A list of 8 to 10 semantically relevant legal concepts. These must:
           - Be meaningful legal, constitutional, civic, or administrative concepts.
           - Be written in lowercase.
           - MUST NOT include article numbers or part names (e.g., "article 243A", "part IXA").
           - DO NOT at any cost generate tags or part or article_nums that aren't in the query given.
        2. "article_nums": A list of article numbers explicitly mentioned in the query (e.g., ["14", "15"]). If none are found, return an empty list.
        3. "part": The exact constitutional part name mentioned (e.g., "Part III"). If none is found, return null.
        You MUST return **only the JSON object**. No extra text, formatting, or keys.
        
        ### Output Format:
        {{
          "tags": ["tag1", "tag2", "... up to 10 tags ..."],
          "article_nums": ["14", "15"],
          "part": "Part III or whatever part the article belongs to"
        }}
        
        Now extract the data from this query:
        "{query}"
        """
    response = model.invoke(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode LLM response. Response was: {response}")
        return {"tags": [], "article_nums": [], "part": None}

def get_embedding(text: str):
    """
    Convert text to normalized embedding using InLegalBERT.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embed(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb / np.linalg.norm(emb)


def reciprocal_rank_fusion(semantic_results, bm25_results, k=80):
    """
    Combines ranked results from multiple search methods using Reciprocal Rank Fusion.
    k is a constant that impacts how much a high-ranking item is favored over
    many medium-ranking items. A common value is 60.
    """
    fused_scores = defaultdict(float)
    
    # Process semantic results
    for rank, (score, article) in enumerate(semantic_results):
        article_id = article.get("metadata").get("article")
        if article_id:
            fused_scores[article_id] += 1 / (rank + k)
            fused_scores[article_id] += score * 0.3 # Add a small bonus from the original score

    # Process BM25 results
    for rank, (score, article) in enumerate(bm25_results):
        article_id = article.get("metadata").get("article")
        if article_id:
            fused_scores[article_id] += 1 / (rank + k)
            fused_scores[article_id] += score * 0.05 # Smaller bonus from BM25 score

    # Re-rank based on the fused scores
    reranked_articles = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    
    # Match back to the full article objects
    article_map = {a[1].get("metadata", {}).get("article"): a[1] for a in (semantic_results + bm25_results)}
    final_results = [(score, article_map.get(doc_id)) for doc_id, score in reranked_articles if article_map.get(doc_id)]
    
    return final_results

def hybrid_retrieval(user_query, semantic_index, bm25_index, top_k=20):
    """
    The new core retrieval function. It performs both semantic and lexical searches
    and fuses the results for more robust context retrieval.
    """
    bm25, all_articles = bm25_index

    # 1. Semantic Search
    query_data = extract_query_components(user_query)
    query_tags = query_data['tags']
    print(query_tags)
    if not query_tags:
        print("Warning: No tags extracted for semantic search. Proceeding with BM25 only.")
        semantic_results = []
    else:
        query_vector = get_embedding(" ".join(query_tags))
        qv = np.asarray(query_vector).reshape(1, -1)
        
        semantic_scores = []
        for tag_vector, article in semantic_index:
            tv = np.asarray(tag_vector).reshape(1, -1)
            similarity = cosine_similarity(qv, tv)[0][0]
            semantic_scores.append((similarity, article))
        
        semantic_results = sorted(semantic_scores, key=lambda x: x[0], reverse=True)
        print("\n--- Semantic Search Results (Top 5) ---")
        for score, article in semantic_results[:5]:
            print(f"Article {article.get("metadata").get("article")}, Part {article.get('metadata', {}).get('part', 'N/A')}: Score {score:.4f}")
    
    # 2. Lexical (BM25) Search
    tokenized_query = user_query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_results = []
    for i, score in enumerate(bm25_scores):
        if score > 0: # Only include documents with a positive match
            bm25_results.append((score, all_articles[i]))
    
    bm25_results = sorted(bm25_results, key=lambda x: x[0], reverse=True)
    print("\n--- BM25 Search Results (Top 5) ---")
    for score, article in bm25_results[:5]:
        print(f"Article {article.get("metadata").get("article")}, Part {article.get('part', 'N/A')}: Score {score:.4f}")

    # 3. Fuse the results
    fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
    
    return fused_results[:top_k]


def cross_encoder_rerank(query, candidates, top_n=5):
    """
    Re-ranks a list of candidate documents using a cross-encoder model.
    """
    if not candidates:
        return []

    # Prepare the input for the cross-encoder
    queries_and_docs = [[query, article[1].get('text', '')] for article in candidates]
    
    # Get the relevance scores from the cross-encoder
    scores = reranker.predict(queries_and_docs)
    
    # Create a new list of (score, article) tuples based on the re-ranker's scores
    reranked_results = []
    for i, candidate in enumerate(candidates):
        reranked_results.append((scores[i], candidate[1]))
    
    # Sort by the new, more accurate scores
    reranked_results.sort(key=lambda x: x[0], reverse=True)
    
    return reranked_results[:top_n]

# --- Summarization and Pruning Functions ---

def prune_and_summarize(query, matched_results):
    """
    A simplified and more efficient summarization function.
    It now performs re-ranking and summarization in a single, powerful prompt.
    """
    if not matched_results:
        return {"relevant_articles": [], "summary": "I am sorry, but I could not find any relevant information to answer your query."}

    # Format all retrieved articles into a single context for the LLM
    context_string = ""
    for score, article in matched_results:
        article_num = article.get("metadata").get("article")
        part = article.get("metadata", {}).get("part", article.get("part", "Unknown"))
        text = article.get("text", "").strip()
        context_string += f"--- Article: {article_num}, Part: {part}, Score: {score:.2f} ---\n{text}\n\n"

    final_prompt = f"""
    You are a highly capable legal assistant. Your task is to analyze the user's query and the provided constitutional articles.
    
    First, identify and summarize the most relevant articles that directly address the user's query. If an article is not relevant, ignore it.
    
    Then, based on your analysis, provide a final, comprehensive answer to the user's query. The answer must be structured as follows:
    
    1. A section titled "Summary" which contains a concise, 3-4 sentence summary of the key findings. This summary should directly answer the query without listing article numbers.
    
    2. A section titled "Relevant Articles" which lists the article numbers that you used to form the summary.
    
    User Query: "{query}"
    
    Provided Articles:
    {context_string}
    
    Final Output:
    """

    response = model.invoke(final_prompt)
    
    # This part can be tricky without a structured API response. We'll parse the text
    # based on the expected format.
    try:
        summary_section = response.split("Summary:")[1].split("Relevant Articles:")[0].strip()
        articles_section = response.split("Relevant Articles:")[1].strip()
        return {
            "summary": summary_section,
            "relevant_articles": articles_section
        }
    except IndexError:
        print("Warning: Failed to parse LLM response. Returning raw response.")
        return {"summary": response, "relevant_articles": "Could not parse."}

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🔄 Loading or building indices...")
    # This is the single entry point. It will check for existing indices
    # and build them if they don't exist.
    semantic_index, bm25_index = load_indices()
    
    if not semantic_index or not bm25_index:
        print("❌ Could not load or create indices. Exiting.")
    else:
        query = "female seats in panchayat "
        print(f"\n🧠 Processing query: '{query}'")

        # Step 1: Perform hybrid retrieval
        hybrid_candidates = hybrid_retrieval(query, semantic_index, bm25_index)
        matched_results = cross_encoder_rerank(query, hybrid_candidates, top_n=5)
        
        print("\n✅ Top Matched Articles:")
        if not matched_results:
            print("No relevant articles found.")
        else:
            for score, article in matched_results:
                print(f"  - Article {article.get("metadata").get("article")}, Part {article.get('metadata', {}).get('part', 'N/A')}: Score {score:.4f}")

        # Step 2: Prune and summarize
        final_response = prune_and_summarize(query, matched_results)
        
        print("\n--- Final Answer ---")
        print("Summary:", final_response['summary'])
        print("Relevant Articles:", final_response['relevant_articles'])
