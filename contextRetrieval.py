import json
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from embedding import load_tag_index
from transformers import AutoTokenizer, AutoModel
import math

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
embed = AutoModel.from_pretrained("law-ai/InLegalBERT")
model = OllamaLLM(model="openchat:latest")

if __name__ == "__main__":
    print("🔄 Loading tag index and initializing models... please wait.")


def extract_query_components(query):
  
    prompt = f"""
        You are a legal information extractor AI specializing in the Constitution of India.

        Your task is to process a user's natural language query and return ONLY a valid JSON object with the following structure:

        1. "tags": A list of 8 to 10 **semantically relevant legal concepts**. These must:
           - Be meaningful legal, constitutional, civic, or administrative concepts.
           - Be loosely or indirectly inferable from the query even if not explicitly stated.
           - Be written in lowercase.
           - **MUST NOT include article numbers or part names** (e.g., "article 243A", "part IXA").
           - DO NOT at any cost generate tags or part or article_nums that aren't in the query given.

        2. "article_nums": A **list of article numbers** explicitly mentioned in the query (e.g., ["14", "15"]). If none are found, return an empty list.

        3. "part": The **exact constitutional part name** mentioned (e.g., "Part III"). If none is found, return null.

        You MUST return **only the JSON object**. No extra text, formatting, or keys.

        ### Output Format:
        {{
          "tags": ["tag1", "tag2", "... up to 10 tags ..."],
          "article_nums": ["14", "15"],
          "part": "Part III"
        }}

        Now extract the data from this query:
        "{query}"
        """

    response = model.invoke(prompt)
    return response

def get_embedding(text: str):
    """Convert text to normalized embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embed(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    # Normalize for cosine similarity
    return emb / np.linalg.norm(emb)

def filter_articles(user_query, tag_index, base_threshold=0.7, top_percent=0.4, min_k=2, max_k=15):
    query_data = json.loads(extract_query_components(user_query))
    query_tags = query_data['tags']
    part = query_data['part']
    article_nums = query_data['article_nums']

    PART_MATCH_WEIGHT = 1.2
    ARTICLE_MATCH_WEIGHT = 1.4

    results = []
    if not query_tags:
        return []

    # Stricter threshold for fewer tags (vague queries)
    tag_count = len(query_tags)
    adjusted_threshold = base_threshold - (0.05 * max(0, 10 - tag_count))

    # Embed query
    query_vector = get_embedding(" ".join(query_tags))
    qv = np.asarray(query_vector).reshape(1, -1)
    print("Query Tags:", query_tags)
    # print("Adjusted Threshold:", adjusted_threshold)

    # for tag_vector, article in tag_index:
    #     tv = np.asarray(tag_vector).reshape(1, -1)
    #     similarity = cosine_similarity(qv, tv)[0][0]
    #     article_tags = article.get("tags", [])
    #     tag_overlap = len(set(query_tags) & set(article_tags)) / max(len(set(query_tags)), 1)
    #     combined_score = 0.7 * similarity + 0.3 * tag_overlap

    #     print(f"Article {article['metadata'].get('article')} | Sim={similarity:.4f} | Overlap={tag_overlap:.2f} | Score={combined_score:.4f}")
  
    for tag_vector, article in tag_index:
        tv = np.asarray(tag_vector).reshape(1, -1)

        try:
            similarity = cosine_similarity(qv, tv)[0][0]
        except Exception:
            continue

        article_tags = article.get("tags", [])
        tag_overlap = len(set(query_tags) & set(article_tags)) / max(len(set(query_tags)), 1)

        combined_score = 0.85 * similarity + 0.15 * tag_overlap


        if part and article["metadata"].get("part") == part:
            combined_score *= PART_MATCH_WEIGHT
        if article_nums and article["metadata"].get("article") == article_nums:
            combined_score *= ARTICLE_MATCH_WEIGHT

        if combined_score >= adjusted_threshold:
            results.append((combined_score, article))

    # Apply relative cutoff
    if results:
        k = max(min_k, int(len(results) * top_percent))
        k = min(k, max_k)  # cap at max_k
        results = results[:k]
    return sorted(results, key=lambda x: x[0], reverse=True)

query = "female seats in gram panchayat ??"

tag_index = load_tag_index()
match = filter_articles(query, tag_index)
print(match)

def prune_and_summarize_batched(query, matched_results, model, batch_size=4):
    def format_batch_context(batch):
        articles_context = ""
        for score, article in batch:
            article_num = article["metadata"].get("article", "Unknown")
            part = article["metadata"].get("part", "Unknown")
            snippet = article.get("text", "").strip().replace("\n", " ")

            articles_context += f"- Article: {article_num}, Part: {part}, Score: {score:.2f}\n  Text: {snippet}\n\n"
        return articles_context

    all_relevant_articles = []

    total_batches = math.ceil(len(matched_results) / batch_size)

    for i in range(total_batches):
        batch = matched_results[i * batch_size:(i + 1) * batch_size]
        context = format_batch_context(batch)

        prompt = f"""
You are a legal assistant AI. Given a user query and a batch of matched articles, perform the following:

You are a legal assistant AI. Given a user query and a batch of matched articles, perform the following:

1. Only keep articles if they are *directly relevant* to the query. Relevance means the article explicitly addresses the subject matter in the user query. 
   Example: If the query is about judges, only keep articles mentioning judges, judicial accountability, impeachment, or removal.
2. If no articles match, return [].
3. The output must only be JSON with the following fields: article, part, text, reason.
4. Do not include any other fields or commentary.
5. If even part of the article is unrelated, discard it entirely.

User Query:
"{query}"

Matched Articles:
{context}

Output format:
{{
  "relevant_articles": [
    {{
      "article": "243A",
      "part": "Part IXA",
      "relevance":[True or False],
      "text":"Article 243A of the Indian Constitution establishes the Gram Sabha, which is a body consisting of persons registered in the electoral rolls for a village within a Panchayat area. The Gram Sabha exercises powers and performs functions at the village level as provided by the State Legislature through law, fostering grassroots-level governance and direct public participation in rural development."
      "reason": "Aligns with user's question on local governance and rights."
    }},
    ...
  ]
}}
"""

        response = model.invoke(prompt)

        prompt = f"""
You are a legal assistant AI. Given a user query and a batch of matched articles, perform the following:

You are a legal assistant AI. Given a user query and a batch of matched articles, perform the following:

1. Only keep articles if they are *directly relevant* to the query. Relevance means the article explicitly addresses the subject matter in the user query. 
   Example: If the query is about judges, only keep articles mentioning judges, judicial accountability, impeachment, or removal.
2. If no articles match, return [].
3. The output must only be JSON with the following fields: article, part, text, reason.
4. Do not include any other fields or commentary.
5. If even part of the article is unrelated, discard it entirely.
6. If the relevance value is False or if reason states that article is unwanted completely ignore that article

User Query:
"{query}"

Matched Articles:
{response}

Output format:
{{
  "relevant_articles": [
    {{
      "article": "243A",
      "part": "Part IXA",
      "text":"Article 243A of the Indian Constitution establishes the Gram Sabha, which is a body consisting of persons registered in the electoral rolls for a village within a Panchayat area. The Gram Sabha exercises powers and performs functions at the village level as provided by the State Legislature through law, fostering grassroots-level governance and direct public participation in rural development."
      "reason": "Aligns with user's question on local governance and rights."
    }},
    ...
  ]
}}
"""
        response = model.invoke(prompt)
        all_relevant_articles.append(response)
    # Final summarization step
    final_prompt = f"""
You are a summarization AI. Given the following relevant constitutional articles, write a concise summary of their key themes and how they relate to the user query.

User Query:
"{query}"

Relevant Articles:
{all_relevant_articles}

Return a summary only (3-4 sentences). Do not repeat article numbers or parts.
"""

    final_summary = model.invoke(final_prompt)

    return {
        "relevant_articles": all_relevant_articles,
        "summary": final_summary.strip()
    }

result = prune_and_summarize_batched(query, match, model)