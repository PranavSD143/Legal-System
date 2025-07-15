import json
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from embedding import load_tag_index
import math

if __name__ == "__main__":
    print("🔄 Loading tag index and initializing models... please wait.")
    embed = OllamaEmbeddings(model="nomic-embed-text:latest")
    model = OllamaLLM(model="openchat:latest")

def extract_query_components(query):
  
    prompt = f"""
        You are a legal information extractor AI specializing in the Constitution of India.

        Your task is to process a user's natural language query and return ONLY a valid JSON object with the following structure:

        1. "tags": A list of 8 to 10 **semantically relevant legal concepts**. These must:
           - Be meaningful legal, constitutional, civic, or administrative concepts.
           - Be loosely or indirectly inferable from the query even if not explicitly stated.
           - Be written in lowercase.
           - **MUST NOT include article numbers or part names** (e.g., "article 243A", "part IXA").

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


def filter_articles(user_query, tag_index, base_threshold=0.6):
    # Extract structured components from the query
    query_data = json.loads(extract_query_components(user_query))
    query_tags = query_data['tags']
    part = query_data['part']
    article_nums = query_data['article_nums']

    PART_MATCH_WEIGHT = 1.2
    ARTICLE_MATCH_WEIGHT = 1.4

    results = []

    # Handle empty or bad tag extraction
    if not query_tags:
        return []

    # Dynamic threshold adjustment based on tag count
    tag_count = len(query_tags)
    adjusted_threshold = base_threshold - (0.05 * max(0, 10 - tag_count))

    # Embed user query tags
    query_vector = embed.embed_query(" ".join(query_tags))
    qv = np.array(query_vector).reshape(1, -1)
    
    for tag_vector, article in tag_index:
        tv = np.array(tag_vector).reshape(1, -1)

        try:
            similarity = cosine_similarity(qv, tv)[0][0]
        except Exception as e:
            print("Similarity calculation failed:", e)
            continue

        # Optional: compute tag overlap score
        article_tags = article.get("tags", [])
        tag_overlap = len(set(query_tags) & set(article_tags)) / max(len(set(query_tags)), 1)

        # Combine both similarity + tag overlap
        combined_score = 0.7 * similarity + 0.3 * tag_overlap

        # Boost for part and article number matches
        if part and article["metadata"].get("part") == part:
            combined_score *= PART_MATCH_WEIGHT
        if article_nums and article["metadata"].get("article") == article_nums:
            combined_score *= ARTICLE_MATCH_WEIGHT

        if combined_score >= adjusted_threshold:
            results.append((combined_score, article))

    return sorted(results, key=lambda x: x[0], reverse=True)



query = "i want to know murder related information"

tag_index = load_tag_index()
match = filter_articles(query, tag_index)
# # print(f"Matches found: {len(match)}")
# for score, article in match:
#     article_num = article["metadata"].get("article", "Unknown")
#     part = article["metadata"].get("part", "Unknown")
#     preview = article.get("text", "")[:120].strip() + "..."
    
#     print(f"Score: {score:.2f} | Article: {article_num} | Part: {part}")
#     print(f"Text: {preview}")
#     print("-" * 80)


def prune_and_summarize_batched(query, matched_results, model, batch_size=4):
    def format_batch_context(batch):
        articles_context = ""
        for score, article in batch:
            article_num = article["metadata"].get("article", "Unknown")
            part = article["metadata"].get("part", "Unknown")
            snippet = article.get("text", "").strip().replace("\n", " ")
            snippet = snippet[:300] + "..." if len(snippet) > 300 else snippet

            articles_context += f"- Article: {article_num}, Part: {part}, Score: {score:.2f}\n  Text: {snippet}\n\n"
        return articles_context

    all_relevant_articles = []

    total_batches = math.ceil(len(matched_results) / batch_size)

    for i in range(total_batches):
        batch = matched_results[i * batch_size:(i + 1) * batch_size]
        context = format_batch_context(batch)

        prompt = f"""
You are a legal assistant AI. Given a user query and a batch of matched articles, perform the following:

1. **Filter out irrelevant articles** based on the query.
2. Return relevant articles with reasoning.
3. Provide a brief batch summary (1-2 sentences).

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
      "reason": "Aligns with user's question on local governance and rights."
    }},
    ...
  ],
  "batch_summary": "This batch includes articles on Gram Sabha powers and village administration."
}}
"""
        try:
            response = model.invoke(prompt)
            parsed = json.loads(response)
            all_relevant_articles.extend(parsed.get("relevant_articles", []))
        except Exception as e:
            print(f"Batch {i+1} failed:", e)
            continue

    # Final summarization step
    final_prompt = f"""
You are a summarization AI. Given the following relevant constitutional articles, write a concise summary of their key themes and how they relate to the user query.

User Query:
"{query}"

Relevant Articles:
{json.dumps(all_relevant_articles, indent=2)}

Return a summary only (3-4 sentences). Do not repeat article numbers or parts.
"""

    final_summary = model.invoke(final_prompt)

    return {
        "relevant_articles": all_relevant_articles,
        "summary": final_summary.strip()
    }

result = prune_and_summarize_batched(query, match, model)
print(json.dumps(result, indent=2))