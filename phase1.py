from contextRetrieval import query
from contextRetrieval import match
from contextRetrieval import model
import math


def prune_and_summarize_batched(query, matched_results, model, batch_size=4):
    summarized_batch_texts = []

    def get_batch_text(batch):
        full_text = ""
        for _, article in batch:
            article_text = article.get("text", "").strip().replace("\n", " ")
            full_text += article_text + "\n\n"
        return full_text.strip()

    total_batches = math.ceil(len(matched_results) / batch_size)

    for i in range(total_batches):
        batch = matched_results[i * batch_size:(i + 1) * batch_size]
        batch_text = get_batch_text(batch)

        summarization_prompt = f"""
You are a legal summarization assistant. Given the full text of several constitutional articles and a user query, summarize the key legal ideas and rights in the batch in a clear, accessible way.

Only return a concise summary (max 5 sentences) of the legal content relevant to the user query. Avoid repeating exact article numbers unless essential.

User Query:
\"\"\"{query}\"\"\"

Article Texts:
\"\"\"{batch_text}\"\"\"

Summary:
"""
        try:
            summary = model.invoke(summarization_prompt).strip()
            summarized_batch_texts.append(summary)
        except Exception as e:
            print(f"Batch {i+1} summarization failed:", e)
            continue

    combined_summaries = "\n".join(summarized_batch_texts)

    final_prompt = f"""
You are a summarization assistant. Given previous summaries of article batches relevant to a user's legal query, combine them into a single, clear summary.

Ensure the response is no more than 10 sentences and captures all key themes. Do not include article numbers unless necessary.

User Query:
\"\"\"{query}\"\"\"

Previous Summaries:
\"\"\"{combined_summaries}\"\"\"

Final Summary:
"""
    final_summary = model.invoke(final_prompt).strip()

    return {
        "summary": final_summary
    }

print(prune_and_summarize_batched(query,match,model))