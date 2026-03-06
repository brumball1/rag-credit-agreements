#these prompts will be sent to Ollama and they will be used to create synthetic query chunk pairs

POSITIVE_QUERY_PROMPT = """You are helping create training data for a search system over credit agreements (loan documents).

Given the following paragraph from a credit agreement, generate exactly {n_queries} realistic search queries that this paragraph would be the correct answer to.

Rules:
- Each query should be a natural question a financial analyst, lawyer, or compliance officer would ask
- Queries must be SPECIFIC to the content in this paragraph (reference specific parties, amounts, dates, or clause details when present)
- The answer must be fully contained within this paragraph alone.
- Do NOT generate generic questions like "What is a credit agreement?" — be specific
- Each query should be answerable ONLY by this paragraph, not by a generic knowledge of credit agreements
- Output ONLY the queries, one per line, numbered 1, 2, etc.

Paragraph:
\"\"\"
{chunk_text}
\"\"\"

Queries:"""


HARD_NEGATIVE_PROMPT = """You are helping create training data for a search system over credit agreements (loan documents).

Given the following paragraph from a credit agreement, generate exactly {n_queries} search query that sounds related to credit agreements but this specific paragraph does NOT answer.

Rules:
- The query should sound plausible and related to credit agreements in general
- The query must NOT be answerable by the paragraph below
- The query should be about a DIFFERENT topic, clause, or aspect than what this paragraph covers
- Make it a realistic question a financial analyst would ask
- Output ONLY the query, one per line, numbered 1, 2, etc.
- Do NOT include any preamble, explanation, or extra text

Paragraph:
\"\"\"
{chunk_text}
\"\"\"

Query:"""
