"""Prompts for generating training data.
they are used in src.build_training_triplets.py to create the triplets for training the model in src.training.train_model.py
"""


POSITIVE_QUERY_PROMPT = """You are generating high-quality training data for a semantic retrieval system over credit agreements (legal financial documents).

Your goal is to create diverse, realistic search queries that professionals (e.g., lawyers, credit analysts, treasury teams, compliance officers) would use to retrieve the paragraph below.

Section context: {section_heading}

Instructions:

Generate exactly {n_queries} DISTINCT queries. Each query must follow a DIFFERENT style:

1. Precise factual query  
   - Ask about a specific obligation, threshold, party, definition, trigger, or condition

2. Paraphrased / indirect query  
   - Rephrase the meaning without copying wording from the paragraph

3. Keyword-style search  
   - 2–6 words, no question mark, like a real search input

4. Analytical / scenario-based query  
   - Ask how, why, when, or what happens under certain conditions

Critical rules:

- Queries must be answerable primarily from this paragraph
- Do NOT reference section numbers, clause numbers, or headings explicitly
- Do NOT copy long phrases verbatim unless legally necessary
- Prefer semantic variation and synonyms over exact phrasing
- Avoid making all queries definition-style, even if the paragraph defines a term
- At least one query should focus on implication, usage, or consequence
- At least one query must use wording that differs from key terms in the paragraph
- Avoid vague or generic queries
- Ensure variation in length, structure, and intent
- Write as a human professional would naturally search

Output rules:

- Output ONLY the queries
- One query per line
- Numbered 1, 2, 3, 4, ...
- No explanations, no headings, no blank lines

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
