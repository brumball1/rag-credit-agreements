import json
from pathlib import Path
import ollama
from tqdm import tqdm
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from src.training.prompts import POSITIVE_QUERY_PROMPT, HARD_NEGATIVE_PROMPT

#   source .venv/bin/activate && python -m src.training.build_training_triplets

OLLAMA_URL = "http://localhost:11434/api/generate"
#DEFAULT_MODEL = "llama3.2"
DEFAULT_MODEL = "gemma3:12b"
MIN_TOKENS = 25
N_POSITIVE_QUERIES = 1
N_HARD_NEGATIVES = 2
N_SAMPLE_CHUNKS = None  # 100, 1000, or None (None = all chunks)

NEGATIVE_STRATEGY    = "random_window"  # "top_k" or "random_window"
NEGATIVE_WINDOW_MIN  = 2               # don't take rank 1 (risk of false negatives)
NEGATIVE_WINDOW_MAX  = 15              # upper bound of sampling window

#EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
#EMBEDDING_MODEL = "ibm-granite/granite-embedding-english-r2"
#EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
#EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
#EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
#EMBEDDING_MODEL = "philschmid/bge-base-financial-matryoshka"

def load_chunks(jsonl_path: Path, min_tokens: int = MIN_TOKENS):
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if len(record["text"]) >= min_tokens:
                chunks.append(record)
    print(f"Loaded {len(chunks)} chunks and filtered out {min_tokens} tokens")
    return chunks

def call_ollama_generate(prompt: str, model: str = DEFAULT_MODEL):
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]


def generate_positive_pairs(chunks: list[dict], n_queries: int = N_POSITIVE_QUERIES) -> list[dict]:
    p_pairs = []
    print(f"Generating queries for {len(chunks)} chunks")

    for chunk in tqdm(chunks, desc="Processing chunks"):
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            prompt = POSITIVE_QUERY_PROMPT.format(
                n_queries=n_queries,
                chunk_text=text
            )
        except KeyError as e:
            print(f"Error formatting prompt: {e}")
            continue

        raw_output = call_ollama_generate(prompt)

        lines = raw_output.strip().split("\n")
        for line in lines:
            cleaned_line = line.strip()
            cleaned_line = re.sub(r"^\d+[\.\)\:]\s*", "", cleaned_line)
            if cleaned_line:
                p_pairs.append({
                    "positive_query": cleaned_line,
                    "ground_truth": text,
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "paragraph_in_doc": chunk.get("paragraph_in_doc", -1)
                })

    return p_pairs


def generate_dense_triplets(positive_pairs: list[dict],
                            corpus_texts: list[str],
                            model_name: str = EMBEDDING_MODEL,
                            top_k: int = N_HARD_NEGATIVES,
                            strategy: str = NEGATIVE_STRATEGY,
                            window_min: int = NEGATIVE_WINDOW_MIN,
                            window_max: int = NEGATIVE_WINDOW_MAX) -> list[dict]:

    model = SentenceTransformer(model_name)
    print(f"Encoding {len(corpus_texts)} corpus chunks")
    print(f"Negative strategy: {strategy}" + (f" (rank {window_min}–{window_max})" if strategy == "random_window" else ""))
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    queries = [p["positive_query"] for p in positive_pairs]
    query_embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    similarities = query_embeddings @ corpus_embeddings.T

    triplets = []

    for i, pair in enumerate(positive_pairs):
        try:
            positive_idx = corpus_texts.index(pair["ground_truth"])
            positive_score = float(similarities[i, positive_idx])
            rank = int(np.sum(similarities[i] > positive_score)) + 1
        except ValueError:
            print(f"Query {i+1}: positive chunk not found in corpus_texts (filtered out?)")
            positive_score, rank = None, None

        # rank all non-positive chunks by similarity (descending)
        ranked_indices = np.argsort(similarities[i])[::-1]
        candidate_indices = [idx for idx in ranked_indices if corpus_texts[idx] != pair["ground_truth"]]

        if strategy == "random_window":
            # sample randomly from within the rank window (0-indexed into candidates)
            lo = min(window_min - 1, len(candidate_indices) - 1)
            hi = min(window_max - 1, len(candidate_indices) - 1)
            pool = candidate_indices[lo:hi + 1]
            chosen = np.random.choice(pool, size=min(top_k, len(pool)), replace=False)
        else:
            # top_k: take the most similar non-positive chunks in order
            chosen = candidate_indices[:top_k]

        for idx in chosen:
            neg_score = float(similarities[i][idx])
            margin = round(positive_score - neg_score, 4) if positive_score is not None else None
            triplets.append({
                "query": pair["positive_query"],
                "positive": pair["ground_truth"],
                "doc_id": pair["doc_id"],
                "paragraph_in_doc": pair["paragraph_in_doc"],
                "positive_score": round(positive_score, 4) if positive_score is not None else None,
                "positive_rank": rank,
                "negative": corpus_texts[idx],
                "negative_score": round(neg_score, 4),
                "margin": margin
            })

        if positive_score is not None and triplets:
            print(f"Query {i+1}: positive_score={positive_score:.4f}, rank={rank}/{len(corpus_texts)}, neg_score={triplets[-1]['negative_score']:.4f}, margin={triplets[-1]['margin']:.4f}")

    return triplets


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "data" / "derived" / "paragraph_chunks.jsonl"

    # file naming
    llm_tag = DEFAULT_MODEL.replace(":", "-")
    embed_tag = EMBEDDING_MODEL.split("/")[-1]  # handles org/model-name format
    chunks_tag = str(N_SAMPLE_CHUNKS) if N_SAMPLE_CHUNKS is not None else "all"

    # stage 1 cache: positive pairs are expensive (LLM), save once and reuse
    pairs_cache_path = base_path / "data" / "derived" / f"pairs_cache__{llm_tag}__{chunks_tag}chunks.jsonl"

    # stage 2 output: named after embedding model + strategy so each combo is unique
    strategy_tag = f"__window{NEGATIVE_WINDOW_MIN}-{NEGATIVE_WINDOW_MAX}" if NEGATIVE_STRATEGY == "random_window" else ""
    output_path = base_path / "data" / "derived" / f"triplets__{llm_tag}__{embed_tag}__{chunks_tag}chunks{strategy_tag}.jsonl"

    # load corpus (always needed for hard negative mining)
    all_chunks = load_chunks(data_path)
    corpus_texts = [c["text"] for c in all_chunks]

    # generate positive pairs via LLM (skip if cache exists)
    if pairs_cache_path.exists():
        print(f"[Stage 1] Cache found — loading pairs from {pairs_cache_path.name}")
        pairs = []
        with pairs_cache_path.open("r") as f:
            for line in f:
                pairs.append(json.loads(line))
        print(f"[Stage 1] Loaded {len(pairs)} cached positive pairs — skipping LLM")
    else:
        print(f"[Stage 1] No cache found — running LLM query generation...")
        sample_chunks = all_chunks[:N_SAMPLE_CHUNKS]  # [:None] returns all chunks
        pairs = generate_positive_pairs(sample_chunks)
        print(f"[Stage 1] Generated {len(pairs)} positive pairs")
        pairs_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with pairs_cache_path.open("w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[Stage 1] Pairs cached to {pairs_cache_path.name}")

    # embed corpus and mine hard negatives
    print(f"\n[Stage 2] Mining hard negatives with: {EMBEDDING_MODEL}")
    triplets = generate_dense_triplets(pairs, corpus_texts)
    print(f"[Stage 2] Generated {len(triplets)} triplets")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")
    print(f"[Stage 2] Saved to {output_path}")
