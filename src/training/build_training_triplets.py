import json
from pathlib import Path
import ollama
from tqdm import tqdm
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.training.prompts import POSITIVE_QUERY_PROMPT
from src.stats.tokeniser import tokeniser

#   source .venv/bin/activate && python -m src.training.build_training_triplets

OLLAMA_URL = "http://localhost:11434/api/generate"
#DEFAULT_MODEL = "llama3.2"
DEFAULT_MODEL = "gemma3:12b"
MIN_TOKENS = 25
N_POSITIVE_QUERIES = 4
N_SAMPLE_CHUNKS = None  # 100, 1000, or None (None = all chunks)

NEGATIVE_STRATEGY    = "random_window" # "top_k", "random_window", or "threshold"
NEGATIVE_WINDOW_MIN  = 3 
NEGATIVE_WINDOW_MAX  = 20 

USE_DENSE_NEGATIVES = True
USE_BM25_NEGATIVES  = True

N_DENSE_NEGATIVES = 1
N_BM25_NEGATIVES  = 1
BM25_WINDOW_MIN   = 1
BM25_WINDOW_MAX   = 10 

#EMBEDDING_MODEL = "all-mpnet-base-v2"
#EMBEDDING_MODEL = "intfloat/e5-base-v2"
#EMBEDDING_MODEL = "ibm-granite/granite-embedding-english-r2"
EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"
#EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
#EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
#EMBEDDING_MODEL = "philschmid/bge-base-financial-matryoshka"

def load_chunks(jsonl_path: Path, min_tokens: int = MIN_TOKENS):
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if len(tokeniser(record["text"])) >= min_tokens:
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
                chunk_text=text,
                section_heading=chunk.get("section_heading") or "General"
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
                    "paragraph_in_doc": chunk.get("section_in_doc", chunk.get("paragraph_in_doc", -1))
                })

    return p_pairs


def generate_dense_triplets(positive_pairs: list[dict],
                            corpus_texts: list[str],
                            model_name: str = EMBEDDING_MODEL,
                            top_k: int = N_DENSE_NEGATIVES,
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
            low = min(window_min - 1, len(candidate_indices) - 1)
            high = min(window_max - 1, len(candidate_indices) - 1)
            pool = candidate_indices[low:high + 1]
            chosen = np.random.choice(pool, size=min(top_k, len(pool)), replace=False)
        elif strategy == "threshold":
            # experiment w values
            relative_margin = 0.95
            absolute_max = 0.80
            absolute_min = 0.20
            
            pool = []
            for idx in candidate_indices:
                neg_score = similarities[i][idx]
                if (neg_score <= relative_margin * positive_score) and (absolute_min <= neg_score <= absolute_max):
                    pool.append(idx)
            
            # if this filter is too strict and we get no negatives, fallback to slightly wider margins
            if len(pool) < top_k:
                print(f"Only found {len(pool)} negatives within thresholds. Relaxing absolute_min to 0.0")
                pool = []
                for idx in candidate_indices:
                    neg_score = similarities[i][idx]
                    if neg_score <= relative_margin * positive_score and neg_score <= absolute_max:
                        pool.append(idx)
            
            # if still not enough, fallback to window behavior but print it
            if len(pool) < top_k:
                print(f"Still not enough negatives for Query {i+1}. Falling back to default random_window.")
                low = min(window_min - 1, len(candidate_indices) - 1)
                high = min(window_max - 1, len(candidate_indices) - 1)
                pool = candidate_indices[low:high + 1]
            
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
                "margin": margin,
                "negative_type": "dense"
            })

        if positive_score is not None and triplets:
            print(f"Query {i+1}: positive_score={positive_score:.4f}, rank={rank}/{len(corpus_texts)}, neg_score={triplets[-1]['negative_score']:.4f}, margin={triplets[-1]['margin']:.4f}")

    return triplets


def generate_bm25_triplets(
    positive_pairs: list[dict],
    corpus_texts: list[str],
    top_k: int = N_BM25_NEGATIVES,
) -> list[dict]:
    print(f"Tokenising corpus for BM25 ({len(corpus_texts)} chunks)...")
    tokenized_corpus = [tokeniser(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    triplets = []

    for pair in tqdm(positive_pairs, desc="BM25 negatives"):
        query_tokens = tokeniser(pair["positive_query"])
        scores = bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        candidates = [idx for idx in ranked_indices if corpus_texts[idx] != pair["ground_truth"]]
        low  = min(BM25_WINDOW_MIN, len(candidates) - 1)
        high = min(BM25_WINDOW_MAX, len(candidates) - 1)
        pool = candidates[low:high + 1]
        chosen = list(np.random.choice(pool, size=min(top_k, len(pool)), replace=False))

        for idx in chosen:
            triplets.append({
                "query": pair["positive_query"],
                "positive": pair["ground_truth"],
                "doc_id": pair["doc_id"],
                "paragraph_in_doc": pair["paragraph_in_doc"],
                "positive_score": None,
                "positive_rank": None,
                "negative": corpus_texts[idx],
                "negative_score": None,
                "margin": None,
                "negative_type": "bm25"
            })

    return triplets


def score_triplets_with_dense(
    triplets: list[dict],
    corpus_texts: list[str],
    model_name: str = EMBEDDING_MODEL,
) -> list[dict]:
    model = SentenceTransformer(model_name)
    corpus_embeddings = model.encode(corpus_texts, batch_size=32,
                                     normalize_embeddings=True, show_progress_bar=True)
    text_to_idx = {text: i for i, text in enumerate(corpus_texts)}

    for t in triplets:
        pos_idx = text_to_idx.get(t["positive"])
        neg_idx = text_to_idx.get(t["negative"])
        if pos_idx is None or neg_idx is None:
            continue
        query_emb = model.encode([t["query"]], normalize_embeddings=True)
        sims = (query_emb @ corpus_embeddings.T)[0]
        pos_score = float(sims[pos_idx])
        neg_score = float(sims[neg_idx])
        t["positive_score"] = round(pos_score, 4)
        t["negative_score"] = round(neg_score, 4)
        t["margin"] = round(pos_score - neg_score, 4)
        t["positive_rank"] = int(np.sum(sims > pos_score)) + 1

    return triplets


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "data" / "derived" / "section_chunks.jsonl"

    # stuff for file naming
    llm_tag = DEFAULT_MODEL.replace(":", "-")
    embed_tag = EMBEDDING_MODEL.split("/")[-1]  # handles org/model-name format
    chunks_tag = str(N_SAMPLE_CHUNKS) if N_SAMPLE_CHUNKS is not None else "all"

    # stage 1 cache: generating pairs is expensive, so we cache it
    pairs_cache_path = base_path / "data" / "derived" / f"pairs_cache__{llm_tag}__section_chunks__{chunks_tag}__{N_POSITIVE_QUERIES}q.jsonl"

    # stage 2 output: named after embedding model + strategy so each combo is unique
    if NEGATIVE_STRATEGY == "random_window":
        strategy_tag = f"__window{NEGATIVE_WINDOW_MIN}-{NEGATIVE_WINDOW_MAX}"
    elif NEGATIVE_STRATEGY == "threshold":
        strategy_tag = "__threshold"
    else:
        strategy_tag = ""

    negative_tags = []
    if USE_DENSE_NEGATIVES:
        negative_tags.append("dense")
    if USE_BM25_NEGATIVES:
        negative_tags.append("bm25")
    negative_tag = "__".join(negative_tags)

    output_path = base_path / "data" / "derived" / f"triplets__{llm_tag}__{embed_tag}__section_chunks__{chunks_tag}__{negative_tag}{strategy_tag}.jsonl"

    # load corpus (always needed for hard negative mining)
    all_chunks = load_chunks(data_path)
    corpus_texts = [c["text"] for c in all_chunks]

    # generate positive pairs via LLM (skip now if cache exists)
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
    print(f"\n[Stage 2] Mining hard negatives (dense={USE_DENSE_NEGATIVES}, bm25={USE_BM25_NEGATIVES})")
    triplets = []

    if USE_DENSE_NEGATIVES:
        print(f"  Running dense negatives with: {EMBEDDING_MODEL}")
        triplets.extend(generate_dense_triplets(pairs, corpus_texts, top_k=N_DENSE_NEGATIVES))

    if USE_BM25_NEGATIVES:
        print(f"  Running BM25 negatives")
        bm25_triplets = generate_bm25_triplets(pairs, corpus_texts, top_k=N_BM25_NEGATIVES)
        print("  Scoring BM25 negatives with dense model")
        bm25_triplets = score_triplets_with_dense(bm25_triplets, corpus_texts)
        triplets.extend(bm25_triplets)

    print(f"[Stage 2] Generated {len(triplets)} triplets total")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")
    print(f"[Stage 2] Saved to {output_path}")
