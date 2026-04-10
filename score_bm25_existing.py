import json
from pathlib import Path
from src.training.build_training_triplets import score_triplets_with_dense, load_chunks

base_path = Path("data/derived")
combined_file = base_path / "triplets__gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20.jsonl"
chunks_file = base_path / "section_chunks.jsonl"

print("Loading chunks...")
corpus_texts = [c["text"] for c in load_chunks(chunks_file)]

print("Loading existing triplets...")
triplets = []
with combined_file.open("r") as f:
    for line in f:
        triplets.append(json.loads(line))

bm25_triplets = [t for t in triplets if t.get("negative_type") == "bm25"]
dense_triplets = [t for t in triplets if t.get("negative_type") == "dense"]

print(f"Loaded {len(bm25_triplets)} BM25 triplets. Scoring them...")
scored_bm25 = score_triplets_with_dense(bm25_triplets, corpus_texts)

print("Saving split and combined files...")
dense_file = base_path / "triplets__gemma3-12b__e5-base-v2__section_chunks__all__dense_only__window3-20.jsonl"
bm25_file = base_path / "triplets__gemma3-12b__e5-base-v2__section_chunks__all__bm25_only__window3-20.jsonl"

with dense_file.open("w") as f:
    for t in dense_triplets:
        f.write(json.dumps(t) + "\n")

with bm25_file.open("w") as f:
    for t in scored_bm25:
        f.write(json.dumps(t) + "\n")

with combined_file.open("w") as f:
    for t in dense_triplets + scored_bm25:
        f.write(json.dumps(t) + "\n")

print("Done!")
