import json
import random
from pathlib import Path
import argparse

#   source .venv/bin/activate && python -m src.training.create_folds data/derived/triplets__gemma3-12b__e5-base-v2__allchunks__window2-15.jsonl


def create_folds(triplets_file: Path, output_dir: Path, chunks_file: Path, seed: int = 42):
    text_to_doc = {}
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            text_to_doc[c["text"]] = c["doc_id"]

    triplets = []
    with triplets_file.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if "doc_id" not in t or t["doc_id"] is None:
                raise ValueError(f"Triplet missing doc_id: {line[:120]}...")
            triplets.append(t)

    doc_ids = sorted(set(t["doc_id"] for t in triplets))
    print(f"Loaded {len(triplets)} triplets across {len(doc_ids)} documents: {doc_ids}")

    random.seed(seed)

    for fold_idx, test_doc in enumerate(doc_ids, start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_triplets = [t for t in triplets if t["doc_id"] == test_doc]
        train_pool = []
        for t in triplets:
            if t["doc_id"] != test_doc:
                # Discard triplet if its negative chunk comes from the test doc
                neg_doc_id = text_to_doc.get(t["negative"])
                if neg_doc_id != test_doc:
                    train_pool.append(t)

        random.shuffle(train_pool)

        split_idx = int(0.9 * len(train_pool))
        train_triplets = train_pool[:split_idx]
        val_triplets = train_pool[split_idx:]

        # section to verify no test doc leakage
        train_docs = set(t["doc_id"] for t in train_triplets)
        val_docs = set(t["doc_id"] for t in val_triplets)
        assert test_doc not in train_docs, f"Test doc {test_doc} leaked into train"
        assert test_doc not in val_docs, f"Test doc {test_doc} leaked into val"
        
        # Verify negative content leak
        for split_name, split_data in [("train", train_triplets), ("val", val_triplets)]:
            for t in split_data:
                neg_doc_id = text_to_doc.get(t["negative"])
                assert neg_doc_id != test_doc, f"Negative from {test_doc} leaked into {split_name}"

        for name, data in [("train", train_triplets), ("val", val_triplets), ("test", test_triplets)]:
            with (fold_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for t in data:
                    f.write(json.dumps(t) + "\n")

        print(f"Fold {fold_idx} | test_doc={test_doc:30s} | "
              f"train={len(train_triplets):,}  val={len(val_triplets):,}  test={len(test_triplets):,}")

        # document distribution in validation set

        val_dist = {}
        for t in val_triplets:
            doc_id = t["doc_id"]
            if doc_id not in val_dist:
                val_dist[doc_id] = 0
            val_dist[doc_id] += 1

        print(f"Validation doc distribution: {val_dist}")

    print(f"\n{len(doc_ids)} folds saved to {output_dir}")


if __name__ == "__main__":
    """
      Idea is that using the parser we can change the number of folds, the seed, and the output directory without 
      having to rewrite the code.

    """

    parser = argparse.ArgumentParser(description="Leave-one-document-out fold creation")
    parser.add_argument("triplets_file", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--chunks", type=str, default=None, help="Path to original paragraph chunks to verify negative doc_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    triplets_path = Path(args.triplets_file).resolve()

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        dataset_name = triplets_path.stem.replace("triplets__", "")
        out_dir = triplets_path.parent / "training" / dataset_name

    if args.chunks:
        chunks_path = Path(args.chunks).resolve()
    else:
        # parent of triplets_path is usually `data/derived`, chunks is `paragraph_chunks.jsonl`
        chunks_path = triplets_path.parent / "paragraph_chunks.jsonl"

    create_folds(triplets_path, out_dir, chunks_path, seed=args.seed)
