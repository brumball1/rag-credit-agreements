import json
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.training.evaluate_pairs import compute_metrics, print_comparison_table

# source .venv/bin/activate && python -m src.training.test_fine_tuned data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_1 --chunks data/derived/section_chunks.jsonl
# source .venv/bin/activate && python -m src.training.test_fine_tuned data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1
# source .venv/bin/activate && python -m src.training.test_fine_tuned data/derived/training/gemma3-12b__e5-base-v2__allchunks__threshold/fold_1

def load_data(fold_dir: Path, chunks_path: Path):
    test_triplets = []
    with (fold_dir / "test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            test_triplets.append(json.loads(line))
            
    corpus_texts = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            corpus_texts.append(c["text"])
            
    return test_triplets, corpus_texts

def evaluate_model(model_name_or_path: str, test_triplets: list[dict], corpus_texts: list[str], model_label: str) -> dict:
    print(f"\nLoading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)
    
    print(f"Encoding {len(corpus_texts)} corpus chunks...")
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    
    queries = [t["query"] for t in test_triplets]
    print(f"Encoding {len(queries)} queries...")
    query_embeddings = model.encode(queries, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    
    similarities = query_embeddings @ corpus_embeddings.T
    
    evaluated_triplets = []
    for i, t in enumerate(test_triplets):
        try:
            positive_idx = corpus_texts.index(t["positive"])
            negative_idx = corpus_texts.index(t["negative"])
            
            pos_score = float(similarities[i, positive_idx])
            neg_score = float(similarities[i, negative_idx])
            
            rank = int(np.sum(similarities[i] > pos_score)) + 1
            margin = pos_score - neg_score
            
            evaluated_triplets.append({
                "positive_score": pos_score,
                "positive_rank": rank,
                "negative_score": neg_score,
                "margin": margin
            })
            
        except ValueError:
            # If a chunk wasn't found in the corpus (shouldn't happen), skip
            continue
            
    # Compute metrics using existing function
    metrics = compute_metrics(evaluated_triplets)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Model on Test Set")
    parser.add_argument("fold_dir", type=str, help="Path to fold directory (e.g. fold_1)")
    parser.add_argument("--base_model", type=str, default="intfloat/e5-base-v2", help="Base model name")
    parser.add_argument("--chunks", type=str, default="data/derived/paragraph_chunks.jsonl")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned weights, overrides the default fold_dir/weights_multiple_negatives_ranking_best")
    args = parser.parse_args()
    
    fold_dir = Path(args.fold_dir).resolve()
    chunks_path = Path(args.chunks).resolve()
    
    if args.model_path:
        ft_model_path = Path(args.model_path).resolve()
    else:
        ft_model_path = fold_dir / "weights_multiple_negatives_ranking_best"
    
    if not ft_model_path.exists():
        print(f"Fine-tuned model not found at {ft_model_path}")
        exit(1)
        
    test_triplets, corpus_texts = load_data(fold_dir, chunks_path)
    
    results = {}
    
    # eval base model
    results["Base Model"] = evaluate_model(args.base_model, test_triplets, corpus_texts, "Base Model")
    
    # Evaluate Fine-Tuned Model
    results["Fine-Tuned Model"] = evaluate_model(str(ft_model_path), test_triplets, corpus_texts, "Fine-Tuned Model")
    
    print(f"\n\n======== TEST SET EVALUATION RESULTS ({fold_dir.name}) ========\n")
    print_comparison_table(results)
