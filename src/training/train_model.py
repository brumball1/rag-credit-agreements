import json
import csv
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def load_triplets(file_path: Path):
    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            examples.append(InputExample(texts=[t["query"], t["positive"], t["negative"]]))
    return examples

def compute_recall_at_10(model, val_triplets_raw, corpus_texts):
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    queries = [t["query"] for t in val_triplets_raw]
    query_embeddings = model.encode(queries, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    similarities = query_embeddings @ corpus_embeddings.T

    ranks = []
    for i, t in enumerate(val_triplets_raw):
        try:
            pos_idx = corpus_texts.index(t["positive"])
            pos_score = float(similarities[i, pos_idx])
            # Rank is number of scores higher than pos_score + 1
            rank = int(np.sum(similarities[i] > pos_score)) + 1
            ranks.append(rank)
        except ValueError:
            continue

    ranks = np.array(ranks)
    n = len(ranks)
    return {
        "Recall@10": float(np.mean(ranks <= 10)) if n else 0,
        "Recall@1": float(np.mean(ranks <= 1))  if n else 0,
        "MRR": float(np.mean(1.0 / ranks))  if n else 0,
        "Mean rank": float(np.mean(ranks)) if n else 0,
        "ranks_used": ranks
    }

def train_model(
    fold_dir: Path,
    corpus_path: Path,
    model_name: str,
    loss_type: str,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 120,
    max_epochs: int = 100,
    patience: int = 3,
    max_grad_norm: float = 1.0,
):
    print(f"Loading Model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Loading Training Data from {fold_dir / 'train.jsonl'}")
    train_examples = load_triplets(fold_dir / "train.jsonl")

    # Load raw val triplets for our custom Recall@10 evaluation
    print(f"Loading Validation Data from {fold_dir / 'val.jsonl'}")
    val_triplets_raw = []
    with (fold_dir / "val.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            val_triplets_raw.append(json.loads(line))

    # Load corpus for ranking evaluation
    print(f"Loading Corpus from {corpus_path}")
    corpus_texts = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            corpus_texts.append(json.loads(line)["text"])

    print(f"Configuring Loss Function: {loss_type}")
    if loss_type == "triplet":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.2
        )
    elif loss_type == "multiple_negatives_ranking":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    output_model_dir = fold_dir / f"weights_{loss_type}"
    best_checkpoint_dir = fold_dir / f"weights_{loss_type}_best"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    best_recall = -1.0
    patience_counter = 0
    history = []

    print(f"\nStarting training (max {max_epochs} epochs, patience={patience})\n")

    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs}")

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            max_grad_norm=max_grad_norm,
            show_progress_bar=True
        )

        metrics = compute_recall_at_10(model, val_triplets_raw, corpus_texts)
        recall = metrics["Recall@10"]
        history.append({"epoch": epoch, "Recall@10": recall, "Recall@1": metrics["Recall@1"], "MRR": metrics["MRR"], "Mean rank": metrics["Mean rank"]})

        print(f"Epoch {epoch:3d} (n={len(metrics['ranks_used'])}), Recall@10={recall:.4f}  Recall@1={metrics['Recall@1']:.4f}  MRR={metrics['MRR']:.4f}  Mean rank={metrics['Mean rank']:.1f}")

        import math
        if math.isnan(recall):
            print(f"NaN detected in Recall@10 — model weights are corrupted. Stopping.")
            break

        if recall > best_recall:
            best_recall = recall
            patience_counter = 0
            print(f"New best Recall@10={best_recall:.4f} — saving checkpoint")
            model.save(str(output_model_dir))
            if best_checkpoint_dir.exists():
                shutil.rmtree(best_checkpoint_dir)
            shutil.copytree(str(output_model_dir), str(best_checkpoint_dir))
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best Recall@10={best_recall:.4f}")
                break

    print(f"\nTraining finished. Best model saved to {best_checkpoint_dir}")

    csv_path = output_model_dir / "training_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "Recall@10", "Recall@1", "MRR", "Mean rank"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Metrics saved to {csv_path}")

    epochs_list = [h["epoch"] for h in history]
    recalls = [h["Recall@10"] for h in history]
    plt.figure()
    plt.plot(epochs_list, recalls, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Recall@10")
    plt.title(f"Recall@10 vs Epoch ({loss_type})")
    plt.grid(True)
    plot_path = output_model_dir / "recall_at_10.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dense embedding model using sentence-transformers.")
    parser.add_argument("fold_dir", type=str, help="Path to the directory containing train.jsonl and val.jsonl")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the JSONL corpus file")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--loss", type=str, choices=["triplet", "multiple_negatives_ranking"], default="multiple_negatives_ranking")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=120)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm (default 1.0). Lower for larger models.")
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir).resolve()
    corpus_path = Path(args.corpus_path).resolve()

    if not (fold_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"train.jsonl not found in {fold_dir}")

    train_model(
        fold_dir=fold_dir,
        corpus_path=corpus_path,
        model_name=args.model_name,
        loss_type=args.loss,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
    )
