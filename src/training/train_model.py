import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator

# source .venv/bin/activate && python -m src.training.train_model data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1 --model_name intfloat/e5-base-v2 --loss multiple_negatives_ranking --epochs 1 --batch_size 4 --learning_rate 2e-5
# source .venv/bin/activate && python -m src.training.train_model data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1 --model_name intfloat/e5-base-v2 --loss triplet --epochs 1 --batch_size 4 --learning_rate 2e-5


def load_triplets(file_path: Path):
    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            # InputExample expects lists of texts: [query, positive, negative]
            examples.append(InputExample(texts=[t["query"], t["positive"], t["negative"]]))
    return examples

def train_model(
    fold_dir: Path, 
    model_name: str, 
    loss_type: str, 
    epochs: int = 1, 
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 120,
    eval_steps: int = 150
):
    print(f"Loading Model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Loading Training Data from {fold_dir / 'train.jsonl'}")
    train_examples = load_triplets(fold_dir / "train.jsonl")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print(f"Loading Validation Data from {fold_dir / 'val.jsonl'}")
    val_examples = load_triplets(fold_dir / "val.jsonl")
    
    # We use TripletEvaluator for validation to get metrics
    val_evaluator = TripletEvaluator.from_input_examples(val_examples, name="val", show_progress_bar=True)

    print(f"Configuring Loss Function: {loss_type}")
    if loss_type == "triplet":
        train_loss = losses.TripletLoss(model=model)
    elif loss_type == "multiple_negatives_ranking":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Output directory for saving model
    output_model_dir = fold_dir / f"weights_{loss_type}"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=epochs,
        evaluation_steps=eval_steps,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=str(output_model_dir),
        show_progress_bar=True
    )
    
    print(f"Training Complete. Model saved to {output_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dense embedding model using sentence-transformers.")
    parser.add_argument("fold_dir", type=str, help="Path to the directory containing train.jsonl and val.jsonl")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2", help="HuggingFace model name or path")
    parser.add_argument("--loss", type=str, choices=["triplet", "multiple_negatives_ranking"], default="multiple_negatives_ranking")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=120)
    parser.add_argument("--eval_steps", type=int, default=150)
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir).resolve()
    
    if not (fold_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"train.jsonl not found in {fold_dir}")

    train_model(
        fold_dir=fold_dir,
        model_name=args.model_name,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps
    )
