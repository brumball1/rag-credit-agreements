import json
import numpy as np
from pathlib import Path
from glob import glob

#   source .venv/bin/activate && python -m src.training.evaluate_pairs

def load_triplets(path: Path) -> list[dict]:
    triplets = []
    with path.open("r") as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


def compute_metrics(triplets: list[dict]) -> dict:
    ranks = []
    margins = []
    positive_scores = []

    for t in triplets:
        rank = t.get("positive_rank")
        margin = t.get("margin")
        pos_score = t.get("positive_score")

        if rank is not None:
            ranks.append(rank)
        if margin is not None:
            margins.append(margin)
        if pos_score is not None:
            positive_scores.append(pos_score)

    ranks = np.array(ranks)
    margins = np.array(margins)
    positive_scores = np.array(positive_scores)

    n = len(ranks)

    return {
        "n_triplets":       len(triplets),
        "n_scored":         n,
        # retrieval quality 
        "MRR":              float(np.mean(1.0 / ranks)) if n else 0,
        "Recall@1":         float(np.mean(ranks <= 1)) if n else 0,
        "Recall@5":         float(np.mean(ranks <= 5)) if n else 0,
        "Recall@10":        float(np.mean(ranks <= 10)) if n else 0,
        "Mean rank":        float(np.mean(ranks)) if n else 0,
        "Median rank":      float(np.median(ranks)) if n else 0,
        # hard negative quality 
        "Mean margin":      float(np.mean(margins)) if len(margins) else 0,
        "Median margin":    float(np.median(margins)) if len(margins) else 0,
        "% margin > 0":     float(np.mean(margins > 0) * 100) if len(margins) else 0,
        # positive score distribution 
        "Mean pos score":   float(np.mean(positive_scores)) if len(positive_scores) else 0,
        "Median pos score": float(np.median(positive_scores)) if len(positive_scores) else 0,
    }


def print_comparison_table(results: dict[str, dict]):
    models = list(results.keys())
    col_width = 28

    # header
    print("\n" + "=" * (22 + col_width * len(models)))
    print(f"{'Metric':<22}" + "".join(f"{m:>{col_width}}" for m in models))
    print("=" * (22 + col_width * len(models)))

    # metric rows
    metric_keys = [
        ("n_triplets", "Triplets"),
        ("MRR", "MRR"),
        ("Recall@1", "Recall@1"),
        ("Recall@5", "Recall@5"),
        ("Recall@10", "Recall@10"),
        ("Mean rank", "Mean rank"),
        ("Median rank", "Median rank"),
        ("Mean margin", "Mean margin"),
        ("Median margin", "Median margin"),
        ("% margin > 0", "% margin > 0"),
        ("Mean pos score", "Mean pos score"),
        ("Median pos score", "Median pos score"),
    ]

    separators = {"MRR", "Mean margin", "Mean pos score"}

    for key, label in metric_keys:
        if label in separators:
            print("-" * (22 + col_width * len(models)))
        row = f"{label:<22}"
        for m in models:
            val = results[m].get(key, 0)
            if key == "n_triplets":
                row += f"{int(val):>{col_width},}"
            elif key == "% margin > 0":
                row += f"{val:>{col_width-1}.1f}%"
            else:
                row += f"{val:>{col_width}.4f}"
        print(row)

    print("=" * (22 + col_width * len(models)))


if __name__ == "__main__":
    import csv
    from datetime import datetime

    base_path = Path(__file__).resolve().parents[2]
    derived_path = base_path / "data" / "derived"

    # auto-discover all triplet files
    triplet_files = sorted(derived_path.glob("triplets__*.jsonl"))

    if not triplet_files:
        print("No triplet files found in data/derived/")
        exit(1)

    print(f"Found {len(triplet_files)} triplet file(s):")
    for f in triplet_files:
        print(f"  - {f.name}")

    results = {}
    file_meta = {}  # store full filename for CSV
    for path in triplet_files:
        # use the embedding model part of the filename as the label
        # format: triplets__<llm>__<embed>__<chunks>chunks.jsonl
        parts = path.stem.split("__")
        # format: triplets__<llm>__<embed>__<chunks>[__<strategy>].jsonl
        embed = parts[2] if len(parts) >= 3 else path.stem
        strategy_suffix = f"__{parts[4]}" if len(parts) >= 5 else ""
        label = embed + strategy_suffix
        print(f"\nLoading {path.name}...")
        triplets = load_triplets(path)
        results[label] = compute_metrics(triplets)
        file_meta[label] = path.name

    print_comparison_table(results)

    # save to CSV
    # strategy:vwindow files contain "window" in their name
    strategies_present = set()
    for path in triplet_files:
        if "window" in path.stem:
            strategies_present.add("window")
        else:
            strategies_present.add("topk")

    if len(strategies_present) > 1:
        csv_name = "model_comparison__mixed.csv"
    elif "window" in strategies_present:
        csv_name = "model_comparison__window_negatives.csv"
    else:
        csv_name = "model_comparison__topk_baseline.csv"

    csv_path = derived_path / csv_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    metric_keys = [
        "n_triplets", "MRR", "Recall@1", "Recall@5", "Recall@10",
        "Mean rank", "Median rank", "Mean margin", "Median margin",
        "% margin > 0", "Mean pos score", "Median pos score"
    ]

    rows_exist = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "model", "source_file"] + metric_keys)
        if not rows_exist:
            writer.writeheader()
        for model, metrics in results.items():
            row = {"timestamp": timestamp, "model": model, "source_file": file_meta[model]}
            row.update({k: metrics.get(k, "") for k in metric_keys})
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")
