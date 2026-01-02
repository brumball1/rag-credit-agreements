import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_token_count_per_document(jsonl_path: Path, window=10):
    rows = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    grouped = df.groupby("doc_id")

    for doc_id, group in grouped:
        group = group.sort_values("paragraph_in_doc")
        group["moving_avg"] = group["token_count"].rolling(window=window, center=True).mean()
        mean_val = group["token_count"].mean()
        plt.figure(figsize=(16, 6))

        plt.bar(group["paragraph_in_doc"], group["token_count"],
                width=1.0, alpha=0.4, label="Paragraph length")
        plt.plot(group["paragraph_in_doc"], group["moving_avg"],
                 color="red", linewidth=2, label=f"{window}-paragraph moving average")

        plt.axhline(mean_val, color="blue", linestyle="--", label=f"Mean = {mean_val:.1f}")

        plt.xlabel("Paragraph index (within document)")
        plt.ylabel("Token count")
        plt.title(f"Token count per paragraph - {doc_id}")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        out_path = jsonl_path.parent /  f"paragraph_lengths_{doc_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved plot for {doc_id} -> {out_path}")