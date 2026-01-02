from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_bigram_per_page(csv_path: Path, doc_id: str, output_dir: Path):

    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    doc_df = df[df["Document"] == doc_id]

    top_per_page = (
        doc_df.sort_values(["Page", "Count"], ascending=[True, False])
        .groupby("Page", as_index=False)
        .first()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(top_per_page["Page"], top_per_page["Count"], marker="o")
    plt.xlabel("Page")
    plt.ylabel("Count of most frequent bigram")
    plt.title(f"Most frequent bigram count per page: {doc_id}")
    plt.grid(True, linestyle="--", alpha=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / f"top_bigram_per_page_{doc_id}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=400)
    plt.close()
    print(f"Saved {outpath}")