from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_word_banks(base_dir: Path):
    derived = base_dir / "data" / "derived"
    raw = pd.read_csv(derived / "word_bank.raw.csv")
    clean = pd.read_csv(derived / "word_bank.csv")
    lemma = pd.read_csv(derived / "word_bank.lemma.csv")
    return raw, clean, lemma


def plot_token_totals(raw, clean, lemma, output_dir: Path):
    raw_total = raw["count"].sum()
    clean_total = clean["count"].sum()
    lemma_total = lemma["count"].sum()
    removed_total = raw_total - clean_total
    percent_removed = (removed_total / raw_total) * 100

    labels = ["Raw corpus", "Cleaned corpus", "Lemmatised corpus"]
    values = [raw_total, clean_total, lemma_total]

    plt.bar(labels, values)
    plt.title(f"Total Tokens Across Corpus Versions\n({percent_removed:.1f}% removed after stopword cleaning)")
    plt.ylabel("Total number of tokens")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    offset = 0.01 * max(values)
    for x, y in enumerate(values):
        plt.text(x, y + offset, f"{y:,}", ha="center")

    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "total_tokens_comparison_barchart.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=800)
    plt.close()
    print(f"Saved {outpath}")


def plot_stopword_pie(raw, clean, output_dir: Path):
    raw_total = raw["count"].sum()
    clean_total = clean["count"].sum()
    removed_total = raw_total - clean_total

    plt.pie(
        [clean_total, removed_total],
        labels=["Remaining words", "Stopwords removed"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Proportion of Stopwords in Corpus")

    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "stopword_pie.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=800)
    plt.close()
    print(f"Saved {outpath}")


def plot_top_20_words(raw, clean, lemma, output_dir: Path):
    top_raw = raw.head(20)
    top_clean = clean.head(20)
    top_lemma = lemma.head(20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].barh(top_raw["words"], top_raw["count"])
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 Words (Raw)")
    axes[0].set_xlabel("Count")

    axes[1].barh(top_clean["words"], top_clean["count"])
    axes[1].invert_yaxis()
    axes[1].set_title("Top 20 Words (Cleaned)")
    axes[1].set_xlabel("Count")

    axes[2].barh(top_lemma["words"], top_lemma["count"])
    axes[2].invert_yaxis()
    axes[2].set_title("Top 20 Words (Lemmatised)")
    axes[2].set_xlabel("Count")

    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "top_20_words_comparison_all.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=1800)
    plt.close()
    print(f"Saved {outpath}")