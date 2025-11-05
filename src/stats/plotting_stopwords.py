import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


raw = pd.read_csv("../../data/derived/word_bank.raw.csv")
clean = pd.read_csv("../../data/derived/word_bank.csv")
lemma = pd.read_csv("../../data/derived/word_bank.lemma.csv")
raw_total = raw["count"].sum()
clean_total = clean["count"].sum()
lemma_total = lemma["count"].sum()
removed_total = raw_total - clean_total
percent_removed = (removed_total / raw_total) * 100

def bar_chart():
    labels = ["Raw corpus", "Cleaned corpus", "Lemmatised corpus"]
    values = [raw_total, clean_total, lemma_total]

    plt.bar(labels, values, color=["blue", "green", "purple"])
    plt.title(f"Total Tokens Across Corpus Versions\n({percent_removed:.1f}% removed after stopword cleaning)")
    plt.ylabel("Total number of tokens")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    offset = 0.01 * max(values)
    for x, y in enumerate(values):
        plt.text(x, y + offset, f"{y:,}", ha="center")

    plt.tight_layout()
    plt.savefig("../../data/derived/figures/total_tokens_comparison_barchart.png", dpi=800)
    plt.show()

def pie_chart():
    plt.pie(
        [clean_total, removed_total],
        labels=["Remaining words", "Stopwords removed"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["green", "red"])
    plt.title("Proportion of Stopwords in Corpus")
    plt.tight_layout()
    plt.savefig("../../data/derived/figures/stopword_pie.png", dpi=800)
    plt.show()

def top_20_words():
    top_raw = raw.head(20)
    top_clean = clean.head(20)
    top_lemma = lemma.head(20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].barh(top_raw["words"], top_raw["count"], color="blue")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 Words (Raw)")
    axes[0].set_xlabel("Count")
    axes[1].barh(top_clean["words"], top_clean["count"], color="green")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 20 Words (Cleaned)")
    axes[1].set_xlabel("Count")
    axes[2].barh(top_lemma["words"], top_lemma["count"], color="purple")
    axes[2].invert_yaxis()
    axes[2].set_title("Top 20 Words (Lemmatised)")
    axes[2].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig("../../data/derived/figures/top_20_words_comparison_all.png", dpi=1800)
    plt.show()

if __name__ == "__main__":
    bar_chart()
    pie_chart()
    top_20_words()
    print("All plots generated and saved to data/derived/figures/")

