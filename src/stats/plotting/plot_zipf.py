from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_zipf(raw_or_lemma: pd.DataFrame, label: str, output_dir: Path, top_n: int = 500):
    freqs = sorted(raw_or_lemma["count"], reverse=True)[:top_n]
    ranks = np.arange(1, top_n + 1)

    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, freqs, marker=".", linestyle="none")
    plt.title(f"Zipf Plot for {label} Token Frequencies")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / f"log_zipf_{label.lower()}_top{top_n}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()
    print(f"Saved {outpath}")


def plot_zipf_with_fit(lemma: pd.DataFrame, output_dir: Path, top_n: int = 500):
    freqs = np.array(sorted(lemma["count"], reverse=True)[:top_n])
    ranks = np.arange(1, top_n + 1)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    fitted = np.exp(intercept + slope * log_ranks)

    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, freqs, marker=".", linestyle="none", label="Lemma data")
    plt.loglog(ranks, fitted, linestyle="solid", label="Fitted power law")

    plt.title("Zipf Plot with Fitted Power Law (Lemma)")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    a = np.exp(intercept)
    b = slope
    eqn_text = f"f = a * r^(b)\na = {a:.2f}\nb = {b:.2f}"
    plt.text(
        0.05,
        0.05,
        eqn_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "zipf_lemma_with_fit.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=800)
    plt.close()
    print(f"Saved {outpath}")