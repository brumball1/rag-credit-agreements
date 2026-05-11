from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    COLORS,
    DERIVED_DIR,
    configure_plots,
    fit_heaps_law,
    fit_zipf_power_law,
    plt,
    read_jsonl,
    save_figure,
    section_output_dir,
    section_chunk_statistics,
    shannon_entropy,
    tokenize,
)


OUTPUT_DIR = section_output_dir("corpus_structure")
SECTION_CHUNKS = DERIVED_DIR / "section_chunks.jsonl"
LEMMA_WORD_BANK = DERIVED_DIR / "word_bank.lemma.csv"
RAW_WORD_BANK = DERIVED_DIR / "word_bank.raw.csv"


def plot_zipf_law(top_n: int = 500) -> None:
    raw_fit = fit_zipf_power_law(RAW_WORD_BANK, top_n=top_n)
    lemma_fit = fit_zipf_power_law(LEMMA_WORD_BANK, top_n=top_n)
    print(f"Verified Zipf alpha (raw): {raw_fit['alpha']:.6f}, R^2 = {raw_fit['r2']:.6f}")
    print(f"Verified Zipf alpha (lemmatised): {lemma_fit['alpha']:.6f}, R^2 = {lemma_fit['r2']:.6f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.loglog(
        lemma_fit["ranks"],
        lemma_fit["freqs"],
        ".",
        color=COLORS["e5_base"],
        markersize=5,
        label="Observed Lemmatised Token Frequency",
    )
    ax.loglog(
        lemma_fit["ranks"],
        lemma_fit["fitted"],
        "-",
        color=COLORS["e5_ft"],
        linewidth=2,
        label="Fitted Power Law",
    )
    ax.set_xlabel("Token Rank")
    ax.set_ylabel("Frequency")
    ax.grid(True, which="both", linewidth=0.4, color="#DEDEDE")
    ax.legend(loc="upper right")
    ax.text(
        0.06,
        0.08,
        "\n".join(
            [
                r"$f(r)\propto r^{-\alpha}$",
                rf"$\alpha_{{\mathrm{{raw}}}}={raw_fit['alpha']:.3f}\;(R^2={raw_fit['r2']:.3f})$",
                rf"$\alpha_{{\mathrm{{lemmatised}}}}={lemma_fit['alpha']:.3f}\;(R^2={lemma_fit['r2']:.3f})$",
            ]
        ),
        transform=ax.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, OUTPUT_DIR / "fig01_zipf_law")


def plot_heaps_law(chunks: list[dict]) -> None:
    fit = fit_heaps_law(chunks)
    print(f"Verified Heaps K: {fit['k']:.6f}, beta = {fit['beta']:.6f}, R^2 = {fit['r2']:.6f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(fit["n_tokens"], fit["vocab"], color=COLORS["e5_base"], linewidth=2, label="Observed Vocabulary Size")
    ax.plot(fit["n_tokens"], fit["fitted"], color=COLORS["gte_ft"], linewidth=2, linestyle="--", label="Fitted Heaps Law")
    ax.set_xlabel(r"Cumulative Token Count $N$")
    ax.set_ylabel(r"Cumulative Vocabulary Size $V(N)$")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="upper left")
    ax.text(
        0.58,
        0.13,
        rf"$V(N)=K N^{{\beta}}$" "\n" rf"$K={fit['k']:.2f},\;\beta={fit['beta']:.3f}$",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, OUTPUT_DIR / "fig02_heaps_law")


def plot_entropy_distribution(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.hist(
        stats["entropy_bits"],
        bins=34,
        color=COLORS["e5_ft"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    mean_entropy = stats["entropy_bits"].mean()
    std_entropy = stats["entropy_bits"].std(ddof=1)
    print(f"Verified Shannon mean H: {mean_entropy:.6f} bits/word; sigma = {std_entropy:.6f} bits/word")
    ax.axvline(mean_entropy, color=COLORS["ink"], linewidth=1.5, linestyle="--", label=rf"Mean $\bar{{H}}={mean_entropy:.2f}$")
    ax.set_xlabel(r"Chunk Entropy $H$ (bits per word)")
    ax.set_ylabel("Number of Chunks")
    ax.legend(loc="upper left")
    low_entropy_count = int((stats["entropy_bits"] < 1.0).sum())
    if low_entropy_count:
        ax.annotate(
            "boilerplate",
            xy=(0.5, low_entropy_count),
            xytext=(1.2, max(low_entropy_count + 15, 25)),
            arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1},
            fontsize=11,
            color=COLORS["ink"],
        )
    ax.text(
        0.56,
        0.78,
        rf"$\bar{{H}}={mean_entropy:.2f}$ bits/word" "\n" rf"$\sigma={std_entropy:.3f}$ bits/word",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, OUTPUT_DIR / "fig03_entropy_distribution")


def entropy_profile_dataframe(chunks: list[dict]) -> pd.DataFrame:
    rows = []
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown")
        chunk_index = chunk_counts.get(doc_id, 0)
        chunk_counts[doc_id] = chunk_index + 1
        tokens = tokenize(chunk.get("text", ""))
        rows.append(
            {
                "doc_id": doc_id,
                "section_chunk_index": chunk_index,
                "section_in_doc": chunk.get("section_in_doc"),
                "entropy_bits": shannon_entropy(tokens),
            }
        )

    frame = pd.DataFrame(rows)
    frame["doc_chunk_count"] = frame.groupby("doc_id")["section_chunk_index"].transform("max") + 1
    frame["document_position"] = np.where(
        frame["doc_chunk_count"] > 1,
        frame["section_chunk_index"] / (frame["doc_chunk_count"] - 1),
        0.0,
    )
    return frame


def plot_mean_entropy_profile(chunks: list[dict]) -> None:
    profile = entropy_profile_dataframe(chunks)
    grid = np.linspace(0.0, 100.0, 101)
    interpolated = []
    for _, doc_profile in profile.groupby("doc_id", sort=True):
        doc_profile = doc_profile.sort_values("document_position")
        smoothed = doc_profile["entropy_bits"].rolling(window=9, center=True, min_periods=1).mean()
        interpolated.append(
            np.interp(
                grid,
                doc_profile["document_position"].to_numpy(dtype=float) * 100,
                smoothed.to_numpy(dtype=float),
            )
        )

    curves = np.vstack(interpolated)
    mean_entropy = curves.mean(axis=0)
    std_entropy = curves.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(
        grid,
        mean_entropy,
        color=COLORS["e5_ft"],
        linewidth=2.4,
        label="Mean Entropy Profile",
    )
    ax.fill_between(
        grid,
        mean_entropy - std_entropy,
        mean_entropy + std_entropy,
        color=COLORS["e5_ft"],
        alpha=0.16,
        linewidth=0,
        label=r"$\pm 1$ document std.",
    )
    ax.set_xlabel("Normalised Section Chunk Position (%)")
    ax.set_ylabel(r"Shannon Entropy $H$ (bits per word)")
    ax.set_title("Mean Entropy Profile Across Documents")
    ax.legend(loc="lower right")
    save_figure(fig, OUTPUT_DIR / "fig04_mean_entropy_profile")


def main() -> None:
    configure_plots()
    chunks = read_jsonl(SECTION_CHUNKS)
    stats = section_chunk_statistics()
    plot_zipf_law()
    plot_heaps_law(chunks)
    plot_entropy_distribution(stats)
    plot_mean_entropy_profile(chunks)


if __name__ == "__main__":
    main()
