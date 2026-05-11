from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    COLORS,
    DERIVED_DIR,
    configure_plots,
    read_jsonl,
    plt,
    save_figure,
    section_output_dir,
)


OUTPUT_DIR = section_output_dir("appendix_support")


def readable_doc_label(doc_id: str) -> str:
    tail = str(doc_id).rsplit("_", 1)[-1]
    return f"Agreement {tail}" if tail.isdigit() else str(doc_id)


def plot_top_word_frequency(top_n: int = 20) -> None:
    word_banks = [
        ("Raw Tokens", DERIVED_DIR / "word_bank.raw.csv", COLORS["e5_base"]),
        ("Cleaned Tokens", DERIVED_DIR / "word_bank.csv", COLORS["e5_ft"]),
        ("Lemmatised Tokens", DERIVED_DIR / "word_bank.lemma.csv", COLORS["gte_ft"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, path, color) in zip(axes, word_banks):
        frame = pd.read_csv(path).head(top_n).iloc[::-1]
        ax.barh(frame["words"], frame["count"], color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x")
    save_figure(fig, OUTPUT_DIR / "fig13_top_word_frequency_comparison")


def chunk_length_frame(path) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path))
    frame["doc_label"] = frame["doc_id"].map(readable_doc_label)
    frame["token_count"] = pd.to_numeric(frame["token_count"], errors="coerce")
    return frame.dropna(subset=["token_count"])


def _draw_chunk_length_panel(ax, frame: pd.DataFrame, title: str, fill_colour: str) -> None:
    doc_labels = sorted(frame["doc_label"].unique())
    values = [frame.loc[frame["doc_label"] == label, "token_count"].to_numpy(dtype=float) for label in doc_labels]
    box = ax.boxplot(
        values,
        tick_labels=doc_labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.3},
        boxprops={"linewidth": 1.0, "color": COLORS["ink"]},
        whiskerprops={"linewidth": 1.0, "color": COLORS["muted"]},
        capprops={"linewidth": 1.0, "color": COLORS["muted"]},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(fill_colour)
        patch.set_alpha(0.36)
    rng = np.random.default_rng(7)
    for idx, vals in enumerate(values, start=1):
        if len(vals) > 500:
            vals = rng.choice(vals, size=500, replace=False)
        jitter = rng.normal(loc=idx, scale=0.035, size=len(vals))
        ax.scatter(jitter, vals, s=8, color=fill_colour, alpha=0.22, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Document")
    ax.tick_params(axis="x", rotation=20)


def plot_chunk_length_comparison() -> None:
    section_frame = chunk_length_frame(DERIVED_DIR / "section_chunks.jsonl")
    paragraph_frame = chunk_length_frame(DERIVED_DIR / "paragraph_chunks.jsonl")
    ymax = max(950, float(section_frame["token_count"].max()), float(paragraph_frame["token_count"].max()))
    ymax = np.ceil(ymax / 50) * 50
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _draw_chunk_length_panel(axes[0], section_frame, "(a) Section chunks", COLORS["e5_base"])
    _draw_chunk_length_panel(axes[1], paragraph_frame, "(b) Paragraph chunks", COLORS["e5_ft"])
    axes[0].set_ylabel("Chunk Length (tokens)")
    for ax in axes:
        ax.set_ylim(0, ymax)
    save_figure(fig, OUTPUT_DIR / "fig12_chunk_length_comparison")


def main() -> None:
    configure_plots()
    plot_chunk_length_comparison()
    plot_top_word_frequency()


if __name__ == "__main__":
    main()
