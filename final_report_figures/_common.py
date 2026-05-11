from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = REPO_ROOT / "data" / "derived"
FIGURE_ROOT = REPO_ROOT / "final_report_figures"
LOCAL_CACHE = REPO_ROOT / "_local_do_not_commit" / "matplotlib"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE))
XDG_CACHE = REPO_ROOT / "_local_do_not_commit" / "cache"
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "ink": "#263238",
    "muted": "#6B7280",
    "grid": "#D9DEE3",
    "e5_base": "#577590",
    "e5_ft": "#2A9D8F",
    "gte_base": "#9C6644",
    "gte_ft": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#6D597A",
}

SECTION_CHUNKS = DERIVED_DIR / "section_chunks.jsonl"
E5_SECTION_TRAINING_ROOT = (
    DERIVED_DIR
    / "training"
    / "gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20"
)


def configure_plots() -> None:
    """Apply a clean, publication-quality visual language across all final figures."""
    plt.rcParams.update(
        {
            # Canvas
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # Spines — left/bottom only, slightly heavier than default
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Labels and titles
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            # Grid — subtle, behind data
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DEDEDE",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            # Ticks — outward, physics convention
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 11,
            # Font
            "font.family": "DejaVu Sans",
            "font.size": 11,
            # Lines
            "lines.linewidth": 1.8,
            # Save
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def section_output_dir(section_name: str) -> Path:
    out_dir = FIGURE_ROOT / section_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int = 400) -> tuple[Path, Path]:
    """Save every report figure as PNG for review and PDF for insertion."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def shannon_entropy(tokens: Iterable[str]) -> float:
    values = list(tokens)
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for token in values:
        counts[token] = counts.get(token, 0) + 1
    probs = np.array(list(counts.values()), dtype=float) / len(values)
    return float(-(probs * np.log2(probs)).sum())


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - residual / total if total else float("nan")


def fit_zipf_power_law(word_bank_path: Path, top_n: int = 500) -> dict[str, float | np.ndarray]:
    frame = pd.read_csv(word_bank_path)
    freqs = np.sort(frame["count"].to_numpy(dtype=float))[::-1][:top_n]
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    fitted = np.exp(intercept + slope * log_ranks)
    return {
        "alpha": float(-slope),
        "intercept": float(intercept),
        "r2": r_squared(log_freqs, np.log(fitted)),
        "ranks": ranks,
        "freqs": freqs,
        "fitted": fitted,
    }


def section_chunk_statistics() -> pd.DataFrame:
    rows = []
    for chunk in read_jsonl(SECTION_CHUNKS):
        tokens = tokenize(chunk.get("text", ""))
        rows.append(
            {
                "doc_id": chunk.get("doc_id", "unknown"),
                "global_section": chunk.get("global_section"),
                "token_count": chunk.get("token_count", len(tokens)),
                "lexical_token_count": len(tokens),
                "unique_tokens": len(set(tokens)),
                "entropy_bits": shannon_entropy(tokens),
            }
        )
    return pd.DataFrame(rows)


def heaps_observed(chunks: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    chunks = read_jsonl(SECTION_CHUNKS) if chunks is None else chunks
    seen: set[str] = set()
    cumulative_tokens: list[int] = []
    vocab_sizes: list[int] = []
    total = 0
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        total += len(tokens)
        seen.update(tokens)
        if total > 0:
            cumulative_tokens.append(total)
            vocab_sizes.append(len(seen))
    return np.asarray(cumulative_tokens, dtype=float), np.asarray(vocab_sizes, dtype=float)


def fit_heaps_law(chunks: list[dict] | None = None) -> dict[str, float | np.ndarray]:
    from scipy.optimize import curve_fit

    n_tokens, vocab = heaps_observed(chunks)
    mask = (n_tokens > 0) & (vocab > 0)

    def heaps_fn(n: np.ndarray, k_value: float, beta_value: float) -> np.ndarray:
        return k_value * np.power(n, beta_value)

    popt, _ = curve_fit(heaps_fn, n_tokens[mask], vocab[mask], p0=[50.0, 0.35], maxfev=10000)
    k_fit, beta_fit = [float(v) for v in popt]
    fitted = heaps_fn(n_tokens, k_fit, beta_fit)
    return {
        "k": k_fit,
        "beta": beta_fit,
        "r2": r_squared(vocab[mask], fitted[mask]),
        "n_tokens": n_tokens,
        "vocab": vocab,
        "fitted": fitted,
    }
