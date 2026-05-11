from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    COLORS,
    DERIVED_DIR,
    E5_SECTION_TRAINING_ROOT,
    configure_plots,
    plt,
    read_jsonl,
    save_figure,
    section_output_dir,
)


OUTPUT_DIR = section_output_dir("e5_before_after")
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SECTION_CHUNKS = DERIVED_DIR / "section_chunks.jsonl"

FOLD_SCORES = DATA_DIR / "e5_fold_scores.csv"
FOLD_METRICS = DATA_DIR / "e5_fold_metrics.csv"
METRIC_SUMMARY = DATA_DIR / "e5_metric_summary.csv"
THERMO_SUMMARY = DATA_DIR / "e5_thermodynamic_summary_by_tau.csv"

CONDITIONS = ["Base E5", "Fine-Tuned E5"]
CONDITION_COLOURS = {
    "Base E5": COLORS["e5_base"],
    "Fine-Tuned E5": COLORS["e5_ft"],
}
DISPLAY_LABELS = {
    "Base E5": "Base E5",
    "Fine-Tuned E5": "Fine-tuned E5",
}


def e5_base_model_path() -> str:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--e5-base-v2" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return "intfloat/e5-base-v2"


def fine_tuned_model_path(fold: int) -> Path:
    return E5_SECTION_TRAINING_ROOT / f"fold_{fold}" / "weights_multiple_negatives_ranking_best"


def load_corpus() -> tuple[list[str], dict[str, int]]:
    chunks = read_jsonl(SECTION_CHUNKS)
    corpus_texts = [row["text"] for row in chunks]
    text_to_idx: dict[str, int] = {}
    for idx, text in enumerate(corpus_texts):
        text_to_idx.setdefault(text, idx)
    return corpus_texts, text_to_idx


def evaluate_condition(
    *,
    fold: int,
    condition: str,
    model_path: str | Path,
    corpus_texts: list[str],
    text_to_idx: dict[str, int],
    temperatures: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sentence_transformers import SentenceTransformer

    fold_dir = E5_SECTION_TRAINING_ROOT / f"fold_{fold}"
    triplets = read_jsonl(fold_dir / "test.jsonl")
    queries = [row["query"] for row in triplets]
    positive_indices = np.array([text_to_idx.get(row["positive"], -1) for row in triplets], dtype=int)
    negative_indices = np.array([text_to_idx.get(row["negative"], -1) for row in triplets], dtype=int)
    valid = (positive_indices >= 0) & (negative_indices >= 0)
    if not bool(valid.all()):
        skipped = int((~valid).sum())
        print(f"Skipping {skipped} rows in fold {fold}, {condition}: positive/negative text not found in corpus.")

    print(f"Loading {condition}, fold {fold}: {model_path}", flush=True)
    model = SentenceTransformer(str(model_path), local_files_only=True)
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embeddings = model.encode(
        queries,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = np.asarray(query_embeddings @ corpus_embeddings.T, dtype=np.float32)

    valid_indices = np.flatnonzero(valid)
    similarities = similarities[valid_indices]
    valid_triplets = [triplets[idx] for idx in valid_indices]
    positive_indices = positive_indices[valid_indices]
    negative_indices = negative_indices[valid_indices]

    row_indices = np.arange(len(valid_triplets))
    positive_scores = similarities[row_indices, positive_indices].astype(float)
    negative_scores = similarities[row_indices, negative_indices].astype(float)
    ranks = (np.sum(similarities > positive_scores[:, None], axis=1) + 1).astype(int)
    sorted_similarities = -np.sort(-similarities.astype(float), axis=1)

    score_rows: list[dict[str, float | int | str]] = []
    thermo_rows: list[dict[str, float | int | str]] = []
    tau_operating = 0.05
    tau_nearest = float(temperatures[np.argmin(np.abs(temperatures - tau_operating))])

    def ordered_probabilities_for_tau(tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scaled_sorted = sorted_similarities / tau
        row_max = scaled_sorted[:, [0]]
        exp_sorted = np.exp(scaled_sorted - row_max)
        denominator = exp_sorted.sum(axis=1, keepdims=True)
        ordered_probs = exp_sorted / denominator
        positive_probs = np.exp((positive_scores / tau) - row_max[:, 0]) / denominator[:, 0]
        log_z = row_max[:, 0] + np.log(denominator[:, 0])
        free_energy = -tau * log_z
        return ordered_probs, positive_probs, free_energy, denominator[:, 0]

    ordered_operating, positive_probability_operating, _, _ = ordered_probabilities_for_tau(tau_nearest)
    operating_entropy = -(ordered_operating * np.log2(np.clip(ordered_operating, 1e-300, None))).sum(axis=1)
    operating_top5 = ordered_operating[:, :5].sum(axis=1)
    operating_top10 = ordered_operating[:, :10].sum(axis=1)

    for local_idx, row in enumerate(valid_triplets):
        score_rows.append(
            {
                "fold": fold,
                "query_id": int(valid_indices[local_idx]),
                "condition": condition,
                "doc_id": row.get("doc_id", ""),
                "negative_type": row.get("negative_type", "unknown"),
                "rank": int(ranks[local_idx]),
                "positive_score": float(positive_scores[local_idx]),
                "negative_score": float(negative_scores[local_idx]),
                "margin": float(positive_scores[local_idx] - negative_scores[local_idx]),
                "retrieval_entropy_tau_0_05": float(operating_entropy[local_idx]),
                "top5_mass_tau_0_05": float(operating_top5[local_idx]),
                "top10_mass_tau_0_05": float(operating_top10[local_idx]),
                "positive_probability_tau_0_05": float(positive_probability_operating[local_idx]),
            }
        )

    count = len(valid_triplets)
    for tau in temperatures:
        tau = float(tau)
        ordered_probs, positive_probs, free_energy, _ = (
            (ordered_operating, positive_probability_operating, None, None)
            if math.isclose(tau, tau_nearest)
            else ordered_probabilities_for_tau(tau)
        )
        if free_energy is None:
            _, _, free_energy, _ = ordered_probabilities_for_tau(tau)
        entropy = -(ordered_probs * np.log2(np.clip(ordered_probs, 1e-300, None))).sum(axis=1)
        top5 = ordered_probs[:, :5].sum(axis=1)
        top10 = ordered_probs[:, :10].sum(axis=1)
        thermo_rows.append(
            {
                "fold": fold,
                "condition": condition,
                "tau": tau,
                "mean_retrieval_entropy_bits": float(np.mean(entropy)),
                "mean_top5_probability_mass": float(np.mean(top5)),
                "mean_top10_probability_mass": float(np.mean(top10)),
                "mean_positive_probability": float(np.mean(positive_probs)),
                "mean_free_energy": float(np.mean(free_energy)),
                "n_queries": count,
            }
        )
    del model, corpus_embeddings, query_embeddings, similarities
    gc.collect()

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(thermo_rows),
    )


def compute_metrics(score_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = score_frame.groupby(["fold", "condition"], sort=True)
    for (fold, condition), group in grouped:
        ranks = group["rank"].to_numpy(dtype=float)
        rows.append(
            {
                "fold": int(fold),
                "condition": condition,
                "n_queries": len(group),
                "Recall@1": float(np.mean(ranks <= 1)),
                "Recall@5": float(np.mean(ranks <= 5)),
                "Recall@10": float(np.mean(ranks <= 10)),
                "MRR": float(np.mean(1.0 / ranks)),
                "Mean Rank": float(np.mean(ranks)),
                "Median Rank": float(np.median(ranks)),
                "Mean Margin": float(group["margin"].mean()),
                "Median Margin": float(group["margin"].median()),
                "Mean Retrieval Entropy tau=0.05": float(group["retrieval_entropy_tau_0_05"].mean()),
                "Mean Top-5 Mass tau=0.05": float(group["top5_mass_tau_0_05"].mean()),
                "Mean Top-10 Mass tau=0.05": float(group["top10_mass_tau_0_05"].mean()),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "Mean Rank",
        "Median Rank",
        "Mean Margin",
        "Mean Retrieval Entropy tau=0.05",
        "Mean Top-5 Mass tau=0.05",
        "Mean Top-10 Mass tau=0.05",
    ]
    rows = []
    for condition, group in metrics.groupby("condition"):
        for metric in metric_cols:
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)),
                    "n_folds": int(group["fold"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_thermo(thermo: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "mean_retrieval_entropy_bits",
        "mean_top5_probability_mass",
        "mean_top10_probability_mass",
        "mean_positive_probability",
        "mean_free_energy",
    ]
    rows = []
    for (condition, tau), group in thermo.groupby(["condition", "tau"]):
        row = {"condition": condition, "tau": float(tau), "n_folds": int(group["fold"].nunique())}
        for col in value_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "tau"])


def ensure_data(recompute: bool = False) -> None:
    required = [FOLD_SCORES, FOLD_METRICS, METRIC_SUMMARY, THERMO_SUMMARY]
    if not recompute and all(path.exists() for path in required):
        return

    temperatures = np.unique(np.concatenate([np.geomspace(0.01, 2.0, 70), np.array([0.05])]))
    corpus_texts, text_to_idx = load_corpus()

    score_frames = []
    thermo_frames = []

    for fold in range(1, 6):
        fold_scores, fold_thermo = evaluate_condition(
            fold=fold,
            condition="Base E5",
            model_path=e5_base_model_path(),
            corpus_texts=corpus_texts,
            text_to_idx=text_to_idx,
            temperatures=temperatures,
        )
        score_frames.append(fold_scores)
        thermo_frames.append(fold_thermo)

        fold_scores, fold_thermo = evaluate_condition(
            fold=fold,
            condition="Fine-Tuned E5",
            model_path=fine_tuned_model_path(fold),
            corpus_texts=corpus_texts,
            text_to_idx=text_to_idx,
            temperatures=temperatures,
        )
        score_frames.append(fold_scores)
        thermo_frames.append(fold_thermo)

    scores = pd.concat(score_frames, ignore_index=True)
    metrics = compute_metrics(scores)
    summary = metric_summary(metrics)
    thermo = aggregate_thermo(pd.concat(thermo_frames, ignore_index=True))

    scores.to_csv(FOLD_SCORES, index=False)
    metrics.to_csv(FOLD_METRICS, index=False)
    summary.to_csv(METRIC_SUMMARY, index=False)
    thermo.to_csv(THERMO_SUMMARY, index=False)
    print(f"Saved E5 before/after data tables in {DATA_DIR}")


def load_scores() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(FOLD_SCORES)


def load_metrics() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(FOLD_METRICS)


def load_summary() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(METRIC_SUMMARY)


def _mean_chunk_tokens() -> float:
    chunks = read_jsonl(SECTION_CHUNKS)
    counts = [row.get("token_count", len(row.get("text", "").split())) for row in chunks]
    return float(np.mean(counts))


def plot_main_performance() -> None:
    summary = load_summary()
    metrics = load_metrics()
    print(metrics[["fold", "condition", "Recall@10", "MRR", "Mean Rank"]].to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [1.45, 0.85]})

    score_metrics = ["Recall@1", "Recall@5", "Recall@10", "MRR"]
    x = np.arange(len(score_metrics))
    width = 0.32
    for offset, condition in [(-width / 2, "Base E5"), (width / 2, "Fine-Tuned E5")]:
        sub = summary[(summary["condition"] == condition) & summary["metric"].isin(score_metrics)]
        sub = sub.set_index("metric").loc[score_metrics].reset_index()
        axes[0].bar(
            x + offset,
            sub["mean"],
            width=width,
            yerr=sub["std"],
            capsize=3,
            color=CONDITION_COLOURS[condition],
            label=DISPLAY_LABELS[condition],
            alpha=0.92,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(score_metrics)
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Retrieval Metric")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="upper left")
    axes[0].set_title("(a) Recall and Reciprocal Rank")

    for idx, condition in enumerate(CONDITIONS):
        sub = summary[(summary["condition"] == condition) & (summary["metric"] == "Mean Rank")].iloc[0]
        axes[1].bar(
            idx,
            sub["mean"],
            yerr=sub["std"],
            capsize=3,
            color=CONDITION_COLOURS[condition],
            alpha=0.92,
        )
        axes[1].text(idx, sub["mean"] + sub["std"] + 1.2, f"{sub['mean']:.1f}", ha="center", va="bottom", fontsize=11)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Base", "Fine-tuned"], rotation=15)
    axes[1].set_ylabel("Mean Positive Rank (lower is better)")
    axes[1].set_title("(b) Rank Compression")

    save_figure(fig, OUTPUT_DIR / "fig05_e5_main_performance_summary")


def plot_margin_sharpening() -> None:
    scores = load_scores()
    scores[["fold", "condition", "margin"]].to_csv(DATA_DIR / "fig07_margin_distribution_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bins = np.linspace(scores["margin"].quantile(0.005), scores["margin"].quantile(0.995), 64)
    for condition in CONDITIONS:
        sub = scores[scores["condition"] == condition]
        median = float(sub["margin"].median())
        ax.hist(
            sub["margin"],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=CONDITION_COLOURS[condition],
            label=f"{DISPLAY_LABELS[condition]} (median {median:.3f})",
        )
        ax.hist(
            sub["margin"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=CONDITION_COLOURS[condition],
        )
        ax.axvline(median, color=CONDITION_COLOURS[condition], linestyle="--", linewidth=1.3)
    ax.axvline(0.0, color=COLORS["ink"], linestyle=":", linewidth=1.3, label="Zero Margin")
    ax.set_xlabel(r"Retrieval Margin $\Delta S = S^+ - S^-$")
    ax.set_ylabel("Probability Density")
    ax.set_title("Retrieval Margin Distribution")
    ax.legend(loc="upper right")
    save_figure(fig, OUTPUT_DIR / "fig07_e5_margin_sharpening")


def plot_similarity_separation() -> None:
    scores = load_scores()
    long = scores.melt(
        id_vars=["fold", "condition"],
        value_vars=["positive_score", "negative_score"],
        var_name="score_type",
        value_name="cosine_similarity",
    )
    long.to_csv(DATA_DIR / "fig10_similarity_distribution_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=True)
    panels = [("positive_score", r"Correct Chunk Similarity $S^+$"), ("negative_score", r"Hard Negative Similarity $S^-$")]
    all_scores = pd.concat([scores["positive_score"], scores["negative_score"]])
    bins = np.linspace(float(all_scores.quantile(0.005)), float(all_scores.quantile(0.995)), 60)
    for ax, (column, label) in zip(axes, panels):
        for condition in CONDITIONS:
            sub = scores[scores["condition"] == condition]
            ax.hist(
                sub[column],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.1,
                color=CONDITION_COLOURS[condition],
                label=DISPLAY_LABELS[condition],
            )
            median = float(sub[column].median())
            ax.axvline(median, color=CONDITION_COLOURS[condition], linestyle="--", linewidth=1.1)
            label_y = 0.92 if condition == "Base E5" else 0.82
            label_x = 0.70 if condition == "Base E5" else median
            ax.text(
                label_x,
                label_y,
                f"median={median:.3f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                color=CONDITION_COLOURS[condition],
                bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.18", "alpha": 0.86},
            )
        ax.set_xlabel(r"Cosine Similarity $S$")
        ax.set_ylabel("Probability Density")
        ax.set_title(label)
    axes[0].legend(loc="upper left")
    save_figure(fig, OUTPUT_DIR / "fig10_e5_similarity_separation")


def plot_thermodynamic_diagnostics() -> None:
    ensure_data()
    thermo = pd.read_csv(THERMO_SUMMARY)
    thermo.to_csv(DATA_DIR / "fig09_thermodynamic_diagnostics_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for condition in CONDITIONS:
        sub = thermo[thermo["condition"] == condition]
        axes[0].plot(sub["tau"], sub["mean_retrieval_entropy_bits_mean"], color=CONDITION_COLOURS[condition], linewidth=2.2, label=DISPLAY_LABELS[condition])
        axes[0].fill_between(
            sub["tau"].to_numpy(dtype=float),
            (sub["mean_retrieval_entropy_bits_mean"] - sub["mean_retrieval_entropy_bits_std"]).to_numpy(dtype=float),
            (sub["mean_retrieval_entropy_bits_mean"] + sub["mean_retrieval_entropy_bits_std"]).to_numpy(dtype=float),
            color=CONDITION_COLOURS[condition],
            alpha=0.12,
        )
        axes[1].plot(sub["tau"], sub["mean_top5_probability_mass_mean"], color=CONDITION_COLOURS[condition], linewidth=2.2, linestyle="-", label=f"{DISPLAY_LABELS[condition]} Top-5")
        axes[1].plot(sub["tau"], sub["mean_top10_probability_mass_mean"], color=CONDITION_COLOURS[condition], linewidth=1.8, linestyle="--", label=f"{DISPLAY_LABELS[condition]} Top-10")

    for ax in axes:
        ax.axvline(0.05, color=COLORS["ink"], linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"Diagnostic Retrieval Temperature $\tau$")
    axes[0].set_ylabel(r"Mean Retrieval Entropy $S_{\mathrm{ret}}$ (bits)")
    axes[0].set_title("(a) Retrieval Disorder")
    axes[0].legend(loc="center right")
    axes[0].text(
        0.052,
        0.12,
        r"$\tau=0.05$",
        transform=axes[0].get_xaxis_transform(),
        fontsize=11,
        color=COLORS["ink"],
        ha="left",
        va="bottom",
    )
    axes[1].set_ylabel("Cumulative Probability Mass")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("(b) Top-k Context Mass")
    axes[1].legend(loc="center right", fontsize=10)
    axes[1].text(
        0.32,
        0.08,
        r"$\tau=0.05$",
        transform=axes[1].transAxes,
        fontsize=11,
        color=COLORS["ink"],
        ha="left",
    )
    fig.subplots_adjust(wspace=0.28)

    save_figure(fig, OUTPUT_DIR / "fig09_e5_thermodynamic_diagnostics")


def plot_context_tokens_for_target_recall() -> None:
    scores = load_scores()
    mean_tokens = _mean_chunk_tokens()
    targets = np.array([0.70, 0.75, 0.80, 0.85, 0.90], dtype=float)
    rows = []
    for (fold, condition), group in scores.groupby(["fold", "condition"]):
        ranks = np.sort(group["rank"].to_numpy(dtype=int))
        for target in targets:
            index = int(math.ceil(target * len(ranks))) - 1
            index = min(max(index, 0), len(ranks) - 1)
            k_required = int(ranks[index])
            rows.append(
                {
                    "fold": int(fold),
                    "condition": condition,
                    "target_recall": target,
                    "k_required": k_required,
                    "estimated_context_tokens": float(k_required * mean_tokens),
                    "mean_chunk_tokens": mean_tokens,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "fig11_context_tokens_for_target_recall_data.csv", index=False)
    summary = (
        frame.groupby(["condition", "target_recall"], as_index=False)
        .agg(
            context_tokens_mean=("estimated_context_tokens", "mean"),
            context_tokens_std=("estimated_context_tokens", "std"),
            k_mean=("k_required", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for condition in CONDITIONS:
        sub = summary[summary["condition"] == condition]
        x_offset = 1.4
        y_offset = -600 if condition == "Base E5" else -200
        ax.errorbar(
            sub["target_recall"] * 100,
            sub["context_tokens_mean"],
            yerr=sub["context_tokens_std"],
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=CONDITION_COLOURS[condition],
            label=DISPLAY_LABELS[condition],
        )
        for _, row in sub.iterrows():
            ax.text(
                row["target_recall"] * 100 + x_offset,
                row["context_tokens_mean"] + y_offset,
                f"k~{row['k_mean']:.0f}",
                ha="center",
                va="top",
                fontsize=11,
                color=CONDITION_COLOURS[condition],
            )
    ax.set_xlabel("Target Recall (%)")
    ax.set_ylabel("Estimated Context Tokens Required")
    ax.set_xticks(targets * 100)
    ax.legend(loc="upper left")
    ax.axvline(80, color=COLORS["ink"], linestyle="--", linewidth=1.1, alpha=0.75)
    save_figure(fig, OUTPUT_DIR / "fig11_e5_context_tokens_for_target_recall")


def plot_query_rank_improvement_waterfall() -> None:
    scores = load_scores()
    base = scores[scores["condition"] == "Base E5"][["fold", "query_id", "rank"]].rename(columns={"rank": "base_rank"})
    ft = scores[scores["condition"] == "Fine-Tuned E5"][["fold", "query_id", "rank"]].rename(columns={"rank": "fine_tuned_rank"})
    frame = base.merge(ft, on=["fold", "query_id"])
    frame["delta_rank"] = frame["base_rank"] - frame["fine_tuned_rank"]
    frame = frame.sort_values("delta_rank", ascending=False).reset_index(drop=True)
    frame["sorted_query_index"] = np.arange(1, len(frame) + 1)
    frame.to_csv(DATA_DIR / "fig08_query_rank_improvement_waterfall_data.csv", index=False)

    improved = int((frame["delta_rank"] > 0).sum())
    worsened = int((frame["delta_rank"] < 0).sum())
    unchanged = int((frame["delta_rank"] == 0).sum())
    median_delta = float(frame["delta_rank"].median())
    total = len(frame)

    fig, ax = plt.subplots(figsize=(10, 4))
    improvement_colour = CONDITION_COLOURS["Fine-Tuned E5"]
    worsening_colour = COLORS["gte_ft"]
    colours = np.where(frame["delta_rank"] >= 0, improvement_colour, worsening_colour)
    ax.bar(
        frame["sorted_query_index"],
        frame["delta_rank"],
        color=colours,
        width=1.0,
        linewidth=0,
        alpha=0.88,
        rasterized=True,
    )
    ax.axhline(0, color=COLORS["ink"], linewidth=1.0)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlim(0.5, total + 0.5)
    ax.set_ylim(-800, 1800)
    ax.set_yticks([-500, -100, -10, 0, 10, 100, 1000])
    ax.set_yticklabels(["-500", "-100", "-10", "0", "10", "100", "1000"])
    ax.set_xlabel(r"Queries sorted by rank improvement $\Delta r$", fontsize=18)
    ax.set_ylabel(r"Rank improvement $\Delta r$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=improvement_colour, linewidth=7, label="Improved or unchanged"),
            plt.Line2D([0], [0], color=worsening_colour, linewidth=7, label="Worsened"),
        ],
        loc="lower left",
        frameon=True,
        handlelength=1.8,
        fontsize=16,
    )
    save_figure(fig, OUTPUT_DIR / "fig08_e5_query_rank_improvement_waterfall")


def plot_fold_consistency() -> None:
    metrics = load_metrics()
    metrics.to_csv(DATA_DIR / "fig06_fold_consistency_data.csv", index=False)

    plot_metrics = ["Recall@10", "MRR"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label in zip(axes, plot_metrics, ["(a)", "(b)"]):
        folds = sorted(metrics["fold"].unique())
        base_vals = []
        ft_vals = []
        for fold in folds:
            base_val = float(metrics[(metrics["fold"] == fold) & (metrics["condition"] == "Base E5")][metric].iloc[0])
            ft_val = float(metrics[(metrics["fold"] == fold) & (metrics["condition"] == "Fine-Tuned E5")][metric].iloc[0])
            base_vals.append(base_val)
            ft_vals.append(ft_val)
            ax.plot([0, 1], [base_val, ft_val], color=COLORS["muted"], linewidth=1.1, alpha=0.55, zorder=1)

        ax.scatter([0] * len(folds), base_vals, color=CONDITION_COLOURS["Base E5"], s=55, zorder=3)
        ax.scatter([1] * len(folds), ft_vals, color=CONDITION_COLOURS["Fine-Tuned E5"], s=55, zorder=3)
        mean_base = np.mean(base_vals)
        mean_ft = np.mean(ft_vals)
        ax.scatter([0], [mean_base], color=CONDITION_COLOURS["Base E5"], s=130, marker="D", zorder=4, edgecolors="white", linewidths=0.9)
        ax.scatter([1], [mean_ft], color=CONDITION_COLOURS["Fine-Tuned E5"], s=130, marker="D", zorder=4, edgecolors="white", linewidths=0.9)

        pct = (mean_ft - mean_base) / mean_base * 100
        arrow_x = 1.22
        ax.annotate(
            "",
            xy=(arrow_x, mean_ft),
            xytext=(arrow_x, mean_base),
            arrowprops=dict(arrowstyle="<->", color=COLORS["ink"], lw=1.2),
        )
        ax.text(
            arrow_x + 0.04,
            (mean_base + mean_ft) / 2,
            f"{pct:+.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            color=COLORS["ink"],
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base E5", "Fine-tuned E5"])
        ax.set_xlim(-0.4, 1.55)
        ax.set_ylabel(metric)
        ax.set_title(f"{label} Per-Fold {metric}")

    save_figure(fig, OUTPUT_DIR / "fig06_e5_fold_consistency")


def main() -> None:
    configure_plots()
    ensure_data()
    plot_main_performance()
    plot_fold_consistency()
    plot_margin_sharpening()
    plot_query_rank_improvement_waterfall()
    plot_thermodynamic_diagnostics()
    plot_similarity_separation()
    plot_context_tokens_for_target_recall()


if __name__ == "__main__":
    main()
