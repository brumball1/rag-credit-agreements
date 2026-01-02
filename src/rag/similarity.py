import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(embeddings_dir: Path):
    embeddings = np.load(embeddings_dir / "embeddings.npy")

    records = []
    with (embeddings_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    df = df.sort_values(["doc_id", "paragraph_in_doc"]).reset_index(drop=True)
    embeddings = embeddings[df.index]
    print(df)
    return df, embeddings

def adjacent_similarity(df, embeddings, outdir: Path):
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("paragraph_in_doc")
        indices = group.index.to_numpy()

        similarities = []
        for i in range(len(indices)-1):
            v1 = embeddings[indices[i]].reshape(1, -1)
            v2 = embeddings[indices[i+1]].reshape(1, -1)

            sim = cosine_similarity(v1, v2)[0][0]
            similarities.append(sim)

        plt.figure(figsize=(14, 4))
        plt.plot(similarities)
        plt.ylim(0, 1)
        plt.title(f"Adjacent Paragraph Similarity – {doc_id}")
        plt.xlabel("Paragraph index")
        plt.ylabel("Cosine similarity")

        outpath = outdir / f"adjacent_similarity_{doc_id}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"Adjacent similarity plot -> {outpath}")

def similarity_heatmap(df, emb, outdir: Path):
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("paragraph_in_doc")
        idxs = group.index.to_numpy()

        sub_emb = emb[idxs]   # shape (N, 768)
        matrix = cosine_similarity(sub_emb)

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="OrRd", interpolation="nearest")
        plt.colorbar(label="Cosine similarity")

        plt.title(f"Similarity Heatmap – {doc_id}")
        plt.xlabel("Paragraph index")
        plt.ylabel("Paragraph index")

        outpath = outdir / f"heatmap_{doc_id}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=500)
        plt.close()

        print(f"Saved heatmap -> {outpath}")


def lag_k_analysis(df, emb, outdir: Path):
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("paragraph_in_doc")
        idxs = group.index.to_numpy()
        n = len(idxs)

        k_values = []
        k = 2
        while k < n:
            k_values.append(k)
            k += 1

        k_values = [k for k in k_values if k < 4]

        plt.figure(figsize=(14, 5))

        for k in k_values:
            sims = []
            for i in range(n - k):
                v1 = emb[idxs[i]].reshape(1, -1)
                v2 = emb[idxs[i + k]].reshape(1, -1)
                sims.append(cosine_similarity(v1, v2)[0][0])

            plt.plot(sims, label=f"k={k}")

        plt.ylim(0, 1)
        plt.xlabel("Paragraph index")
        plt.ylabel("Cosine similarity")
        plt.title(f"Lag Similarity Curves – {doc_id}")
        plt.legend(loc="lower right")

        outpath = outdir / f"lag_similarity_{doc_id}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=400)
        plt.close()

        print(f"Saved lag similarity curves -> {outpath}")


def compute_lag_summary(df, emb, outdir: Path, max_k=256):
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("paragraph_in_doc")
        idxs = group.index.to_numpy()
        n = len(idxs)

        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        ks = [k for k in ks if k < n]

        stats = {"k": [], "mean": [], "median": [], "std": []}

        for k in ks:
            vals = []
            for i in range(n - k):
                v1 = emb[idxs[i]].reshape(1, -1)
                v2 = emb[idxs[i + k]].reshape(1, -1)
                sim = cosine_similarity(v1, v2)[0][0]
                vals.append(sim)

            stats["k"].append(k)
            stats["mean"].append(np.mean(vals))
            stats["median"].append(np.median(vals))
            stats["std"].append(np.std(vals))

        plt.figure(figsize=(10, 5))
        plt.plot(stats["k"], stats["mean"], marker="o", label="mean similarity")
        plt.fill_between(
            stats["k"],
            np.array(stats["mean"]) - np.array(stats["std"]),
            np.array(stats["mean"]) + np.array(stats["std"]),
            color="lightblue",
            alpha=0.4,
            label="±1 std"
        )

        for k_val, mean_val, std_val in zip(stats["k"], stats["mean"], stats["std"]):
            label = f"{mean_val:.2f} ± {std_val:.2f}"
            plt.text(
                k_val,
                mean_val + 0.02,  # small vertical offset
                label,
                ha="center",
                fontsize=8,
                color="black"
            )

        plt.xscale("log", base=2)
        plt.ylim(0, 1)
        plt.xlabel("Lag k (log scale)")
        plt.ylabel("Cosine similarity")
        plt.title(f"Lag-k Similarity Summary - {doc_id}")
        plt.legend()

        outpath = outdir / f"lag_summary_{doc_id}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()


def lag_similarity_lines(df, embeddings, outdir: Path, ks=(1, 2, 3)):
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("paragraph_in_doc")
        indices = group.index.to_numpy()
        n = len(indices)

        for k in ks:
            similarities = []
            for i in range(n - k):
                v1 = embeddings[indices[i]].reshape(1, -1)
                v2 = embeddings[indices[i + k]].reshape(1, -1)
                sim = cosine_similarity(v1, v2)[0][0]
                similarities.append(sim)

            avg_sim = np.mean(similarities)

            plt.figure(figsize=(14, 4))
            plt.plot(similarities, label="Cosine Similarity")
            plt.axhline(
                avg_sim,
                linestyle="--",
                linewidth=1.5,
                color="red",
                label=f"Mean = {avg_sim:.2f}"
            )
            plt.text(
                0.98,
                avg_sim,
                f"{avg_sim:.2f}",
                transform=plt.gca().get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=9,
                color="red"
            )
            plt.ylim(0, 1)
            plt.title(f"Paragraph Similarity (k={k}) – {doc_id}")
            plt.xlabel("Paragraph index")
            plt.ylabel("Cosine similarity")

            outpath = outdir / f"lag_{k}_similarity_{doc_id}.png"
            plt.tight_layout()
            plt.savefig(outpath, dpi=300)
            plt.close()

def similarity_heatmap_target(df, emb, outdir: Path, target_doc_id=None):
    section_starts = {
        "Front matter": 0,
        "Article I – Definitions": 26,
        "Article II – The Credits": 92,
        "Article III – Representations and Warranties": 191,
        "Article IV – Conditions": 215,
        "Article V – Affirmative Covenants": 228,
        "Article VI – Negative Covenants": 247,
        "Article VII – Events of Default": 274,
        "Article VIII – Agents": 284,
        "Article IX – Miscellaneous": 311,
        "Article X – Guarantee": 358,
        "Schedules": 387,
        "Exhibits": 389,
    }

    for doc_id, group in df.groupby("doc_id"):

        if target_doc_id is not None and doc_id != target_doc_id:
            continue

        group = group.sort_values("paragraph_in_doc")
        idxs = group.index.to_numpy()

        sub_emb = emb[idxs]
        matrix = cosine_similarity(sub_emb)

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="OrRd", interpolation="nearest")
        plt.colorbar(label="Cosine similarity")

        for name, idx in section_starts.items():
            plt.axvline(idx, color="blue", linestyle="-", linewidth=1.4, alpha=0.9)

        plt.title(f"Similarity Heatmap with Section Boundaries – {doc_id}")
        plt.xlabel("Paragraph index")
        plt.ylabel("Paragraph index")

        outpath = outdir / f"heatmap_{doc_id}_sections.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=500)
        plt.close()

        print(f"Saved annotated heatmap -> {outpath}")

def run_similarity_analysis(embeddings_dir: Path):
    df, embeddings = load_embeddings(embeddings_dir)

    outdir = embeddings_dir / "similarity"
    outdir.mkdir(exist_ok=True, parents=True)

    adjacent_similarity(df, embeddings, outdir)
    similarity_heatmap(df, embeddings, outdir)
    # lag_k_analysis(df, embeddings, outdir)
    compute_lag_summary(df, embeddings, outdir)
    lag_similarity_lines(df, embeddings, outdir)
    similarity_heatmap_target(df, embeddings, outdir, target_doc_id="lboro_credit_agreement_1")


    print("Done all similarity analysis")

