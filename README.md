# RAG for Credit Agreements

This repository contains the code, cached data, trained fold outputs and final-report figures for a project on improving Retrieval-Augmented Generation (RAG) over long credit agreements.

The project asks whether statistical-mechanics diagnostics and locally generated synthetic supervision can improve legal-document retrieval while reducing reliance on large downstream context windows. The final pipeline uses section-aware chunking, Gemma3-12B synthetic query generation, hybrid dense/BM25 hard negatives, and E5-base-v2 fine-tuning with Multiple Negatives Ranking Loss (MNRL).

## Headline Results

Five credit agreements were evaluated with leave-one-document-out cross-validation. E5-base-v2 was compared before and after fine-tuning on section-aware hybrid triplets.

| Metric | Base E5 | Fine-tuned E5 | Change |
|---|---:|---:|---:|
| Recall@10 | 0.637 +/- 0.040 | 0.797 +/- 0.027 | +0.160 |
| Recall@5 | 0.528 +/- 0.037 | 0.680 +/- 0.032 | +0.152 |
| Recall@1 | 0.265 +/- 0.029 | 0.331 +/- 0.015 | +0.066 |
| MRR | 0.387 +/- 0.031 | 0.483 +/- 0.016 | +0.096 |
| Mean rank | 45.96 +/- 10.47 | 14.77 +/- 3.15 | -31.19 |

At the MNRL operating point, tau = 0.05:

| Diagnostic | Base E5 | Fine-tuned E5 |
|---|---:|---:|
| Retrieval entropy | 10.49 bits | 4.75 bits |
| Top-5 probability mass | 0.013 | 0.589 |
| Top-10 probability mass | 0.023 | 0.687 |

For an 80% recall target, the estimated context requirement fell from about 35 chunks, or 9,900 tokens, to about 11 chunks, or 3,080 tokens. This is a roughly 68% reduction in required context tokens.

## Corpus and Training Data

The corpus consists of five credit agreements supplied for the project. After PDF extraction and cleaning, the report analyses 448,331 cleaned tokens. For corpus statistics, stop-word filtering and lemmatisation reduce this to 244,557 tokens and 4,824 unique tokens. Stop words are kept for triplet generation, negative mining and model fine-tuning because E5 was pre-trained on natural text.

| Dataset artifact | Count | Notes |
|---|---:|---|
| `paragraph_chunks.jsonl` | 2,137 chunks | Original paragraph strategy; many chunks exceeded E5's 512-token limit. |
| `section_chunks.jsonl` | 1,582 chunks | Section-aware chunks used for the final E5 results. |
| Section chunks used for query generation | 1,565 chunks | 17 very short chunks removed before synthetic-query generation. |
| Synthetic query-positive pairs | 6,260 pairs | Four Gemma3-12B queries per retained section chunk. |
| Final E5 section-aware triplets | 12,520 triplets | One dense hard negative and one BM25 hard negative per query-positive pair. |

Corpus statistics reported in the final write-up:

| Statistic | Value |
|---|---:|
| Zipf alpha, raw top-500 tokens | 0.995, R^2 = 0.996 |
| Zipf alpha, lemmatised/filtered top-500 tokens | 0.805, R^2 = 0.975 |
| Heaps law | V(N) = 84.70 N^0.327, R^2 = 0.984 |
| Mean section entropy | 6.015 bits per word |
| Median section entropy | 6.208 bits per word |
| Section entropy standard deviation | 0.735 bits per word |

## Method Summary

The final report follows this pipeline:

1. Extract and clean text from five PDF credit agreements using `pdfplumber`.
2. Split documents using section-aware chunking with a 512-token cap.
3. Generate four local synthetic queries per retained section chunk using Gemma3-12B via Ollama.
4. Mine hard negatives using dense E5 retrieval and BM25, combined through a hybrid strategy.
5. Build five leave-one-document-out folds.
6. Fine-tune E5-base-v2 with MNRL, batch size 16, learning rate 2e-5, MNRL scale 20.0 and early-stop patience 3.
7. Evaluate retrieval metrics and thermodynamic diagnostics on the held-out agreement in each fold.

GTE-ModernBERT-base was tested as an auxiliary model because of its longer context length, but only fold 1 was fine-tuned/evaluated due to time constraints. It is not treated as a five-fold comparison.

## Per-Fold E5 Results

| Fold | Test triplets | Base R@10 | Fine-tuned R@10 | Base MRR | Fine-tuned MRR | Fine-tuned mean rank |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,200 | 0.621 | 0.798 | 0.377 | 0.491 | 14.9 |
| 2 | 1,744 | 0.642 | 0.774 | 0.393 | 0.468 | 16.7 |
| 3 | 3,800 | 0.604 | 0.776 | 0.359 | 0.467 | 17.5 |
| 4 | 1,608 | 0.703 | 0.842 | 0.437 | 0.506 | 9.5 |
| 5 | 3,168 | 0.613 | 0.793 | 0.369 | 0.485 | 15.3 |

Query-level rank movement after fine-tuning:

| Category | Count |
|---|---:|
| Total queries | 12,520 |
| Improved | 6,856 |
| Unchanged | 3,052 |
| Worsened | 2,612 |
| Median rank improvement | +1 |
| Mean rank improvement | +30.7 |
| Largest improvement | +1,428 ranks |
| Largest worsening | -571 ranks |

## Final Report Figures

The `final_report_figures/` directory contains plots used in the final report, plus small cached data tables needed to regenerate them. Figure filenames are numbered in final-report order.

| Script | Figures generated |
|---|---|
| `final_report_figures/corpus_structure.py` | `fig01_zipf_law`, `fig02_heaps_law`, `fig03_entropy_distribution`, `fig04_mean_entropy_profile` |
| `final_report_figures/e5_before_after.py` | `fig05_e5_main_performance_summary`, `fig06_e5_fold_consistency`, `fig07_e5_margin_sharpening`, `fig08_e5_query_rank_improvement_waterfall`, `fig09_e5_thermodynamic_diagnostics`, `fig10_e5_similarity_separation`, `fig11_e5_context_tokens_for_target_recall` |
| `final_report_figures/appendix_support.py` | `fig12_chunk_length_comparison`, `fig13_top_word_frequency_comparison` |

Regenerate the final-report figures:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 final_report_figures/corpus_structure.py
PYTHONDONTWRITEBYTECODE=1 python3 final_report_figures/e5_before_after.py
PYTHONDONTWRITEBYTECODE=1 python3 final_report_figures/appendix_support.py
```

Each figure is saved as both `.png` and `.pdf`.

## Setup

Clone the repo:

```bash
git clone https://github.com/brumball1/rag-credit-agreements
cd rag-credit-agreements
```

Pull cached training data, if using Git LFS:

```bash
git lfs install
git lfs pull
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the Pipeline

The cached pairs and triplets are already present under `data/derived/`, so the full local LLM generation step does not need to be repeated for normal use.

### 1. Build Triplets

Section chunks with hybrid dense/BM25 negatives:

```bash
python3 -m src.training.build_training_triplets \
  --chunks data/derived/section_chunks.jsonl \
  --pairs data/derived/pairs_cache__gemma3-12b__section_chunks__all__4q.jsonl \
  --strategy dense_bm25_window \
  --window_min 3 --window_max 20
```

Paragraph chunks with the original window-negative strategy:

```bash
python3 -m src.training.build_training_triplets \
  --chunks data/derived/paragraph_chunks.jsonl \
  --pairs data/derived/pairs_cache__gemma3-12b__allchunks.jsonl \
  --strategy window \
  --window_min 2 --window_max 15
```

### 2. Create Folds

```bash
python3 -m src.training.create_folds \
  data/derived/triplets__gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20.jsonl \
  --chunks data/derived/section_chunks.jsonl
```

This creates `data/derived/training/<dataset>/fold_1/` through `fold_5/`, each with `train.jsonl`, `val.jsonl`, and `test.jsonl`.

### 3. Train E5

```bash
for fold in 1 2 3 4 5; do
  python3 -m src.training.train_model \
    data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_$fold \
    --corpus_path data/derived/section_chunks.jsonl \
    --model_name intfloat/e5-base-v2 \
    --loss multiple_negatives_ranking \
    --batch_size 16 \
    --learning_rate 2e-5
done
```

The best checkpoint is saved to `fold_N/weights_multiple_negatives_ranking_best/`.

### 4. Test a Fold

```bash
python3 -m src.training.test_fine_tuned \
  data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_1 \
  --chunks data/derived/section_chunks.jsonl \
  --base_model intfloat/e5-base-v2
```

## Project Structure

```text
src/
  extract/              PDF parsing, cleaning, and chunking
  training/             triplet generation, fold creation, training, evaluation
  rag/                  embedding and similarity helpers
  stats/                token, word-frequency, n-gram, and plotting utilities
data/
  raw/                  original PDFs, not tracked
  derived/              chunks, cached pairs, triplets, folds, model outputs
configs/
  stopwords/            legal stop-word lists
final_report_figures/   cleaned final-report plotting scripts and figure outputs
```

## Notes

- Section-aware chunking is the final strategy used for the reported E5 results because it preserves legal sections and enforces the 512-token E5 limit.
- Paragraph chunking is retained as the original baseline strategy, but it regularly exceeded the E5 token limit and caused truncation.
- Hybrid negative mining combines dense and BM25 negatives, which is useful for legal text where many clauses share the same surface vocabulary but differ semantically.
- The five-fold E5 evaluation is the main result. GTE results are auxiliary because GTE was only fine-tuned/evaluated on fold 1.
