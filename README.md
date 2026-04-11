# RAG for Credit Agreements

Fine-tuning embedding models on credit agreement documents using LLM-generated training triplets and Multiple Negatives Ranking Loss (MNRL), evaluated with leave-one-document-out cross-validation across five credit agreements.

---

## Results

### E5-base-v2 — Section Chunks (5-fold average)

| Metric | Base Model | Fine-Tuned |
|--------|-----------|------------|
| Recall@10 | 0.638 | **0.797** |
| Recall@1 | 0.265 | 0.327 |
| MRR | 0.387 | 0.481 |
| Mean rank | 45.96 | 14.84 |

Per-fold test results:

| Fold | Triplets | Base Recall@10 | Fine-Tuned Recall@10 |
|------|----------|---------------|---------------------|
| 1 | 2,200 | 0.6209 | 0.7982 |
| 2 | 1,744 | 0.6422 | 0.7741 |
| 3 | 3,800 | 0.6037 | 0.7763 |
| 4 | 1,608 | 0.7027 | 0.8420 |
| 5 | 3,168 | 0.6130 | 0.7929 |

Training: MNRL loss, batch size 16, lr 2e-5, patience=3, early stopped at ~5 epochs.

---

## Setup

**Clone the repo**
```bash
git clone https://github.com/brumball1/rag-credit-agreements
cd rag-credit-agreements
```

**Pull cached training data (requires git-lfs)**

Mac:
```bash
brew install git-lfs
git lfs install && git lfs pull
```

Linux:
```bash
sudo apt install git-lfs
git lfs install && git lfs pull
```

Windows — download git-lfs from https://git-lfs.com, then:
```powershell
git lfs install
git lfs pull
```

**Create a virtual environment and install dependencies**

Mac / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Datasets

After cloning and pulling LFS, `data/derived/` contains:

| File | Description |
|------|-------------|
| `paragraph_chunks.jsonl` | Paragraph-level chunks (original chunking strategy) |
| `section_chunks.jsonl` | Section-aware chunks (improved chunking strategy, 1,582 chunks) |
| `pairs_cache__gemma3-12b__allchunks.jsonl` | LLM-generated positive query pairs for paragraph chunks |
| `pairs_cache__gemma3-12b__section_chunks__all__4q.jsonl` | LLM-generated positive query pairs for section chunks (4 queries per chunk) |
| `triplets__gemma3-12b__e5-base-v2__allchunks__window2-15.jsonl` | Triplets: e5 + paragraph chunks + window negatives (6,686 triplets) |
| `triplets__gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20.jsonl` | Triplets: e5 + section chunks + hybrid negatives (12,520 triplets) |
| `triplets__gemma3-12b__gte-modernbert-base__allchunks__window2-15.jsonl` | Triplets: GTE + paragraph chunks + window negatives (6,686 triplets) |
| `triplets__gemma3-12b__gte-modernbert-base__section_chunks__all__dense__bm25__window3-20.jsonl` | Triplets: GTE + section chunks + hybrid negatives (12,520 triplets) |

You do not need to re-run LLM query generation — the pairs caches are pre-computed. Start from step 3 (build triplets) or step 4 (create folds) if triplets already exist.

---

## Pipeline

### 1. Build Triplets

Generates hard negatives for the cached positive pairs. The negative mining strategy is set via command line:

```bash
# Section chunks with hybrid negatives (recommended)
python -m src.training.build_training_triplets \
  --chunks data/derived/section_chunks.jsonl \
  --pairs data/derived/pairs_cache__gemma3-12b__section_chunks__all__4q.jsonl \
  --strategy dense_bm25_window \
  --window_min 3 --window_max 20

# Paragraph chunks with window negatives (original strategy)
python -m src.training.build_training_triplets \
  --chunks data/derived/paragraph_chunks.jsonl \
  --pairs data/derived/pairs_cache__gemma3-12b__allchunks.jsonl \
  --strategy window \
  --window_min 2 --window_max 15
```

Output is saved to `data/derived/triplets__<generator>__<model>__<chunks>__<strategy>.jsonl`.

---

### 2. Evaluate Triplets (optional)

Check triplet quality before training — MRR, Recall@1/5/10, and margin stats across all triplet files:

```bash
python -m src.training.evaluate_pairs
```

---

### 3. Create Folds

Leave-one-document-out cross-validation. Each fold holds out one credit agreement as the test set. The train/val split is done at chunk level to prevent data leakage (a chunk either goes entirely to train or entirely to val).

```bash
# Section chunks (recommended)
python -m src.training.create_folds \
  data/derived/triplets__gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20.jsonl \
  --chunks data/derived/section_chunks.jsonl

# Paragraph chunks
python -m src.training.create_folds \
  data/derived/triplets__gemma3-12b__e5-base-v2__allchunks__window2-15.jsonl \
  --chunks data/derived/paragraph_chunks.jsonl
```

Creates `data/derived/training/<dataset>/fold_1/` through `fold_5/`, each containing `train.jsonl`, `val.jsonl`, `test.jsonl`.

---

### 4. Train

Trains with MNRL. Evaluates Recall@10 on the val set after each epoch and stops early if no improvement for 3 consecutive epochs.

**E5-base-v2 (section chunks):**

Windows:
```powershell
foreach ($fold in 1..5) {
    .\.venv\Scripts\python.exe -m src.training.train_model `
      "data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_$fold" `
      --corpus_path data/derived/section_chunks.jsonl `
      --model_name intfloat/e5-base-v2 `
      --loss multiple_negatives_ranking `
      --batch_size 16 `
      --learning_rate 2e-5
}
```

Mac / Linux:
```bash
for fold in 1 2 3 4 5; do
  caffeinate python -m src.training.train_model \
    data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_$fold \
    --corpus_path data/derived/section_chunks.jsonl \
    --model_name intfloat/e5-base-v2 \
    --loss multiple_negatives_ranking \
    --batch_size 16 \
    --learning_rate 2e-5
done
```

**GTE-ModernBERT-base (section chunks)** — use a lower learning rate to prevent gradient explosion:

Windows:
```powershell
foreach ($fold in 1..5) {
    .\.venv\Scripts\python.exe -m src.training.train_model `
      "data/derived/training/gemma3-12b__gte-modernbert-base__section_chunks__all__dense__bm25__window3-20/fold_$fold" `
      --corpus_path data/derived/section_chunks.jsonl `
      --model_name Alibaba-NLP/gte-modernbert-base `
      --loss multiple_negatives_ranking `
      --batch_size 16 `
      --learning_rate 5e-6 `
      --max_grad_norm 1.0
}
```

Other options:
```
--loss triplet              # use triplet loss instead of MNRL
--batch_size 8              # reduce if GPU runs out of memory
--max_epochs 100            # default
--patience 3                # epochs without improvement before stopping
```

The best checkpoint is saved to `fold_N/weights_multiple_negatives_ranking_best/`.

---

### 5. Test

Compares base model vs fine-tuned on the held-out test set:

```bash
python -m src.training.test_fine_tuned \
  data/derived/training/gemma3-12b__e5-base-v2__section_chunks__all__dense__bm25__window3-20/fold_1 \
  --chunks data/derived/section_chunks.jsonl \
  --base_model intfloat/e5-base-v2
```

For GTE:
```bash
python -m src.training.test_fine_tuned \
  data/derived/training/gemma3-12b__gte-modernbert-base__section_chunks__all__dense__bm25__window3-20/fold_1 \
  --chunks data/derived/section_chunks.jsonl \
  --base_model Alibaba-NLP/gte-modernbert-base
```

---

## Project Structure

```
src/
  extract/    # PDF parsing and section-aware chunking
  training/   # triplet generation, fold creation, training, evaluation
  rag/        # embedding + retrieval
  stats/      # word frequency, ngrams, plotting
data/
  raw/        # original PDFs (not tracked)
  derived/    # chunks, pairs cache, triplets, folds (LFS tracked)
configs/
  stopwords/  # domain-specific stopword lists for BM25
```

---

## Notes

- **Chunking**: `section_chunks.jsonl` uses a section-aware chunker that respects article/section/sub-clause boundaries. This produces more semantically coherent chunks than paragraph splitting for legal documents.
- **Negative mining**: Hybrid negatives (dense + BM25) produce harder negatives than window sampling alone. BM25 negatives are particularly effective for legal text with high keyword density.
- **Data leakage**: `create_folds.py` splits at chunk level. Earlier versions split at triplet or query level, which caused inflated validation scores. The test set is always held out at document level and is unaffected by this fix.
- **GTE gradient stability**: `Alibaba-NLP/gte-modernbert-base` is sensitive to learning rate with MNRL. Use `--learning_rate 5e-6` and `--max_grad_norm 1.0` to prevent loss collapse.
