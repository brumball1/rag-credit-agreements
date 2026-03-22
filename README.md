# RAG for Credit Agreements

Fine-tuning embedding models on credit agreement documents using hard negative triplets and Multiple Negatives Ranking Loss (MNRL).

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

## Pipeline

After cloning you'll have the cached positive pairs already in `data/derived/`. You don't need to re-run the LLM query generation step. The steps below are:

1. Build triplets (pick your negative sampling strategy)
2. Evaluate the triplets (optional but useful)
3. Create folds
4. Train
5. Test

---

### 1. Build Triplets

This generates negatives for the cached positive pairs. Open `src/training/build_training_triplets.py` and set the strategy at the top of the file:

```python
NEGATIVE_STRATEGY = "random_window"   # recommended — samples from rank 2-15
# NEGATIVE_STRATEGY = "threshold"     # similarity-score based filtering
```

Then run:
```bash
python -m src.training.build_training_triplets
```

This will save a file like `data/derived/triplets__gemma3-12b__e5-base-v2__allchunks__window2-15.jsonl`.

---

### 2. Evaluate the Triplets (optional but useful)

Before training, you can check the quality of the triplets — whether the positives rank highly and whether the negatives are actually hard:

```bash
python -m src.training.evaluate_pairs
```

This prints a comparison table across any triplet files found in `data/derived/`, showing MRR, Recall@1/5/10, and margin stats.

---

### 3. Create Folds

This does leave-one-document-out cross-validation — each fold holds out one credit agreement as the test set:

```bash
python -m src.training.create_folds \
  data/derived/triplets__gemma3-12b__e5-base-v2__allchunks__window2-15.jsonl
```

This creates `data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1/`, `fold_2/`, etc., each with `train.jsonl`, `val.jsonl`, `test.jsonl`.

---

### 4. Train

Trains `intfloat/e5-base-v2` with MNRL. Evaluates Recall@10 on the val set after each epoch and stops early if it doesn't improve for 3 epochs in a row.

Mac:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
caffeinate python -m src.training.train_model \
  data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1 \
  --loss multiple_negatives_ranking \
  --batch_size 16 \
  --learning_rate 2e-5
```

Linux / Windows:
```bash
python -m src.training.train_model \
  data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1 \
  --loss multiple_negatives_ranking \
  --batch_size 16 \
  --learning_rate 2e-5
```

The best checkpoint is saved to `fold_1/weights_multiple_negatives_ranking_best/`.

Other options:
```
--loss triplet              # use triplet loss instead
--batch_size 8              # reduce if you run out of memory
--max_epochs 100            # default
--patience 3                # epochs without improvement before stopping
```

---

### 5. Test

Compares the base model vs. your fine-tuned models on the held-out test set:

```bash
python -m src.training.test_fine_tuned \
  data/derived/training/gemma3-12b__e5-base-v2__allchunks__window2-15/fold_1
```

Prints a table with MRR, Recall@1/5/10, and mean rank for Base, MNRL fine-tuned, and Triplet fine-tuned side by side.

---

## Project Structure

```
src/
  extract/    # PDF parsing and chunking
  training/   # triplet generation, fold creation, training, evaluation
  rag/        # embedding + retrieval
  stats/      # word freq, ngrams, plotting
data/
  raw/        # original PDFs (not tracked)
  derived/    # chunks, pairs cache, triplets, folds (LFS tracked)
```
