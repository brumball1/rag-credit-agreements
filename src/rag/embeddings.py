import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def read_records(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def embed_all_mpnet_base_v2(jsonl_path: Path):
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(f"sentence-transformers/{model_name}")

    records = list(read_records(jsonl_path))
    texts = [rec["text"] for rec in records]

    print(f"Embedding {len(texts)} paragraphs from {jsonl_path}")
    embeddings = model.encode(texts,batch_size=16,show_progress_bar=True)

    out_dir = jsonl_path.parent / "embeddings" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            f.write(json.dumps({
                "id": i,
                "doc_id": rec["doc_id"],
                "paragraph_in_doc": rec["paragraph_in_doc"],
                "token_count": rec["token_count"],
                "page_number": rec["page_number"] #not needed but leavig in just incase
            }) + "\n")

    print(f"Saved embeddings: {out_dir / 'embeddings.npy'} shape={embeddings.shape}")
    print(f"Saved metadata: {out_dir / 'metadata.jsonl'}")
    print(embeddings)
    return embeddings