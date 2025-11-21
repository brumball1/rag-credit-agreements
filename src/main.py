from pathlib import Path

from extract.chunk_pages import chunk_pages
# from extract.chunk_paragraphs import chunk_paragraphs   # future
# from rag.semantic_chunking import agentic_chunk          # future

from stats.stopwords import load_stopwords
from stats.word_freq import word_frequency, save_word_bank
from stats.ngrams import (
    get_bigrams,
    get_bigrams_per_page,
    get_trigrams,
    get_trigrams_per_page,
)

# from rag.embeddings import embed_mpnet, embed_bge_m3      # future


class PipelineConfig:
    def __init__(
        self,
        chunking="page",
        embedding="all-mpnet-base-v2",
    ):
        self.chunking = chunking
        self.embedding = embedding


def main(config: PipelineConfig):

    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"
    derived_dir = base_dir / "data" / "derived"

    # -----------------------
    # CHUNKING
    # -----------------------
    if config.chunking == "page":
        chunk_pages(processed_dir, derived_dir / "page_chunks.jsonl")

    # elif config.chunking == "paragraph":
    #     chunk_paragraphs(processed_dir, derived_dir / "page_chunks.jsonl")

    # elif config.chunking == "agentic":
    #     agentic_chunk(processed_dir, derived_dir / "page_chunks.jsonl")

    # -----------------------
    # STOPWORDS
    # -----------------------
    stops = load_stopwords(base_dir)

    # -----------------------
    # N-GRAMS
    # -----------------------
    get_bigrams(processed_dir, stops, derived_dir / "bigrams.csv")
    get_bigrams_per_page(processed_dir, stops, derived_dir / "bigrams_per_page.csv")
    get_trigrams(processed_dir, stops, derived_dir / "trigrams.csv")
    get_trigrams_per_page(processed_dir, stops, derived_dir / "trigrams_per_page.csv")

    # -----------------------
    # WORD FREQUENCIES
    # -----------------------
    freq = word_frequency(processed_dir, stops)
    save_word_bank(freq, derived_dir)

    # -----------------------
    # EMBEDDINGS (later)
    # -----------------------
    # if config.embedding == "all-mpnet-base-v2":
    #     embed_mpnet(derived_dir / "page_chunks.jsonl")

    # elif config.embedding == "bge-m3":
    #     embed_bge_m3(derived_dir / "page_chunks.jsonl")


if __name__ == "__main__":
    cfg = PipelineConfig()
    main(cfg)