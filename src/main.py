from pathlib import Path
from extract.chunk_pages import chunk_pages
from extract.chunk_paragraphs import chunk_paragraphs
from rag.embeddings import embed_all_mpnet_base_v2
from stats.stopwords import load_stopwords
from stats.word_freq import word_frequency, save_word_bank
from stats.ngrams import (get_bigrams, get_bigrams_per_page, get_trigrams, get_trigrams_per_page)
from stats.plotting.plot_paragraph_lengths import plot_token_count_per_document
from rag.similarity import (run_similarity_analysis)



class PipelineConfig:
    def __init__(
        self,
        chunking="paragraph",
        embedding="all-mpnet-base-v2",
    ):
        self.chunking = chunking
        self.embedding = embedding


def main(config: PipelineConfig):

    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"
    derived_dir = base_dir / "data" / "derived"
    paragraph_chunks = base_dir / "data" / "derived" / "paragraph_chunks.jsonl"


    # if config.chunking == "page":
    #     chunk_pages(processed_dir, derived_dir / "page_chunks.jsonl")
    #
    # elif config.chunking == "paragraph":
    #     chunk_paragraphs(processed_dir, derived_dir / "paragraph_chunks.jsonl")
    #
    # stops = load_stopwords(base_dir)

    # get_bigrams(processed_dir, stops, derived_dir / "bigrams.csv")
    # get_bigrams_per_page(processed_dir, stops, derived_dir / "bigrams_per_page.csv")
    # get_trigrams(processed_dir, stops, derived_dir / "trigrams.csv")
    # get_trigrams_per_page(processed_dir, stops, derived_dir / "trigrams_per_page.csv")
    # plot_token_count_per_document(paragraph_chunks)

    # freq = word_frequency(processed_dir, stops)
    # save_word_bank(freq, derived_dir)

    if config.embedding == "all-mpnet-base-v2":
        # embed_all_mpnet_base_v2(derived_dir / "paragraph_chunks.jsonl")

        embeddings_dir = derived_dir / "embeddings" / "all-mpnet-base-v2"
        run_similarity_analysis(embeddings_dir)





if __name__ == "__main__":
    cfg = PipelineConfig()
    main(cfg)
