from collections import Counter
from pathlib import Path
from typing import List

from nltk import bigrams as nltk_bigrams, trigrams as nltk_trigrams

from src.extract.page_reader import iterate_pages
from src.stats.tokeniser import tokeniser


def get_bigrams(input_dir: Path, stops: List[str], output_path: Path) -> None:
    """
    Global bigram counts over the whole corpus.
    """
    bigram_counter = Counter()

    for name, page in iterate_pages(input_dir):
        tokens = tokeniser(page.get("text", ""))
        filtered_tokens = [t for t in tokens if t not in stops]

        for bg in nltk_bigrams(filtered_tokens):
            bigram_counter[bg] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Bigram,Count\n")
        for (w1, w2), count in bigram_counter.most_common():
            bigram_text = f"{w1} {w2}"
            f.write(f"\"{bigram_text}\",{count}\n")

    print(f"Saved {len(bigram_counter)} bigrams to {output_path}")


def get_bigrams_per_page(input_dir: Path, stops: List[str], output_path: Path) -> None:
    """
    Bigram counts per page. Each row has document id, page, bigram, count.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Document,Page,Bigram,Count\n")

        for name, page in iterate_pages(input_dir):
            doc_id = name.replace(".pages_clean.pages.jsonl", "")
            page_number = page.get("page")
            tokens = tokeniser(page.get("text", ""))

            filtered_tokens = [t for t in tokens if t not in stops]
            bigrams_per_page = Counter(nltk_bigrams(filtered_tokens))

            for (w1, w2), count in bigrams_per_page.items():
                bigram_text = f"{w1} {w2}"
                f.write(f"\"{doc_id}\",{page_number},\"{bigram_text}\",{count}\n")


def get_trigrams(input_dir: Path, stops: List[str], output_path: Path) -> None:
    """
    Global trigram counts over the whole corpus.
    """
    trigram_counter = Counter()

    for name, page in iterate_pages(input_dir):
        tokens = tokeniser(page.get("text", ""))
        filtered_tokens = [t for t in tokens if t not in stops]

        for tg in nltk_trigrams(filtered_tokens):
            trigram_counter[tg] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Trigram,Count\n")
        for (w1, w2, w3), count in trigram_counter.most_common():
            trigram_text = f"{w1} {w2} {w3}"
            f.write(f"\"{trigram_text}\",{count}\n")

    print(f"Saved {len(trigram_counter)} trigrams to {output_path}")


def get_trigrams_per_page(input_dir: Path, stops: List[str], output_path: Path) -> None:
    """
    Trigram counts per page. Each row has document id, page, trigram, count.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Document,Page,Trigram,Count\n")

        for name, page in iterate_pages(input_dir):
            doc_id = name.replace(".pages_clean.pages.jsonl", "")
            page_number = page.get("page")
            tokens = tokeniser(page.get("text", ""))

            filtered_tokens = [t for t in tokens if t not in stops]
            trigrams_per_page = Counter(nltk_trigrams(filtered_tokens))

            for (w1, w2, w3), count in trigrams_per_page.items():
                trigram_text = f"{w1} {w2} {w3}"
                f.write(f"\"{doc_id}\",{page_number},\"{trigram_text}\",{count}\n")