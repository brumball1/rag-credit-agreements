from collections import Counter
from pathlib import Path
from typing import List

from extract.page_reader import iterate_pages
from stats.tokeniser import tokeniser, lemmatise


def word_frequency(input_dir: Path, stops: List[str]) -> Counter:
    """
    Compute lemma frequency across all pages, skipping stopwords.
    """
    freq = Counter()
    for name, page in iterate_pages(input_dir):
        tokens = tokeniser(page.get("text", ""))
        filtered_tokens = [t for t in tokens if t not in stops]
        lemmas = lemmatise(filtered_tokens)
        freq.update(lemmas)
    return freq


def save_word_bank(frequency: Counter, output_dir: Path) -> None:
    """
    Save lemma frequencies to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "word_bank.lemma.csv"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("words,count\n")
        for word, count in frequency.most_common():
            f.write(f"{word},{count}\n")

    print(f"Saved word bank to {output_path}")