from pathlib import Path
from typing import List

from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)


def load_stopwords(base_dir: Path) -> List[str]:
    """
    Load NLTK English stopwords and optional custom legal stopwords.
    """
    stops = stopwords.words("english")

    legal_file = base_dir / "configs" / "stopwords" / "stopwords_legal.txt"
    if legal_file.exists():
        with legal_file.open("r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stops.append(word)
        print(f"Successfully loaded {len(stops)} stopwords")
    else:
        print(f">>> No custom stopwords found in {legal_file} - only using NLTK stopwords <<<")

    return stops