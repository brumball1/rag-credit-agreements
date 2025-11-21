import re
from typing import List

import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def tokeniser(text: str) -> List[str]:
    """
    Lowercase text and extract word tokens using a regular expression.
    """
    return re.findall(r"\b\w+\b", text.lower())


def lemmatise(tokens: List[str]) -> List[str]:
    """
    Lemmatise a list of tokens using spaCy.
    """
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]