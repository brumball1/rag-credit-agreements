import json
import re
import nltk
import spacy
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords


nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")

#converts all text to lowercase and then extracts words
def tokeniser(text: str):
    return re.findall(r"\b\w+\b", text.lower())

#the idea is that the func will look in the processed data dir for files ending in .pages.jsonl, making a list of them.
#if no files are present/don't have the *.pages.jsonl extension throw FileNotFoundError
#loop through and open all files
#in each of those .jsonl go through each line of code and if any wasted space/characters still they should be stripped else skip
#returns a tuple of file name and the pages text
def iterate_pages(input_dir: Path):
    files = list(input_dir.glob("*.pages.jsonl"))
    if not files:
        raise FileNotFoundError(f" >>> No .pages.jsonl files found in {input_dir} <<<")
    for fp in files:  # loop though all files
        with fp.open("r", encoding="utf-8") as f:
            for line in f:  # loop through each line in each file
                line = line.strip()
                if not line:
                    continue
                yield fp.name, json.loads(line)

def load_stopwords():
    #uses the English NLTK built-in stopwords
    stops = stopwords.words("english")
    base_dir = Path(__file__).resolve().parents[2]
    #customisible legal stopwords
    legal_file = base_dir / "configs" / "stopwords" / "stopwords_legal.txt"
    if legal_file.exists():
        with open(legal_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stops.append(word)
        print(f"Successfully loaded {len(stops)} stopwords")
    else:
        print(f">>> No custom stopwords found in {legal_file} - only using NLTK stopwords <<<")
    return stops

#creates a Counter
#goes through the returned pairs yielded by iterate_pages(), taking text from each "page"
#tokenises the text into individual words
#removes stopwords
def word_frequency(input_dir: Path, stops):
    freq = Counter()
    for name, pages in iterate_pages(input_dir):
        tokens = tokeniser(pages.get("text", ""))
        filtered_tokens = []
        for t in tokens:
            if t not in stops:
                filtered_tokens.append(t)
        lemmas = lemmatise(filtered_tokens)
        freq.update(lemmas)
    return freq

def lemmatise(tokens):
    doc = nlp(" ".join(tokens))
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return lemmas

def save_word_bank(frequency, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "word_bank.lemma.csv"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("words,count\n")
        for word, count in frequency.most_common():
            f.write(f"{word},{count}\n")
    print(f"Saved word bank to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"
    derived_dir = base_dir / "data" / "derived"
    stops = load_stopwords()
    freq = word_frequency(processed_dir, stops)
    save_word_bank(freq, derived_dir)
