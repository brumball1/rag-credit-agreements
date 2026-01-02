import json
from pathlib import Path

from stats.tokeniser import tokeniser
from extract.page_reader import iterate_pages

def chunk_paragraphs(input_dir: Path, output_path: Path) -> None:

    token_counter = 0
    paragraph_counter = 0
    paragraph_in_document = 0
    current_doc = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (output_path.open("w", encoding="utf-8") as out):
        for name, page in iterate_pages(input_dir):
            doc_id = name.replace(".pages_clean.pages.jsonl", "")
            page_number = page.get("page")
            text = page.get("text", "")

            if doc_id != current_doc:
                current_doc = doc_id
                token_counter = 0
                paragraph_in_document = 0

            paragraphs = []
            raw = text.split("\n\n")

            for p in raw:
                cleaned = p.strip()
                if cleaned:
                    paragraphs.append(cleaned)

            for paragraph in paragraphs:
                tokens = tokeniser(paragraph)
                num_tokens = len(tokens)

                token_start = token_counter
                token_end = token_counter + num_tokens

                record = {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "paragraph_in_doc": paragraph_in_document,
                    "global_paragraph": paragraph_counter,
                    "token_count": num_tokens,
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": paragraph
                }

                out.write(json.dumps(record) + "\n")

                # Increment counters
                paragraph_counter += 1
                paragraph_in_document += 1
                token_counter = token_end

    print(f"Chunked {paragraph_counter} paragraphs total.")

