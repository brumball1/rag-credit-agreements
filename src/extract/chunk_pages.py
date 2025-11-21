import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any

from stats.tokeniser import tokeniser
from extract.page_reader import iterate_pages


def chunk_pages(input_dir: Path, output_path: Path) -> None:
    """
    Build per page token metadata:
      - document id
      - page number
      - token count
      - token start and end within that document.
    """
    page_counter = 0
    token_counter = 0
    current_doc = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for name, page in iterate_pages(input_dir):
            doc_id = name.replace(".pages_clean.pages.jsonl", "")
            page_number = page.get("page")
            text = page.get("text", "")

            tokens = tokeniser(text)
            num_tokens = len(tokens)

            if doc_id != current_doc:
                token_counter = 0
                current_doc = doc_id

            token_start = token_counter
            token_end = token_counter + num_tokens

            record = {
                "Document Name": doc_id,
                "Page Number": page_number,
                "Token Count": num_tokens,
                "Token Start": token_start,
                "Token End": token_end,
            }

            out.write(json.dumps(record) + "\n")

            token_counter = token_end
            page_counter += 1

    print(f"Saved page chunks to {output_path} (pages processed: {page_counter})")