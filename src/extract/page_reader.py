import json
from pathlib import Path

def iterate_pages(input_dir: Path):
    files = list(input_dir.glob("*.pages.jsonl"))
    if not files:
        raise FileNotFoundError(f">>> No .pages.jsonl files found in {input_dir} <<<")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield fp.name, json.loads(line)