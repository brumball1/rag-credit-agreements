import pdfplumber
import json
from pathlib import Path
from tqdm import tqdm

def convert_pdf_to_jsonl(pdf_path: str, output_dir: str = "data/interim"):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}.pages.jsonl"

    try:
        with pdfplumber.open(pdf_path) as pdf, open(output_path, "w", encoding="utf-8") as out_file:
            for i, page in enumerate(tqdm(pdf.pages, desc=f"Processing {pdf_path.name}", unit="page"), start=1):
                text = page.extract_text(layout=True) or ""
                record = {"page": i, "text": text}
                out_file.write(json.dumps(record) + "\n")

        print(f" Successfully converted: {pdf_path.name} → {output_path.name} ({len(pdf.pages)} pages)")

    except Exception as e:
        print(f">>> Failed to process {pdf_path.name}: {e} <<<")

def batch_convert(input_path: str):
    input_path = Path(input_path)
    if input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
        for pdf_file in tqdm(pdf_files, desc="Batch converting PDFs", unit="file"):
            convert_pdf_to_jsonl(pdf_file)
    elif input_path.suffix.lower() == ".pdf":
        convert_pdf_to_jsonl(input_path)
    else:
        print(">>> Please provide a valid PDF file or folder containing PDFs. <<<")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:\n  python src/extract/pdf_to_jsonl.py file.pdf\n  or\n  python src/extract/pdf_to_jsonl.py folder/")
    else:
        batch_convert(sys.argv[1])



