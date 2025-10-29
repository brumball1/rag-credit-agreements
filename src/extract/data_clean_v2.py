import re
import json
from pathlib import Path

def clean_credit_agreement(text: str) -> str:
    """
    strip boilerplate, headers, footers, and formatting artifacts from credit-agreement-like pdf text.
    keeps article/section headings.
    """

    text = text.replace("\u00a0", " ")  # NBSP -> space

    #urls + timestamps
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(
        r"(?im)^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4})"
        r"(?:,\s*|\s+)\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s+Document\s*$",
        "",
        text,
    )

    #exhibit/watermark stamps
    text = re.sub(r"(?im)^\s*EX-\d+\.\d+\s*$", "", text)  # bare EX-10.1 line
    text = re.sub(r"(?im)^\s*EX-\d+\.\d+(?:\s+\d+)?\s+\S+\.htm[l]?\s+EX-\d+\.\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*(EXECUTION\s+(?:COPY|VERSION)|CONFIDENTIAL|DRAFT|PRELIMINARY)\s*$", "", text)

    #file path crumbs / internal ids
    text = re.sub(r"(?im)\([A-Z]{2,}\)\s*\d+/\d+/.+?\.(?:docx?|pdf)", "", text)   # (NY) ... .doc
    text = re.sub(r"(?i)\b[ A-Z-]*DOCS[\\/][\w.\-/\\]+\b", "", text)              # US-DOCS\12345.6 etc.

    # page furniture
    text = re.sub(r"(?im)^\s*PAGE\s+\d+\s*$", "", text)
    #numbers: "- 3 -", "— 5 —", "12"
    text = re.sub(r"(?im)^\s*[—–-]?\s*\d{1,4}\s*[—–-]?\s*$", "", text)
    #roman numeral pages
    text = re.sub(r"(?im)^\s*[ivxlcdm]{1,7}\s*$", "", text)
    #page counters: "3/167", "3 / 167", "Page 3 of 167", "3/167 pages"
    text = re.sub(r"(?im)^\s*(?:page\s+)?\d+\s*/\s*\d+(?:\s*pages?)?\s*$", "", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$", "", text)

    #table of contents blocks
    text = re.sub(
        r"(?is)\bTABLE\s+OF\s+CONTENTS\b.*?(?=^\s*(?:Article\s+[IVXLC]+|ARTICLE\s+[IVXLC]+|Section\s+\d|\Z))",
        "",
        text,
    )
    # dotted leader tails like "... 87"
    text = re.sub(r"(?m)\s*\.{2,}\s*\d+\s*$", "", text)

    # decorative titles when isolated
    text = re.sub(
        r"(?im)^\s*(CREDIT AGREEMENT|REVOLVING CREDIT AGREEMENT|TERM LOAN CREDIT AGREEMENT|TABLE OF CONTENTS)\s*$",
        "",
        text,
    )

    # de-hyphenate across line breaks, then unwrap soft line breaks
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)  # trans-\n action -> transaction
    text = re.sub(
        r"(?m)(?<![.?!:])\n(?!\n|^\s*(?:Article|Section|Schedule|Exhibit|ARTICLE|SECTION|SCHEDULE|EXHIBIT)\b|^\s*\(?[a-z]\)|^\s*\d+\.)",
        " ",
        text,
    )

    # misc boilerplate lines
    text = re.sub(r"(?im)^\s*Document continues on next page.*$", "", text)
    text = re.sub(r"(?im)^\s*\[(?:Signature|Remainder of) [Pp]age.*\].*$", "", text)
    text = re.sub(r"\b[A-Z]{3,}-\d{7,}\b", "", text)  # Bates-like

    # tidying up spaces like double or triple blanks
    text = re.sub(r"[ \t]{2,}", " ", text)        # fixes blank spaces between words
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"^\s+$", "", text, flags=re.M)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text.strip()


def processed_pages_jsonl(inp_path: Path, out_path: Path):
    """Process *.pages.jsonl files — keeps page count 1:1."""
    with inp_path.open('r', encoding='utf-8') as f:
        pages = [json.loads(line) for line in f]

    cleaned = []
    for rec in pages:
        rec['text'] = clean_credit_agreement(rec.get('text', ''))
        cleaned.append(rec)

    with out_path.open('w', encoding='utf-8') as w:
        for rec in cleaned:
            w.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Cleaned {len(cleaned)} pages → {out_path.name}")


def main():
    base_dir = Path(__file__).resolve().parents[2]
    interim_dir = base_dir / "data" / "interim"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(interim_dir.glob("*.pages.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .pages.jsonl files in {interim_dir}")

    for inp in files:
        out = processed_dir / f"{inp.stem}_clean.pages.jsonl"
        processed_pages_jsonl(inp, out)


if __name__ == "__main__":
    main()