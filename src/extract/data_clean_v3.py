"""
This script applies a set of regex-based cleaning rules to credit agreement
documents represented as JSON lines files, preserving legal structure
while stripping boilerplate and page furniture.  One key change in this
version is that decorative title lines such as "CREDIT AGREEMENT" are
only removed after the first page.  The first page often contains the
real title (e.g. "CREDIT AGREEMENT BETWEEN X AND Y"), so decorative
title removal is skipped on that page.  Each page remains one-to-one
with the input so downstream analyses can maintain page-level metadata.
"""

import re
import json
from pathlib import Path
from ftfy import fix_text  # type: ignore


def clean_credit_agreements(text: str, page_num: int | None = None) -> str:
    """
    Strip boilerplate, headers, footers, and formatting artifacts from
    credit‑agreement‑like PDF text. Keeps article/section headings. A
    page number can be provided so that decorative title lines are not
    removed on the first page (page_num == 0).

    This version normalizes the input using ftfy.fix_text() to repair
    encoding errors and converts non‑breaking spaces to regular spaces.
    It also removes zero‑width Unicode characters (\u200B, \u200C, \u200D,
    \uFEFF) that often appear in PDF output.
    """
    # First repair text using ftfy. Replace NBSP with a regular space before fixing.
    text = fix_text(text.replace("\u00a0", " "))

    # 1) URLs + timestamps
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(
        r"(?im)^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4})"
        r"(?:,\s*|\s+)\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s+Document\s*$",
        "",
        text,
    )
    # remove entire lines that start with a date/time pattern (e.g. '10/8/25, 10:16 AM CreditAgree2024')
    text = re.sub(
        r"(?im)^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4})"
        r"(?:,\s*|\s+)\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b.*$",
        "",
        text,
    )

    # 2) exhibit/watermark stamps (line-scoped)
    text = re.sub(r"(?im)^\s*EX-\d+\.\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*EX-\d+\.\d+(?:\s+\d+)?\s+\S+\.htm[l]?\s+EX-\d+\.\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*(EXECUTION\s+(?:COPY|VERSION)|CONFIDENTIAL|DRAFT|PRELIMINARY)\s*$", "", text)

    # 3) file path crumbs / internal ids
    text = re.sub(r"(?im)\([A-Z]{2,}\)\s*\d+/\d+/.+?\.(?:docx?|pdf)", "", text)
    text = re.sub(r"(?i)\b[ A-Z-]*DOCS[\\/][\w.\-/\\]+\b", "", text)

    # 4) page furniture
    # remove explicit 'PAGE N' lines
    text = re.sub(r"(?im)^\s*PAGE\s+\d+\s*$", "", text)
    # remove hyphenated numeric page markers like '- 3 -' or '— 5 —', but avoid plain numeric headings
    text = re.sub(r"(?im)^\s*[—–-]\s*\d{1,4}\s*[—–-]\s*$", "", text)
    # remove Roman numeral page markers (i, ii, iii, etc.)
    text = re.sub(r"(?im)^\s*[ivxlcdm]{1,7}\s*$", "", text)
    # remove page counters like '3/167', '3 / 167 pages', 'Page 3 of 167'
    text = re.sub(r"(?im)^\s*(?:page\s+)?\d+\s*/\s*\d+(?:\s*pages?)?\s*$", "", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$", "", text)
    text = re.sub(r'_{3,}', '', text)

    # 5) table of contents blocks
    text = re.sub(
        r"(?is)\bTABLE\s+OF\s+CONTENTS\b.*?(?=^\s*(?:Article\s+[IVXLC]+|ARTICLE\s+[IVXLC]+|Section\s+\d|\Z))",
        "",
        text,
    )
    # dotted leader tails like "... 87"
    text = re.sub(r"(?m)\s*\.{2,}\s*\d+\s*$", "", text)

    # 6) decorative titles when isolated (only remove after first page)
    # On page_num == 0, skip removal to preserve real title lines.
    if page_num is None or page_num > 0:
        text = re.sub(
            r"(?im)^\s*(CREDIT AGREEMENT|REVOLVING CREDIT AGREEMENT|TERM LOAN CREDIT AGREEMENT|TABLE OF CONTENTS)\s*$",
            "",
            text,
        )

    # 7) de-hyphenate across line breaks, then unwrap soft line breaks
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(
        r"(?m)(?<![.?!:])\n(?!\n|^\s*(?:Article|Section|Schedule|Exhibit|ARTICLE|SECTION|SCHEDULE|EXHIBIT)\b|^\s*\(?[a-z]\)|^\s*\d+\.)",
        " ",
        text,
    )

    # 8) misc boilerplate lines
    text = re.sub(r"(?im)^\s*Document continues on next page.*$", "", text)
    text = re.sub(r"(?im)^\s*\[(?:Signature|Remainder of) [Pp]age.*\].*$", "", text)
    text = re.sub(r"\b[A-Z]{3,}-\d{7,}\b", "", text)  # Bates-like

    # 9) spacing tidy
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"^\s+$", "", text, flags=re.M)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # remove invisible zero‑width characters (ZWSP, ZWNJ, ZWJ, BOM)
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

    return text.strip()


def processed_pages_jsonl(inp_path: Path, out_path: Path):
    """
    Process *.pages.jsonl files — keeps page count 1:1.
    For each page, calls clean_credit_agreement with the page number so that
    decorative titles are only removed after the first page.
    """
    with inp_path.open("r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    cleaned = []
    for i, rec in enumerate(pages):
        # pass page index into cleaner
        rec["text"] = clean_credit_agreements(rec.get("text", ""), page_num=i)
        cleaned.append(rec)

    with out_path.open("w", encoding="utf-8") as w:
        for rec in cleaned:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

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