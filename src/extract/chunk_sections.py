import re
import json
from pathlib import Path

from src.stats.tokeniser import tokeniser
from src.extract.page_reader import iterate_pages

# E5 / similar encoder hard limit
MAX_TOKENS = 512

# Sections with fewer tokens than this are page-end fragments that should be
# merged with continuation text from subsequent pages rather than emitted alone.
MIN_TOKENS = 30

# Matches Article / Section headings at the start of a line.
#
# Regex deliberately captures generously — the helper _trim_heading()
# post-processes the captured string to remove body-text overflow.
#
# False-positive guard: Section/Article patterns reject lines where the
# first word after the identifier is lowercase (cross-references like
# "Section 7.7 has occurred").
_SECTION_SPLIT_RE = re.compile(
    r"(?m)"
    r"(^[ \t]*"
    r"(?:"
    # Article I / ARTICLE IV DEFINITIONS  (ALL-CAPS title + punctuation)
    r"(?:ARTICLE|Article)\.?\s+[IVXLCDM\d]+\.?(?!\s+[a-z])(?:\s+[A-Z][A-Z ;:,()]{0,60}\b)?"
    # Section 1.01(a). Defined Terms  (title runs to first '.' or '"')
    r"|(?:SECTION|Section)\.?\s+\d+(?:\.\d+)+(?:\([a-zA-Z0-9]+\))?\.?(?!\s+[a-z])"
    r'(?:\s+[^.\n"]+\b)?'
    # 1.01. Defined Terms  (bare numeric, same rule)
    r'|\d+(?:\.\d+)+(?:\([a-zA-Z0-9]+\))?\.?\s+[A-Z][^.\n"]*\b'
    r")"
    r")"
)

# Two consecutive lowercase-starting words signal the start of body text.
# Used by _trim_heading() to truncate headings that captured too much.
_BODY_TEXT_RE = re.compile(r'\s[a-z]\w*\s+[a-z]')

# Splits before any quoted defined term regardless of line position,
# e.g. "Adjusted Term SOFR Rate" means … or mid-paragraph "Alternate Base Rate Loan" means …
_DEFINITION_RE = re.compile(r'(?="[A-Z][^"]+"\s+(?:means|is defined|has the meaning))')

# Sentence boundary: split after . ; : when followed by a capital letter, quote, or paren.
_SENTENCE_RE = re.compile(r'(?<=[.;:])\s+(?=[A-Z"(])')


def _trim_heading(raw: str) -> tuple[str, str]:
    """
    Trim a raw heading capture to just the structural identifier + short title.

    Returns (heading, remainder) where *remainder* is the body-text overflow
    that should be prepended back to the chunk body.

    Truncation points (earliest wins):
      1. First quoted term (")
      2. Two consecutive lowercase-starting words (body text signal)

    Trailing punctuation (. , ; :) is stripped from the heading.
    """
    cut = len(raw)

    # 1. Cut at first quoted term
    quote_pos = raw.find('"')
    if 0 < quote_pos < cut:
        cut = quote_pos

    # 2. Cut at two consecutive lowercase-starting words
    m = _BODY_TEXT_RE.search(raw)
    if m and m.start() < cut:
        cut = m.start()

    heading = raw[:cut].strip().rstrip('.,;: ')
    remainder = raw[cut:]
    return heading, remainder


def _split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """
    Split *text* at section/article headings.

    Returns a list of (heading, body) pairs.  The first element may have
    heading=None when there is text before the first heading.
    """
    parts = _SECTION_SPLIT_RE.split(text)
    # parts layout (capturing group):
    #   [pre, heading0, body0, heading1, body1, ...]
    result: list[tuple[str | None, str]] = []

    pre = parts[0]
    if pre.strip():
        result.append((None, pre))

    i = 1
    while i < len(parts) - 1:
        raw_heading = parts[i]
        body = parts[i + 1]

        heading, remainder = _trim_heading(raw_heading)
        if remainder.strip():
            body = remainder + body

        result.append((heading, body))
        i += 2

    return result


def _midpoint_split(text: str) -> list[str]:
    """
    Tier 2: recursively bisect text near its midpoint, snapping the cut to the
    nearest sentence-ending word (one whose last character is . ; or :).
    Tier 3 word-window fires only when no punctuation exists anywhere.
    """
    if len(tokeniser(text)) <= MAX_TOKENS:
        return [text] if text.strip() else []

    words = text.split()
    mid = len(words) // 2

    # Scan outward from midpoint for a clean sentence boundary
    for offset in range(0, mid):
        for i in (mid - offset, mid + offset):
            if 0 < i < len(words) and words[i - 1][-1] in ".;:":
                left = " ".join(words[:i]).strip()
                right = " ".join(words[i:]).strip()
                result = []
                for half in (left, right):
                    if half:
                        result.extend(_midpoint_split(half))
                return result

    # Tier 3: no punctuation anywhere — greedy word-window cut.
    # Use tokeniser for counting because whitespace words and \b\w+\b tokens
    # diverge on hyphenated / apostrophe'd terms.
    chunks: list[str] = []
    buf: list[str] = []
    buf_count = 0
    for w in words:
        w_count = len(tokeniser(w))
        if buf_count + w_count > MAX_TOKENS and buf:
            chunks.append(" ".join(buf))
            buf, buf_count = [w], w_count
        else:
            buf.append(w)
            buf_count += w_count
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _split_long_unit(text: str) -> list[str]:
    """
    Guaranteed ≤ MAX_TOKENS splitter for a single oversized paragraph/sentence.

    Tier 1 — sentence-boundary greedy merge: split on [.;:] + capital, merge
              greedily. Any resulting chunk still over the limit goes to Tier 2.
    Tier 2 — midpoint snap: recursively bisect, snapping to the nearest
              sentence end.  Falls through to Tier 3 when no punctuation exists.
    Tier 3 — word-window hard cut (last resort, inside _midpoint_split).
    """
    text = text.strip()
    if not text or len(tokeniser(text)) <= MAX_TOKENS:
        return [text] if text else []

    # Tier 1: sentence-boundary greedy merge
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) > 1:
        chunks: list[str] = []
        buf: list[str] = []
        buf_count = 0
        for sent in sentences:
            sent_count = len(tokeniser(sent))
            if buf_count + sent_count > MAX_TOKENS and buf:
                chunks.append(" ".join(buf))
                buf, buf_count = [sent], sent_count
            else:
                buf.append(sent)
                buf_count += sent_count
        if buf:
            chunks.append(" ".join(buf))

        # Any chunk still over limit (single long sentence) → Tier 2
        result = []
        for chunk in chunks:
            if len(tokeniser(chunk)) > MAX_TOKENS:
                result.extend(_midpoint_split(chunk))
            else:
                result.append(chunk)
        return result

    # No sentence splits found at all — go straight to Tier 2
    return _midpoint_split(text)


def _hard_split(text: str) -> list[str]:
    """Greedily merge paragraphs into chunks that stay at or below MAX_TOKENS."""
    paragraphs = [
        p for p in text.split("\n\n")
        if p.strip() and not re.fullmatch(r'[.;,:\s]+', p.strip())
    ]
    chunks: list[str] = []
    buffer_parts: list[str] = []
    buffer_count = 0

    for para in paragraphs:
        para_count = len(tokeniser(para))
        if para_count > MAX_TOKENS:
            # Flush current buffer before force-splitting the oversized paragraph
            if buffer_parts:
                chunks.append("\n\n".join(buffer_parts))
                buffer_parts, buffer_count = [], 0
            chunks.extend(_split_long_unit(para))
            continue
        if buffer_count + para_count > MAX_TOKENS and buffer_parts:
            chunks.append("\n\n".join(buffer_parts))
            buffer_parts, buffer_count = [para], para_count
        else:
            buffer_parts.append(para)
            buffer_count += para_count

    if buffer_parts:
        chunks.append("\n\n".join(buffer_parts))

    return chunks or ([text] if text.strip() else [])


def _split_oversized(text: str) -> list[str]:
    """
    Split an oversized section body into sub-chunks <= MAX_TOKENS.

    Tries definition-boundary splits first (suitable for Section 1.01 style
    blocks where every defined term starts a new paragraph).  Falls back to
    greedy paragraph merging when no definition boundaries are found.
    """
    def_parts = [p for p in _DEFINITION_RE.split(text) if p.strip()]

    if len(def_parts) <= 1:
        return _hard_split(text)

    # Greedily merge definition entries into chunks <= MAX_TOKENS
    chunks: list[str] = []
    buffer_parts: list[str] = []
    buffer_count = 0

    for part in def_parts:
        part_count = len(tokeniser(part))
        if part_count > MAX_TOKENS:
            # A single definition entry is itself too long — hard-split it
            if buffer_parts:
                chunks.append("\n\n".join(buffer_parts))
                buffer_parts = []
                buffer_count = 0
            chunks.extend(_hard_split(part))
        elif buffer_count + part_count > MAX_TOKENS and buffer_parts:
            chunks.append("\n\n".join(buffer_parts))
            buffer_parts = [part]
            buffer_count = part_count
        else:
            buffer_parts.append(part)
            buffer_count += part_count

    if buffer_parts:
        chunks.append("\n\n".join(buffer_parts))

    return chunks


def chunk_sections(input_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_section_counter = 0
    section_in_doc = 0
    current_doc: str | None = None
    current_heading: str | None = None
    current_start_page: int | None = None
    current_text_parts: list[str] = []
    token_counter = 0

    def flush(out, force: bool = False) -> None:
        """Emit the current section.  If *force* is False and the body is
        under MIN_TOKENS, skip emission so the fragment carries forward."""
        nonlocal global_section_counter, token_counter, section_in_doc

        text = "\n\n".join(
            p for p in current_text_parts
            if p.strip() and not re.fullmatch(r'[.;,:\s]+', p.strip())
        )
        if not text.strip():
            current_text_parts.clear()
            return

        num_tokens = len(tokeniser(text))

        # Page-end fragment guard: keep accumulating unless forced (doc
        # boundary or final flush) so tiny remainders merge with the next page.
        if not force and num_tokens < MIN_TOKENS:
            return

        sub_chunks = _split_oversized(text) if num_tokens > MAX_TOKENS else [text]
        needs_sub_index = len(sub_chunks) > 1

        for i, chunk_text in enumerate(sub_chunks):
            chunk_count = len(tokeniser(chunk_text))
            token_start = token_counter
            token_end = token_counter + chunk_count

            record: dict = {
                "doc_id": current_doc,
                "section_heading": current_heading,
                "section_in_doc": section_in_doc,
                "global_section": global_section_counter,
                "start_page": current_start_page,
                "token_count": chunk_count,
                "token_start": token_start,
                "token_end": token_end,
                "text": chunk_text,
            }
            if needs_sub_index:
                record["sub_chunk"] = i

            out.write(json.dumps(record) + "\n")

            global_section_counter += 1
            token_counter = token_end

        # Increment once per logical section regardless of how many sub-chunks
        # it was split into — keeps section_in_doc useful for distance metrics
        section_in_doc += 1
        # Clear parts so they are not re-emitted
        current_text_parts.clear()

    with output_path.open("w", encoding="utf-8") as out:
        for name, page in iterate_pages(input_dir):
            doc_id = name.replace(".pages_clean.pages.jsonl", "")
            page_number = page.get("page")
            text = page.get("text", "")

            if doc_id != current_doc:
                flush(out, force=True)
                current_doc = doc_id
                current_heading = None
                current_start_page = page_number
                current_text_parts = []
                section_in_doc = 0
                token_counter = 0

            segments = _split_on_headings(text)

            for heading, body in segments:
                if heading is None:
                    # Continuation of whatever section is already open
                    current_text_parts.append(body)
                else:
                    # New heading found — flush the current section first.
                    # Not forced: if the current section is a tiny page-end
                    # fragment (< MIN_TOKENS), it stays in current_text_parts
                    # and merges with the next page's continuation.
                    flush(out)
                    if not current_text_parts:
                        # flush emitted successfully — start fresh section
                        current_heading = heading
                        current_start_page = page_number
                        current_text_parts = [body]
                    else:
                        # flush skipped (fragment too small) — keep
                        # accumulating under the existing heading, append
                        # the new heading text and body as content
                        current_text_parts.append(heading)
                        current_text_parts.append(body)

        # Flush the final section of the last document
        flush(out, force=True)

    print(f"Chunked {global_section_counter} sections total.")
