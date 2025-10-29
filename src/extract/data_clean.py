from pathlib import Path
import json, re
from ftfy import fix_text
from unidecode import unidecode

#paths
base_dir = Path(__file__).resolve().parents[2] #gets back to root
interim_dir = base_dir /"data"/"interim"
processed_dir = base_dir /"data"/"processed"
processed_dir.mkdir(parents = True, exist_ok = True)

# Make sure to change this to the document you want to clean
document_basename = "SPV-Credit-Agt"  # just include the file name w/o the extension .pages.jsonl
INP = interim_dir / f"{document_basename}.pages.jsonl"
OUT = processed_dir / f"{document_basename}_clean.pages.jsonl"

# regex you can tune per corpus
re_footer = re.compile(r"\(NY\)\s*\d+/\d+/.+?\.doc", re.I)    # a file footer like "(NY) 07865/007/LLC/SPV.Credit.Agreement.doc"
re_execution_copy = re.compile(r"\bEXECUTION\s+COPY\b", re.I)    # front page EXECUTION COPY
re_page_line = re.compile(r"^\s*PAGE\s+\d+\s*$", re.I | re.M)


def clean_page_text(t: str) -> str:
    t = fix_text(unidecode(t))

    # removes boilerplate stuff
    t = re_footer.sub("", t)
    t = re_execution_copy.sub("", t)
    t = re_page_line.sub("", t)

    # removes hyphens at line breaks like obliga- and tions
    t = re.sub(r"-\s*\n\s*", "", t)

    # combines soft line breaks into one line
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    #removes table of contents
    t = re.sub(r"TABLE OF CONTENTS.*?(?=ARTICLE\s+\d+)", "", t, flags=re.I | re.S)

    # removes whitespace
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    t = re.sub(r"\f", "", t)

    #final trim up
    return t.strip()

def main():
    if not INP.exists():
        raise FileNotFoundError(f" >>> Input not found: {INP} <<< ")

   #ensures we did not lose any pages on cleaning
    pages_in = 0
    pages_out = 0

    with INP.open("r", encoding="utf-8") as file_in, OUT.open("w", encoding="utf-8") as file_out:
        for line in file_in:
            record = json.loads(line)
            pages_in += 1
            record["text"] = clean_page_text(record["text"])
            file_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pages_out += 1

    print(f"Cleaned {pages_out} pages → {OUT}")
    assert pages_in == pages_out, " >>> Page count mismatch—should be identical! <<<"

if __name__ == "__main__":
    main()
