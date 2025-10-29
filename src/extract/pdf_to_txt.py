import pdfplumber
import os
import sys

def convert_pdf_to_txt(pdf_path, output_path=None):
    if not output_path:
        output_path = pdf_path.replace(".pdf", ".txt")

    try:
        with pdfplumber.open(pdf_path) as pdf, open(output_path, 'w', encoding='utf-8') as out_file:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    out_file.write(text + "\n\n")
        print(f"Converted: {pdf_path} -> {output_path}")
    except Exception as e:
        print(f">>> Failed to process {pdf_path}: {e} <<<")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python convert_pdf.py file.pdf\n  or\n  python convert_pdf.py folder/")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                full_pdf_path = os.path.join(input_path, filename)
                convert_pdf_to_txt(full_pdf_path)
    elif input_path.lower().endswith('.pdf'):
        convert_pdf_to_txt(input_path)
    else:
        print("Please provide a valid PDF file or folder.")
