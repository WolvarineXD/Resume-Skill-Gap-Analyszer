# preprocessing/text_extractor.py

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    Returns raw extracted text.
    """
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return text.strip()