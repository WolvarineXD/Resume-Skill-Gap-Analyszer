# preprocessing/text_cleaner.py

import re


def clean_text(text: str) -> str:
    """
    Cleans extracted text while preserving semantic meaning.
    - Normalises whitespace (tabs, multiple spaces → single space)
    - Removes non-ASCII / unusual Unicode characters
    - Does NOT remove stopwords, numbers, or punctuation (they carry meaning
      for embeddings and LLM context)
    """
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)       # Collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)    # At most two consecutive newlines

    # Remove unusual unicode / non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()