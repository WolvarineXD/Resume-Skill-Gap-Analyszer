# parser/entity_extractor.py

import re
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def extract_structured_info(text: str) -> dict:
    """
    Extracts structured information from resume text.
    Returns dictionary with key entities.
    """

    info = {}

    # -----------------------------
    # Email
    # -----------------------------
    email_match = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text
    )
    info["email"] = email_match[0] if email_match else None

    # -----------------------------
    # Phone Number (10-digit)
    # -----------------------------
    phone_match = re.findall(r"\b\d{10}\b", text)
    info["phone"] = phone_match[0] if phone_match else None

    # -----------------------------
    # LinkedIn
    # -----------------------------
    linkedin_match = re.findall(
        r"linkedin\.com/[^\s]+", text.lower()
    )
    info["linkedin"] = linkedin_match[0] if linkedin_match else None

    # -----------------------------
    # Name (Heuristic: First Line)
    # -----------------------------
    first_line = text.split("\n")[0]
    possible_name = first_line.split()[:2]
    info["name"] = " ".join(possible_name)

    # -----------------------------
    # NER (Organization, Date, Location)
    # -----------------------------
    doc = nlp(text)

    organizations = []
    dates = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "GPE":
            locations.append(ent.text)

    info["organizations"] = list(set(organizations))
    info["dates"] = list(set(dates))
    info["locations"] = list(set(locations))

    return info