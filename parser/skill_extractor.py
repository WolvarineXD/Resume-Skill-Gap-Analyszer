# parser/skill_extractor.py


SKILL_KEYWORDS = [
    "recruitment",
    "hr",
    "payroll",
    "employee engagement",
    "training",
    "labor law",
    "talent acquisition",
    "performance management",
    "compliance",
    "communication",
    "onboarding",
    "employee lifecycle",
    "hr operations"
]


def extract_skills(text: str) -> list:
    """
    Extracts predefined HR-related skills from text.
    """

    text_lower = text.lower()
    detected = []

    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            detected.append(skill)

    return list(set(detected))