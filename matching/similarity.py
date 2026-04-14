# matching/similarity.py

from sentence_transformers import util


def compute_similarity(embedding1, embedding2) -> float:
    """
    Compute cosine similarity between two embeddings.
    Returns a float in [0, 1] since embeddings are normalised.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    similarity = util.cos_sim(embedding1, embedding2)
    return float(similarity.item())


def compute_cosine_score_40(similarity_score: float) -> float:
    """
    Convert cosine similarity (0-1) to a score out of 40, with a soft floor.

    WHY THE FLOOR?
    Any two professional documents written in English share generic vocabulary
    (management, planning, processes, communication, development) which gives a
    raw cosine of ~0.25-0.35 even when the documents are completely irrelevant
    to each other.  Without a floor, these 'free baseline' points inflate the
    scores of clearly irrelevant candidates.

    FLOOR = 0.20:
      Raw cosine < 0.20 -> 0 pts
      Raw cosine = 0.20 -> 0 pts  (floor)
      Raw cosine = 0.33 -> (0.13 / 0.80) * 40 =  6.5 pts  (low relevance)
      Raw cosine = 0.47 -> (0.27 / 0.80) * 40 = 13.5 pts  (moderate relevance)
      Raw cosine = 0.65 -> (0.45 / 0.80) * 40 = 22.5 pts  (strong relevance)
      Raw cosine = 1.00 -> 40 pts               (perfect match)

    This creates good separation between truly relevant and irrelevant
    candidates without being overly aggressive on moderate-relevance resumes.
    """
    _FLOOR = 0.18                                   # minimum useful similarity
    similarity_score = max(0.0, min(1.0, similarity_score))
    if similarity_score <= _FLOOR:
        return 0.0
    effective = (similarity_score - _FLOOR) / (1.0 - _FLOOR)
    return round(effective * 40, 2)


def rank_resumes(resume_scores: list) -> list:
    """
    Rank resumes by 'final_score' descending.
    Tiebreak: alphabetical filename.
    """
    return sorted(
        resume_scores,
        key=lambda x: (x["final_score"], x["file"]),   # ✅ FIX: use 'final_score' not 'score'
        reverse=True
    )