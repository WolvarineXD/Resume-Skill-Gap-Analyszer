# matching/weighted_scorer.py
#
# NOTE: This module is DEPRECATED.
# The pure cosine similarity approach here has been superseded by the LLM-based
# skill scorer in llm_scorer.py + score_calculator.py, which provides much
# richer signal (strength, source, depth, evidence) beyond raw embedding distance.
#
# Kept here for reference / ablation studies only.

from sentence_transformers import util


def compute_weighted_skill_score(
    resume_embedding,
    embedder,
    skill_weight_dict: dict
) -> float:
    """
    [DEPRECATED] Computes weighted skill score (0–100) using pure cosine similarity.

    skill_weight_dict example:
    {
        "recruitment": 9,
        "payroll": 6,
        "compliance": 4
    }
    """

    total_weight = sum(skill_weight_dict.values())

    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0

    for skill, weight in skill_weight_dict.items():
        skill_embedding    = embedder.get_embedding(skill)
        similarity         = util.cos_sim(skill_embedding, resume_embedding)
        similarity_score   = float(similarity.item())
        weighted_sum      += similarity_score * weight

    weighted_score = (weighted_sum / total_weight) * 100

    return round(weighted_score, 2)