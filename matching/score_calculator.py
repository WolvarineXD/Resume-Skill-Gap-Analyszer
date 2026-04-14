# matching/score_calculator.py
"""
Hybrid per-skill scorer — maximally stable design.

Score per skill = 0.85 x deterministic_cosine + 0.15 x llm_adjustment

Why 85/15?
  The per-skill cosine (cosine between skill embedding and the best matching
  resume sentence) is 100% reproducible: same model weights, same text,
  same result every single time.

  The LLM categorical label (strong/moderate/weak) can drift between runs
  even at temperature=0 because Groq routes across multiple model instances.
  Keeping LLM at only 15% means:

    Max drift per skill from LLM tier flip = 0.15 x 3.1 = 0.47 pts
    With 9 skills all flipping simultaneously (extreme case) = 4.2 pts MAX
    In practice (1-2 skills flip) = 0.5 - 1.0 pts total

  This is within the user's accepted tolerance of "1-2 points OK".

90-pt breakdown:
  40 pts — JD <-> Resume cosine (already deterministic, unchanged)
  60 pts — Per-skill scores (this file), weighted average
"""

from __future__ import annotations


# -- LLM tier parameters — 4 tiers ------------------------------------------------
#
#   STRONG    (7.5) : Skill explicitly DEMONSTRATED in work experience or project.
#                     Must show actual activity, not just job title or listing.
#
#   CONFIRMED (6.2) : Skill DIRECTLY NAMED in a skills section, certification,
#                     or education entry.  No demonstration needed.
#
#   MODERATE  (5.0) : Skill NOT directly listed but adjacent/supporting evidence
#                     is present in the same domain.
#
#   WEAK      (2.2) : Wrong domain, vague mention, or no evidence at all.
#
_BASE  = {"strong": 7.5, "confirmed": 6.2, "moderate": 5.0, "weak": 2.2}

# Source boost — must match EXACTLY the values the LLM is instructed to return:
#   work | project | education | skills_list | general
_SRC   = {"work": 0.8, "project": 0.6, "education": 0.3, "skills_list": 0.2, "general": 0.0}

# Depth boost — must match EXACTLY the values the LLM is instructed to return:
#   direct | adjacent | passing | none
_DEPTH = {"direct": 0.7, "adjacent": 0.35, "passing": 0.1, "none": 0.0}

# Confidence anchoring: even at conf=0 we get 72% of base (smoother curve)
_CONF_MIN = 0.72

# Blend weights: B=cosine is the STABLE component, A=LLM is the ADJUSTMENT
_W_COSINE = 0.85   # deterministic anchor
_W_LLM    = 0.15   # LLM quality adjustment


def _llm_score_raw(strength: str, source: str, depth: str, confidence: float) -> float:
    """Compute the LLM signal component on a [0, 10] scale."""
    base = _BASE.get(strength, 2.5)
    base += _SRC.get(source,   0.0)
    base += _DEPTH.get(depth,  0.0)
    # Confidence interpolates between 72% and 100% of base
    score = base * (_CONF_MIN + (1.0 - _CONF_MIN) * confidence)
    return min(score, 10.0)


def compute_skill_score(details: list, weighted_skills: dict) -> float:
    """
    Compute weighted skill score on a 0-60 scale.

    Each dict in `details` must contain:
      skill, strength, source, depth, confidence
      per_skill_cosine (float [0,1]) -- added by main.py before this call

    The per_skill_cosine should be the MAX of:
      - cosine(skill_embedding, resume_embedding)  [broad relevance]
      - cosine(skill_embedding, best_sentence_embedding)  [specific mention]
    This is computed and injected by main.py after the LLM call.
    """
    total_weight = sum(weighted_skills.values())
    if total_weight == 0:
        return 0.0

    achieved  = 0.0
    skill_map = {d["skill"].lower(): d for d in details}

    for skill, weight in weighted_skills.items():
        d = skill_map.get(skill.lower())

        if d is None:
            strength, source, depth, confidence = "weak", "general", "none", 0.05
            per_skill_cosine = 0.0
            evidence         = ""
        else:
            strength         = d.get("strength",         "weak").lower()
            source           = d.get("source",           "general").lower()
            depth            = d.get("depth",            "none").lower()
            confidence       = float(d.get("confidence", 0.05))
            per_skill_cosine = float(d.get("per_skill_cosine", 0.0))
            evidence         = d.get("evidence", "")

        # Signal A: LLM adjustment (15% weight, narrow tier gaps)
        sig_a = _llm_score_raw(strength, source, depth, confidence)

        # Signal B: deterministic cosine anchor (85% weight)
        # Cosine multiplier by tier:
        #   STRONG / CONFIRMED  -> LLM verified skill is present.  Full cosine.
        #   MODERATE            -> Adjacent evidence. Small 10% reduction.
        #   WEAK + has evidence -> Vague match only.  20% penalty.
        #   WEAK + no evidence  -> Confirmed absent.  50% penalty.
        if strength in ("strong", "confirmed"):
            cosine_multiplier = 1.0
        elif strength == "moderate":
            cosine_multiplier = 0.90
        elif evidence.strip():           # weak but LLM found some relevant text
            cosine_multiplier = 0.80
        else:                            # weak + empty = skill absent
            cosine_multiplier = 0.55

        sig_b = per_skill_cosine * 10.0 * cosine_multiplier   # [0, 10]

        # Blend
        if per_skill_cosine > 0.0:
            score = _W_LLM * sig_a + _W_COSINE * sig_b
        else:
            score = sig_a

        score     = min(score, 10.0)
        achieved += weight * (score / 10.0)

    return round((achieved / total_weight) * 60, 2)