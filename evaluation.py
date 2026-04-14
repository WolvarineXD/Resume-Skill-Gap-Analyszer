# evaluation.py
"""
Evaluation metrics for the Hybrid ATS Resume Screener.

Metrics are grouped into tiers:
  Tier 1 – Ranking Quality  : NDCG@k, Precision@k, MRR  (need ground truth)
  Tier 2 – Scoring Reliability : spread, score-weight corr, confidence calibration
  Tier 3 – LLM Output Quality : evidence coverage, strength distribution
  Tier 4 – System Performance  : latency (tracked in main.py)
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


# ═══════════════════════════════════════════════════════
# TIER 2 — SCORE STATS
# ═══════════════════════════════════════════════════════

def compute_score_stats(results: list) -> dict:
    """
    Basic score distribution across all ranked resumes.
    Expects each result to have a 'final_score' key.
    """
    scores = [r["final_score"] for r in results]

    return {
        "mean":   round(float(np.mean(scores)),  2),
        "std":    round(float(np.std(scores)),   2),
        "min":    round(float(np.min(scores)),   2),
        "max":    round(float(np.max(scores)),   2),
        "spread": round(float(np.max(scores) - np.min(scores)), 2)
    }


# ═══════════════════════════════════════════════════════
# TIER 1 — RANK STABILITY  (requires two runs)
# ═══════════════════════════════════════════════════════

def compute_rank_stability(prev_results: list, new_results: list) -> dict:
    """
    Spearman ρ and Kendall τ between two ranking runs.
    Useful for checking consistency when re-running with same inputs.

    ✅ FIX: Only compares files present in BOTH runs (intersection).
    """
    prev_scores = {r["file"]: r["final_score"] for r in prev_results}
    new_scores  = {r["file"]: r["final_score"] for r in new_results}

    # ✅ Intersection only — prevents KeyError
    common = list(set(prev_scores.keys()) & set(new_scores.keys()))

    if len(common) < 2:
        return {
            "spearman": None,
            "kendall":  None,
            "note":     "Need at least 2 common files to compute rank stability."
        }

    prev = [prev_scores[k] for k in common]
    new  = [new_scores[k]  for k in common]

    spearman_corr, _ = spearmanr(prev, new)
    kendall_corr,  _ = kendalltau(prev, new)

    return {
        "spearman": round(float(spearman_corr), 3),
        "kendall":  round(float(kendall_corr),  3)
    }


# ═══════════════════════════════════════════════════════
# TIER 3 — SKILL COVERAGE
# ═══════════════════════════════════════════════════════

def compute_skill_coverage(details_list: list) -> dict:
    """How many skills were evaluated per resume (on average)."""
    if not details_list:
        return {"avg_skills_detected": 0, "min_skills": 0, "max_skills": 0}

    counts = [len(d) for d in details_list]

    return {
        "avg_skills_detected": round(float(np.mean(counts)), 2),
        "min_skills":          int(np.min(counts)),
        "max_skills":          int(np.max(counts))
    }


# ═══════════════════════════════════════════════════════
# TIER 3 — LLM CONFIDENCE STATS
# ═══════════════════════════════════════════════════════

def compute_confidence_stats(details_list: list) -> dict:
    """
    Average and std of LLM confidence scores.
    Low std suggests the LLM is not discriminating well.
    """
    confs = [
        d.get("confidence", 0.5)
        for details in details_list
        for d in details
    ]

    if not confs:
        return {"avg_confidence": 0.0, "std_confidence": 0.0}

    return {
        "avg_confidence": round(float(np.mean(confs)), 3),
        "std_confidence": round(float(np.std(confs)),  3)
    }


# ═══════════════════════════════════════════════════════
# TIER 3 — STRENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════

def compute_strength_distribution(details_list: list) -> dict:
    """
    Percentage of skills rated strong / moderate / weak across all resumes.
    A healthy screener should NOT be 90% 'weak' — that suggests the context
    window is too narrow or the LLM is being too conservative.
    """
    counts = {"strong": 0, "moderate": 0, "weak": 0}
    total  = 0

    for details in details_list:
        for d in details:
            s = d.get("strength", "weak").lower()
            counts[s] = counts.get(s, 0) + 1
            total += 1

    if total == 0:
        return {k: "0.0%" for k in counts}

    return {
        k: f"{round(v / total * 100, 1)}%"
        for k, v in counts.items()
    }


# ═══════════════════════════════════════════════════════
# TIER 3 — EVIDENCE COVERAGE
# ═══════════════════════════════════════════════════════

def compute_evidence_coverage(details_list: list) -> dict:
    """
    Fraction of skill evaluations where the LLM returned actual evidence text.
    Low rate = LLM is producing empty/useless evidence → bad skill scoring.
    """
    total        = 0
    with_evidence = 0

    for details in details_list:
        for d in details:
            total += 1
            if d.get("evidence", "").strip():
                with_evidence += 1

    if total == 0:
        return {"evidence_coverage_rate": "N/A", "skills_with_evidence": 0, "total_skills": 0}

    return {
        "evidence_coverage_rate": f"{round(with_evidence / total * 100, 1)}%",
        "skills_with_evidence":   with_evidence,
        "total_skills":           total
    }


# ═══════════════════════════════════════════════════════
# TIER 2 — CONFIDENCE CALIBRATION CHECK
# ═══════════════════════════════════════════════════════

def compute_confidence_calibration(details_list: list) -> dict:
    """
    Check alignment between LLM strength label and confidence value.
    A well-calibrated LLM should have:
      strong   → high confidence  (≥ 0.7)
      moderate → mid  confidence  (0.4–0.7)
      weak     → low  confidence  (≤ 0.4)

    Returns the average confidence per strength bucket.
    """
    buckets = {"strong": [], "moderate": [], "weak": []}

    for details in details_list:
        for d in details:
            s = d.get("strength", "weak").lower()
            c = d.get("confidence", 0.0)
            if s in buckets:
                buckets[s].append(c)

    result = {}
    for strength, confs in buckets.items():
        result[f"avg_conf_{strength}"] = round(float(np.mean(confs)), 3) if confs else None

    return result


# ═══════════════════════════════════════════════════════
# TIER 1 — PRECISION@k  (requires ground truth)
# ═══════════════════════════════════════════════════════

def compute_precision_at_k(ranked_results: list, relevant_files: set, k: int = 3) -> float:
    """
    Precision@k: fraction of top-k results that are truly relevant.
    Requires a set of filenames considered 'good' candidates (ground truth).

    Example: relevant_files = {"resume_A.pdf", "resume_C.pdf"}
    """
    top_k = [r["file"] for r in ranked_results[:k]]
    hits  = sum(1 for f in top_k if f in relevant_files)
    return round(hits / k, 3)


# ═══════════════════════════════════════════════════════
# TIER 1 — MRR  (requires ground truth)
# ═══════════════════════════════════════════════════════

def compute_mrr(ranked_results: list, relevant_files: set) -> float:
    """
    Mean Reciprocal Rank: 1 / rank_of_first_relevant_result.
    Returns 0.0 if no relevant result appears in the ranking.
    """
    for i, r in enumerate(ranked_results, 1):
        if r["file"] in relevant_files:
            return round(1.0 / i, 3)
    return 0.0


# ═══════════════════════════════════════════════════════
# TIER 1 — NDCG@k  (requires relevance scores)
# ═══════════════════════════════════════════════════════

def compute_ndcg_at_k(ranked_results: list, relevance_scores: dict, k: int = 5) -> float | None:
    """
    NDCG@k using sklearn.
    relevance_scores: dict {filename: relevance_score}
                      (e.g., manual ratings: 0/1/2/3)

    Returns None if not enough data or sklearn unavailable.
    """
    try:
        from sklearn.metrics import ndcg_score

        k = min(k, len(ranked_results))
        if k < 2:
            return None

        true_relevance = [[relevance_scores.get(r["file"], 0) for r in ranked_results[:k]]]
        predicted      = [[r["final_score"]                    for r in ranked_results[:k]]]

        return round(float(ndcg_score(true_relevance, predicted)), 3)

    except Exception:
        return None


# ═══════════════════════════════════════════════════════
# FORMAT EVALUATION REPORT  (returns string — used by main.py)
# ═══════════════════════════════════════════════════════

def format_evaluation(results: list, all_details: list) -> str:
    """
    Build a comprehensive evaluation report as a single string.
    Returns the string so main.py can include it in one atomic write.
    ASCII-safe (no emoji) to avoid Windows cp1252 encoding errors.
    """
    sep = "=" * 52
    L   = []    # accumulate lines

    def add(text=""):
        L.append(str(text))

    add()
    add(sep)
    add("  EVALUATION REPORT")
    add(sep)

    # ── Score stats ──
    stats = compute_score_stats(results)
    add()
    add("  Score Distribution")
    add(f"    Mean Score  : {stats['mean']:6.2f} / 100")
    add(f"    Std Dev     : {stats['std']:6.2f}")
    add(f"    Min Score   : {stats['min']:6.2f} / 100")
    add(f"    Max Score   : {stats['max']:6.2f} / 100")
    disc = "Low spread — hard to differentiate" if stats["spread"] < 10 else "Good discrimination"
    add(f"    Spread      : {stats['spread']:6.2f}  [{disc}]")

    # ── Skill coverage ──
    coverage = compute_skill_coverage(all_details)
    add()
    add("  Skill Coverage")
    add(f"    Avg Evaluated : {coverage['avg_skills_detected']}")
    add(f"    Min Detected  : {coverage['min_skills']}")
    add(f"    Max Detected  : {coverage['max_skills']}")

    # ── LLM confidence ──
    conf = compute_confidence_stats(all_details)
    add()
    add("  LLM Confidence")
    add(f"    Average : {conf['avg_confidence']}")
    uniform_note = "  [Very uniform — low discrimination]" if conf["std_confidence"] < 0.05 else ""
    add(f"    Std Dev : {conf['std_confidence']}{uniform_note}")

    # ── Calibration ──
    cal = compute_confidence_calibration(all_details)
    add()
    add("  Confidence Calibration (avg conf per strength level)")
    for k, v in cal.items():
        label = k.replace("avg_conf_", "").capitalize()
        add(f"    {label:10s}: {v if v is not None else 'N/A'}")

    # ── Strength distribution ──
    dist = compute_strength_distribution(all_details)
    add()
    add("  Strength Distribution (all skills x all resumes)")
    for strength, pct in dist.items():
        bar_len = int(float(pct.rstrip("%")) / 5)
        add(f"    {strength.capitalize():10s}: {pct}  {'|' * bar_len}")

    # ── Evidence coverage ──
    ev = compute_evidence_coverage(all_details)
    add()
    add("  Evidence Coverage")
    add(f"    Rate                 : {ev['evidence_coverage_rate']}")
    add(f"    Skills with evidence : {ev.get('skills_with_evidence', 'N/A')} / {ev.get('total_skills', 'N/A')}")

    add()
    add(sep)
    add("  NOTE: Precision@k, MRR, NDCG@k require ground-truth labels.")
    add("        Import compute_precision_at_k / compute_mrr / compute_ndcg_at_k")
    add("        and call with a set of known-good resume filenames.")
    add(sep)

    return "\n".join(L)


# ── Keep the old print version for backward compatibility ──────────────────
def print_full_evaluation(results: list, all_details: list) -> None:
    """Backward-compatible wrapper — delegates to format_evaluation."""
    print(format_evaluation(results, all_details))