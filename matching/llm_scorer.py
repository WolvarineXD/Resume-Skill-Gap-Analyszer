# matching/llm_scorer.py

from groq import Groq
import os
import json
import re
import time
import logging
import hashlib

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================================================================
# DISK-BASED LLM RESPONSE CACHE
# =========================================================================
# Cache key = SHA-256( sorted_skill_names + resume_text_hash )
#
# WHY NOT include the JD?
#   The LLM evaluates "does this resume demonstrate skill X?" — purely
#   skill vs resume content.  The JD embedding never enters the LLM
#   prompt.  JD variation (pasting with extra whitespace / line endings)
#   was the #1 cause of cache misses.  Removing JD means:
#       Same PDF + same skills -> ALWAYS the same cache HIT.
#
# WHY NOT use context sentences as the key?
#   Sentences are retrieved via floating-point cosine similarity, which
#   has tiny non-determinism between runs.  Sentence-based keys therefore
#   changed every run -> cache never hit.  Text-hash keys are perfectly
#   stable across all runs.
# =========================================================================

_CACHE_FILE    = ".llm_cache.json"
_CACHE_VERSION = "v8"   # v8: 4-tier system (confirmed between strong and moderate)

_call_cache: dict = {}


def _load_cache() -> None:
    global _call_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("__version__") == _CACHE_VERSION:
                _call_cache = data.get("entries", {})
        except Exception:
            _call_cache = {}


def _save_cache() -> None:
    try:
        payload = {"__version__": _CACHE_VERSION, "entries": _call_cache}
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _make_cache_key(skills: list, resume_text: str) -> str:
    """
    Stable cache key: SHA-256( sorted_skills + resume_text_hash ).
    JD deliberately excluded — see module docstring above.
    """
    payload = {
        "v": _CACHE_VERSION,
        "skills": sorted(s.lower().strip() for s in skills),
        "resume": hashlib.sha256(
            resume_text[:20_000].encode("utf-8", errors="replace")
        ).hexdigest()[:24],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


_load_cache()   # load once at import time


# =========================================================================
# EVIDENCE VALIDATOR  (anti-hallucination, graduated penalty)
# =========================================================================

def _validate_evidence(evidence: str, context_sentences: list) -> tuple:
    """
    Returns (is_valid: bool, overlap_ratio: float).
    Overlap tiers (handled in caller):
      >= 0.40  OK
      0.20-0.40 soft: confidence *= 0.70
      < 0.20   hard: clear evidence, cap confidence at 0.20
    """
    ev = evidence.strip()
    if not ev:
        return True, 1.0

    if not context_sentences:
        return False, 0.0

    # Strip section-tag prefixes before overlap check
    _TAGS = ["[WORK EXP] ", "[PROJECT]  ", "[SKILLS]   ",
             "[EDUCATION]", "[OBJECTIVE]", "[CERT]     ", "[GENERAL]  "]
    clean_ctx = []
    for s in context_sentences:
        cs = s
        for tag in _TAGS:
            cs = cs.replace(tag, "")
        clean_ctx.append(cs.strip())

    ev_words = set(re.findall(r"\w+", ev.lower()))
    if not ev_words:
        return True, 1.0

    best_overlap = 0.0
    for ctx in clean_ctx:
        ctx_words = set(re.findall(r"\w+", ctx.lower()))
        if not ctx_words:
            continue
        overlap = len(ev_words & ctx_words) / len(ev_words)
        best_overlap = max(best_overlap, overlap)

    return best_overlap >= 0.40, best_overlap


# =========================================================================
# LLM CALL
# =========================================================================

def call_llm(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    """Call Groq API with retry on rate-limit / transient errors."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=3000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(delay * (attempt + 1))
            else:
                logger.error(f"LLM call failed after {retries} attempts: {e}")
                raise


# =========================================================================
# JSON PARSER
# =========================================================================

def parse_json(raw: str) -> list:
    """
    Extract a JSON array from the LLM response.
    Handles: markdown fences, trailing commas, slight formatting errors.
    """
    raw = raw.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)

    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Sometimes the LLM wraps the array: {"results": [...]}
            for v in result.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Extract the first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        # Remove trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parse failed — returning empty list.")
    return []


# =========================================================================
# MAIN EVALUATION FUNCTION
# =========================================================================

def evaluate_all_skills_with_llm(
    skills:            list,
    skill_context_map: dict,
    resume_text:       str = "",
    jd_text:           str = "",
) -> list:
    """
    Evaluate all skills in a single LLM call.

    Each entry in skill_context_map[skill] is a section-tagged string
    like "[WORK EXP] Managed interviews and recruitement pipelines."
    The prompt teaches the LLM how to use these tags to calibrate strength.

    Post-processing:
      - Hallucination guard: if LLM evidence does not overlap with context,
        evidence is cleared and confidence is capped with a graduated penalty.
      - Keyword floor: if skill keyword appears in context AND LLM still says
        weak, upgrade to moderate.  Guarantees: explicit listing = moderate min.
    """

    # Build prompt
    prompt = (
        "You are a precise resume screening engine. Evaluate each skill based ONLY "
        "on the evidence sentences provided. Each sentence is prefixed with its "
        "section tag showing WHERE in the resume it came from.\n\n"

        "STRENGTH TIERS (4 levels — use EXACTLY one):\n"
        "  'strong'    = skill explicitly DEMONSTRATED with context in [WORK EXP] or [PROJECT].\n"
        "               e.g. 'Managed recruitment drives', 'Conducted 50 interviews'\n"
        "  'confirmed' = skill DIRECTLY NAMED/LISTED by exact word in [SKILLS], [CERT], or [EDUCATION].\n"
        "               No demonstration needed. The name itself appearing = confirmed.\n"
        "               e.g. [SKILLS] section has 'Recruitment' or 'Communication' listed\n"
        "  'moderate'  = skill is NOT directly listed but ADJACENT evidence shows related ability.\n"
        "               e.g. 'Supported hiring managers' implies interviewing adjacently.\n"
        "  'weak'      = wrong domain, vague, or no evidence at all.\n\n"

        "SECTION TAG RULES:\n"
        "  [WORK EXP] or [PROJECT] -> eligible for 'strong' if skill is clearly demonstrated.\n"
        "  [SKILLS]                -> 'confirmed' if skill name appears. Not 'moderate'.\n"
        "  [EDUCATION] / [CERT]    -> 'confirmed' for relevant coursework or cert.\n"
        "  [OBJECTIVE]             -> max 'moderate'. ASPIRATIONAL, not proven.\n"
        "  [GENERAL]               -> 'moderate' at best. Use judgment.\n\n"

        "STRICT RULES:\n"
        "1. evidence: VERBATIM QUOTE from one provided sentence (no paraphrasing).\n"
        "2. If no evidence: evidence='', strength='weak', confidence=0.1\n"
        "3. [OBJECTIVE] = aspirational only. Max strength 'moderate'.\n"
        "4. [SKILLS] section listing = 'confirmed'. NOT 'moderate'. NOT 'strong'.\n"
        "5. NEVER map a tech skill (Java, Python, Docker...) to HR or unrelated content.\n\n"

        "FIELD CONSTRAINTS (exactly these values or it is an error):\n"
        "  strength   : EXACTLY one of: strong | confirmed | moderate | weak\n"
        "  source     : EXACTLY one of: work | project | education | skills_list | general\n"
        "  depth      : EXACTLY one of: direct | adjacent | passing | none\n"
        "  confidence : float in [0.0, 1.0]\n\n"

        "OUTPUT: JSON array only. No markdown. "
        "Keys: skill, evidence, strength, source, depth, confidence, reason\n\n"
        "CANDIDATE EVIDENCE:\n"
    )
    for skill, ctx in skill_context_map.items():
        block = "\n".join(ctx) if ctx else "(no evidence sentences found)"
        prompt += f"\n### Skill: {skill}\n{block}\n"

    # Cache lookup
    cache_key = _make_cache_key(skills, resume_text)
    cache_hit = cache_key in _call_cache

    if cache_hit:
        raw_response = _call_cache[cache_key]
    else:
        raw_response = call_llm(prompt)
        _call_cache[cache_key] = raw_response
        _save_cache()

    parsed    = parse_json(raw_response)
    parsed_map = {
        p.get("skill", "").strip().lower(): p
        for p in parsed
        if isinstance(p, dict)
    }

    final = []
    for skill in skills:
        p = parsed_map.get(skill.lower(), {})

        # Defensive str() coercion on every field: LLM sometimes returns
        # integers or null instead of strings (e.g. "depth": 0 -> crash).
        raw_strength = str(p.get("strength", "weak") or "weak").lower().strip()
        strength     = raw_strength if raw_strength in ("strong", "confirmed", "moderate", "weak") else "weak"

        try:
            conf = float(p.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.0

        conf = max(conf, {"strong": 0.70, "confirmed": 0.55, "moderate": 0.40, "weak": 0.05}.get(strength, 0.05))

        evidence              = str(p.get("evidence", "") or "").strip()
        context               = skill_context_map.get(skill, [])
        hallucination_cleared = False
        is_valid, overlap     = _validate_evidence(evidence, context)

        if not is_valid:
            if overlap < 0.20:
                evidence              = ""
                conf                  = min(conf, 0.20)
                hallucination_cleared = True
                if strength == "strong":
                    strength = "moderate"
            else:
                conf = round(conf * 0.70, 3)

        # ── Keyword-based score floor ────────────────────────────────────
        raw_reason = ""
        source_kw  = None   # set only if keyword floor fires
        depth_kw   = None
        if strength == "weak" and conf < 0.40 and context:
            ctx_text   = " ".join(context).lower()
            skill_words = [w.lower() for w in skill.split() if len(w) > 3]
            if skill_words and any(w in ctx_text for w in skill_words):
                strength              = "confirmed"   # directly listed = confirmed tier
                depth_kw              = "passing"
                source_kw             = "skills_list"
                conf                  = max(conf, 0.45)
                hallucination_cleared = False   # valid upgrade, NOT hallucination
                if not evidence.strip():
                    # Pull the first context line that has the keyword as evidence
                    _TAGS = ["[WORK EXP] ", "[PROJECT]  ", "[SKILLS]   ",
                             "[EDUCATION]", "[OBJECTIVE]", "[CERT]     ", "[GENERAL]  "]
                    for ctx_line in context:
                        if any(w in ctx_line.lower() for w in skill_words):
                            ev = ctx_line
                            for tag in _TAGS:
                                ev = ev.replace(tag, "")
                            evidence = ev.strip()
                            break
                raw_reason = "Skill keyword found directly in resume. Upgraded to moderate."

        # Strength-aware fallback for reason
        if not raw_reason:
            raw_reason = str(p.get("reason", "") or "").strip()
        if not raw_reason:
            raw_reason = {
                "strong":    "Explicit demonstration found in resume.",
                "confirmed": "Skill directly listed/named in resume.",
                "moderate":  "Adjacent or supporting evidence found.",
                "weak":      "No clear evidence found.",
            }.get(strength, "No clear evidence found.")

        # Use keyword-floor source/depth if they were set, otherwise LLM values
        final_source = source_kw if source_kw else str(p.get("source", "general") or "general").lower().strip()
        final_depth  = depth_kw  if depth_kw  else str(p.get("depth",  "none")    or "none").lower().strip()

        final.append({
            "skill":                skill,
            "evidence":             str(evidence or ""),
            "strength":             strength,
            "source":               final_source,
            "depth":                final_depth,
            "confidence":           round(conf, 3),
            "reason":               raw_reason,
            "hallucination_cleared": hallucination_cleared,
            "cache_hit":            cache_hit,
        })

    return final
