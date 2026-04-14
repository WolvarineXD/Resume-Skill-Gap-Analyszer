# main.py
# ╔══════════════════════════════════════════════════════════════════╗
# ║  ALL NOISE SUPPRESSIONS MUST HAPPEN HERE — BEFORE ANY IMPORT   ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
import sys
import logging

# 1) Silence ALL noisy third-party loggers
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"

for _noisy in [
    "httpx", "httpcore", "h11", "hpack",
    "groq", "groq._base_client",
    "sentence_transformers", "sentence_transformers.SentenceTransformer",
    "transformers", "transformers.tokenization_utils_base",
    "huggingface_hub", "huggingface_hub.utils._http",
    "huggingface_hub.file_download",
    "filelock", "urllib3", "requests",
]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# 2) Disable ALL tqdm progress bars
try:
    from tqdm import tqdm as _tqdm_cls
    from tqdm.auto import tqdm as _tqdm_auto_cls
    from functools import partialmethod as _pm
    _tqdm_cls.__init__      = _pm(_tqdm_cls.__init__,      disable=True)
    _tqdm_auto_cls.__init__ = _pm(_tqdm_auto_cls.__init__, disable=True)
except Exception:
    pass

# ── Now safe to import everything else ─────────────────────────────────────
import time

from sentence_transformers import util

from preprocessing.text_extractor import extract_text_from_pdf
from preprocessing.text_cleaner   import clean_text
from parser.entity_extractor      import extract_structured_info

from matching.embedder         import Embedder
from matching.similarity       import compute_similarity, compute_cosine_score_40
from matching.llm_scorer       import evaluate_all_skills_with_llm
from matching.score_calculator import compute_skill_score

from evaluation import format_evaluation

# ─────────────────────────────────────────────
# LOGGING  (our own logger only)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LAZY spaCy LOADER
# ─────────────────────────────────────────────
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            spacy.util.logger.setLevel(logging.CRITICAL)
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy unavailable ({e}). Using fallback sentence split.")
    return _nlp


# ─────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────
def split_sentences(text: str) -> list:
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text[:50_000])
        return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


# ─────────────────────────────────────────────
# SENTENCE FILTER
# ─────────────────────────────────────────────
_NOISE = ["@gmail", "@yahoo", "@hotmail", "linkedin", "phone:",
          "www.", "http", "+91", "github.com"]

def clean_sentences(sentences: list) -> list:
    return [
        s for s in sentences
        if not any(tok in s.lower() for tok in _NOISE)
        and len(s.split()) >= 5
    ]


# ─────────────────────────────────────────────
# SECTION HEURISTICS (for context tagging)
# ─────────────────────────────────────────────
_SECTION_HEADERS = {
    "work":    ["experience", "employment", "work history", "internship", "intern", "career"],
    "project": ["project", "projects", "portfolio", "case study"],
    "skills":  ["skill", "skills", "competenc", "proficien", "abilities", "tools", "technology"],
    "edu":     ["education", "qualification", "academic", "degree", "university", "college"],
    "obj":     ["objective", "summary", "profile", "about me", "career goal", "seeking"],
    "cert":    ["certif", "award", "achievement", "honor"],
}

def _classify_sections(text: str) -> list:
    """
    Walk through lines and tag each line with a section label.
    Returns list of (line, section_tag) tuples.
    """
    lines    = text.split("\n")
    current  = "general"
    tagged   = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Short ALL-CAPS or title-case lines with no punctuation → likely a section header
        if len(stripped.split()) <= 4 and (stripped.isupper() or stripped.istitle()):
            low = stripped.lower()
            for sec, keywords in _SECTION_HEADERS.items():
                if any(k in low for k in keywords):
                    current = sec
                    break
        tagged.append((stripped, current))
    return tagged


def get_relevant_sentences(skill, sentences, embedder, sent_embeddings, top_k=5):
    """
    Returns (top_sentences: list[str], max_raw_cosine: float).
    top_k increased to 5 to give the LLM more evidence to work with.
    """
    if not sentences:
        return [], 0.0

    skill_emb = embedder.get_embedding(skill)
    scored    = []        # (boosted_score, raw_sim, sentence)
    for i, s in enumerate(sentences):
        s_low   = s.lower()
        raw_sim = util.cos_sim(skill_emb, sent_embeddings[i]).item()
        boost   = 0.0
        if any(x in s_low for x in ["experience", "worked", "handled", "intern", "led", "managed"]):
            boost += 0.08
        if any(x in s_low for x in ["project", "developed", "built", "designed", "implemented"]):
            boost += 0.06
        boosted = raw_sim + boost
        if boosted > 0.18 or skill.split()[0].lower() in s_low:   # lowered from 0.22
            scored.append((boosted, raw_sim, s))

    scored.sort(reverse=True, key=lambda x: x[0])

    top_sents   = [s for _, _, s in scored[:top_k]]
    max_raw_sim = scored[0][1] if scored else 0.0
    return top_sents, max_raw_sim


def get_skill_context_tagged(skill, raw_text, sentences, embedder, sent_embeddings, top_k=5):
    """
    Build LLM context for one skill with section-tagged sentences.

    HOW IT WORKS:
    1. Keyword scan: find ANY sentence directly containing the skill word(s).
       These are tagged [SKILLS]/(work)/[MENTION] and prepended — they ALWAYS
       go to the LLM regardless of cosine similarity.
    2. Semantic retrieval: find top-k semantically relevant sentences.
       These are tagged with their detected resume section.
    3. Combine: keyword hits first, then semantic hits (deduplicated).

    WHY THIS MATTERS:
    Without keyword injection, a skill listed in a skills section
    (e.g. 'Recruitment and Onboarding') has a low cosine similarity to
    'onboarding' because the sentence is dense with multiple topics.
    The `_MIN_CONTEXT_SIM` filter then blocks it entirely.
    With keyword injection, the LLM ALWAYS sees it.
    """
    skill_words = [w.lower() for w in skill.split() if len(w) > 3]
    if not skill_words:
        skill_words = [skill.lower()]

    # Step 1: Keyword scan — tag and inject direct mentions
    tagged_lines  = _classify_sections(raw_text)
    section_of    = {}   # sentence_text -> section_tag
    for line_text, sec in tagged_lines:
        section_of[line_text.strip()] = sec

    _SEC_LABEL = {
        "work":    "[WORK EXP]",
        "project": "[PROJECT] ",
        "skills":  "[SKILLS]  ",
        "edu":     "[EDUCATION]",
        "obj":     "[OBJECTIVE]",
        "cert":    "[CERT]    ",
        "general": "[GENERAL] ",
    }

    keyword_hits = []   # (sentence, tag_label)
    seen = set()
    for s in sentences:
        s_low = s.lower()
        if any(w in s_low for w in skill_words):
            sec   = section_of.get(s.strip(), "general")
            label = _SEC_LABEL.get(sec, "[GENERAL] ")
            keyword_hits.append((label + " " + s, s))
            seen.add(s)

    # Step 2: Semantic retrieval
    top_sents, max_sim = get_relevant_sentences(skill, sentences, embedder, sent_embeddings, top_k)
    semantic_tagged = []
    for s in top_sents:
        if s not in seen:
            sec   = section_of.get(s.strip(), "general")
            label = _SEC_LABEL.get(sec, "[GENERAL] ")
            semantic_tagged.append(label + " " + s)
            seen.add(s)

    # Combine: keyword hits (max 3) + semantic hits
    combined  = [line for line, _ in keyword_hits[:3]] + semantic_tagged[:top_k]
    return combined, max_sim, bool(keyword_hits)


# ─────────────────────────────────────────────
# API KEY VALIDATION
# ─────────────────────────────────────────────
def validate_env():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.error(
            "GROQ_API_KEY is not set.\n"
            "  PowerShell : $env:GROQ_API_KEY = 'your_key'\n"
            "  CMD        : set GROQ_API_KEY=your_key"
        )
        sys.exit(1)


# ─────────────────────────────────────────────
# ATOMIC OUTPUT HELPER
# Collects all lines then writes in one syscall.
# Prevents ANY interleaving with background threads.
# ─────────────────────────────────────────────
class OutputBuffer:
    def __init__(self):
        self._lines = []

    def add(self, text: str = ""):
        self._lines.append(text)

    def flush(self):
        """Write everything to stdout in a single atomic call."""
        sys.stdout.write("\n".join(self._lines) + "\n")
        sys.stdout.flush()
        self._lines = []


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    validate_env()

    resume_folder = "data/resumes"
    if not os.path.isdir(resume_folder):
        logger.error(f"Folder '{resume_folder}' not found.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(resume_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.error("No PDF resumes found in data/resumes/")
        sys.exit(1)

    # ── Interactive header (printed live, before bulk output) ----------
    print()
    print("=" * 56)
    print("  HYBRID ATS RESUME SCREENER")
    print("  60% LLM Skill Score  +  40% Semantic Cosine Score")
    print("=" * 56)
    print()

    # ── Job Description ------------------------------------------------
    jd_raw = input("Paste the Job Description (press Enter when done):\n").strip()
    if not jd_raw:
        logger.error("Job description cannot be empty.")
        sys.exit(1)
    jd_text = clean_text(jd_raw)

    # ── Skill weights --------------------------------------------------
    skill_weights: dict = {}
    print()
    print("Enter skills and weights.")
    print("  10=must-have   5=nice-to-have   1=bonus")
    print("  Type 'done' when finished.\n")

    while True:
        skill = input("  Skill  : ").lower().strip()
        if skill == "done":
            break
        if not skill:
            continue
        if skill in skill_weights:
            print(f"  [skip] '{skill}' already added.")
            continue
        try:
            weight = int(input("  Weight : "))
            if weight <= 0:
                raise ValueError
        except ValueError:
            print("  [!] Enter a positive integer.")
            continue
        skill_weights[skill] = weight
        print(f"  [+] {skill}  (weight={weight})")

    if not skill_weights:
        logger.error("No skills entered.")
        sys.exit(1)

    print(f"\n  Skills : {list(skill_weights.keys())}")
    print(f"  Found  : {len(pdf_files)} resume(s)\n")

    # ── Load models ---------------------------------------------------
    print("Loading models ...")
    embedder     = Embedder()
    jd_embedding = embedder.get_embedding(jd_text)
    get_nlp()

    # Precompute skill embeddings ONCE (not per resume).
    # These are 100% deterministic and act as the stable anchor in scoring.
    skill_embeddings = {skill: embedder.get_embedding(skill) for skill in skill_weights}

    print("Models ready.\n")

    # ── Suppress all logging during processing so nothing sneaks into
    #    the progress lines
    logging.disable(logging.WARNING)

    # ══════════════════════════════════════════════════════════════════
    # PROCESSING LOOP
    # Progress is printed AFTER each resume finishes  (no `end=""`)
    # so that no background thread output can appear on the same line.
    # Per-resume details go into a buffer, printed atomically at end.
    # ══════════════════════════════════════════════════════════════════
    print("=" * 56)
    print("  Processing resumes ...")
    print("=" * 56)

    results: list         = []
    all_details: list     = []
    detail_buffer: list   = []  # one string per resume, printed after loop

    for idx, file in enumerate(pdf_files, 1):
        path    = os.path.join(resume_folder, file)
        t_start = time.time()

        # ── Extract raw text ------------------------------------------
        raw_text = extract_text_from_pdf(path)
        if not raw_text or not raw_text.strip():
            elapsed = round(time.time() - t_start, 1)
            # Print AFTER processing (no end="" race condition)
            print(f"  [{idx:>2}/{len(pdf_files)}]  {file}  --  SKIP (empty/scanned PDF)  {elapsed}s")
            continue

        # ── Entity extraction (from raw text, before cleaning) ---------
        info = extract_structured_info(raw_text)

        # ── Clean text for embeddings + LLM ---------------------------
        text = clean_text(raw_text)

        # ── Cosine score (40 pts) -------------------------------------
        resume_embedding = embedder.get_embedding(text)
        cosine_score     = compute_cosine_score_40(
            compute_similarity(resume_embedding, jd_embedding)
        )

        # ── Sentence splitting + batch embed (once per resume) --------
        sentences  = clean_sentences(split_sentences(raw_text))
        sent_embs  = embedder.get_batch_embeddings(sentences) if sentences else []

        # ── Skill context map + section-tagged keyword injection ---------
        # NEW: get_skill_context_tagged() does two things:
        #   1. Keyword scan: any sentence containing the skill word is tagged
        #      and prepended as [SKILLS]/[WORK EXP]/[OBJECTIVE]... so the LLM
        #      knows WHERE the evidence comes from (crucial for accuracy).
        #   2. Semantic retrieval: top-5 sentences by cosine, also section-tagged.
        # _MIN_CONTEXT_SIM lowered to 0.28 (was 0.38) — less aggressive filtering.
        # Keyword hits bypass the threshold entirely.
        _MIN_CONTEXT_SIM = 0.28

        skill_context_map  = {}
        skill_max_sims     = {}
        skill_keyword_hits = {}   # bool: keyword found in resume for this skill
        for skill in skill_weights:
            tagged_ctx, max_sim, kw_found = get_skill_context_tagged(
                skill, raw_text, sentences, embedder, sent_embs
            )
            # Pass context if: keyword found OR embedding similarity above threshold
            skill_context_map[skill]  = tagged_ctx if (kw_found or max_sim >= _MIN_CONTEXT_SIM) else []
            skill_max_sims[skill]     = max_sim
            skill_keyword_hits[skill] = kw_found

        # ── LLM evaluation (60 pts) ------------------------------------
        llm_results = evaluate_all_skills_with_llm(
            list(skill_weights.keys()),
            skill_context_map,
            resume_text = raw_text,
            jd_text     = jd_text,
        )

        # ── Inject per-skill cosine (deterministic 85% anchor) ----------
        for d in llm_results:
            sk     = d["skill"]
            sk_emb = skill_embeddings.get(sk)
            if sk_emb is not None:
                broad_sim    = float(util.cos_sim(sk_emb, resume_embedding).item())
                sent_max_sim = skill_max_sims.get(sk, 0.0)
                d["per_skill_cosine"]   = max(0.0, min(1.0, max(broad_sim, sent_max_sim)))
                d["keyword_in_resume"]  = skill_keyword_hits.get(sk, False)
            else:
                d["per_skill_cosine"]  = 0.0
                d["keyword_in_resume"] = False

        all_details.append(llm_results)

        # ── Score calculation -----------------------------------------
        skill_score = compute_skill_score(llm_results, skill_weights)
        final_score = round(skill_score + cosine_score, 2)
        elapsed     = round(time.time() - t_start, 1)

        all_cached = all(d.get("cache_hit", False) for d in llm_results)
        cache_tag  = "CACHED" if all_cached else "LLM"

        results.append({
            "file":         file,
            "name":         info.get("name",  "Unknown"),
            "email":        info.get("email", "N/A"),
            "llm_score":    skill_score,
            "cosine_score": cosine_score,
            "final_score":  final_score,
            "latency_s":    elapsed,
            "cache_tag":    cache_tag,
        })

        # ── Progress line — printed AFTER all processing (no end="") --
        print(f"  [{idx:>2}/{len(pdf_files)}]  {file:<35}  {elapsed:5.1f}s  [{cache_tag}]")

        # ── Buffer resume detail block (printed atomically later) ------
        lines = [
            "",
            "-" * 56,
            f"  File   : {file}",
            f"  Name   : {info.get('name',  'N/A')}",
            f"  Email  : {info.get('email', 'N/A')}",
            "-" * 56,
            f"  LLM Score    : {skill_score:6.2f} / 60   [{cache_tag}]",
            f"  Cosine Score : {cosine_score:6.2f} / 40",
            f"  Final Score  : {final_score:6.2f} / 100",
            f"  Time         : {elapsed}s",
            "",
            "  Skill Breakdown:",
        ]
        for d in llm_results:
            icon  = {"strong": "[+]", "confirmed": "[*]", "moderate": "[~]", "weak": "[-]"}.get(d["strength"], "[ ]")
            psc   = d.get("per_skill_cosine", 0.0)
            lines.append(f"    {icon} [{d['strength'].upper():10s}] {d['skill']}")
            lines.append(
                f"         Conf:{d['confidence']:.2f}  "
                f"SkillSim:{psc:.2f}  "
                f"Src:{d['source'][:10]}  Depth:{d['depth'][:8]}"
            )
            if d.get("hallucination_cleared"):
                lines.append("         [GUARD: hallucinated evidence was cleared — see reason]")
            raw_ev = d["evidence"] if d["evidence"] else "(none found in resume)"
            # Hard-wrap at 65 chars to prevent terminal-width artifacts
            for chunk_start in range(0, len(raw_ev), 65):
                prefix = "         Evidence: " if chunk_start == 0 else "                   "
                lines.append(prefix + raw_ev[chunk_start:chunk_start + 65])
            lines.append(f"         Reason  : {d['reason'][:75]}")

        detail_buffer.append("\n".join(lines))

    # ── Restore logging ------------------------------------------------
    logging.disable(logging.NOTSET)

    if not results:
        logger.error("No resumes were processed successfully.")
        sys.exit(1)

    # ── Sort results --------------------------------------------------
    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # ══════════════════════════════════════════════════════════════════
    # BUILD ALL POST-LOOP OUTPUT AS A SINGLE STRING
    # One sys.stdout.write() call = no interleaving possible.
    # ══════════════════════════════════════════════════════════════════
    out = OutputBuffer()

    # ── RESUME DETAILS ─────────────────────────────────────────────
    out.add()
    out.add("=" * 56)
    out.add("  RESUME DETAILS")
    out.add("=" * 56)
    for block in detail_buffer:
        out.add(block)

    # ── FINAL RANKING ─────────────────────────────────────────────
    out.add()
    out.add("=" * 56)
    out.add("  FINAL RANKING")
    out.add("=" * 56)

    medals = {1: " #1", 2: " #2", 3: " #3"}
    for i, r in enumerate(ranked, 1):
        bar   = "#" * int(r["final_score"] / 5)
        medal = medals.get(i, f"{i:>3}")
        ctag  = r.get("cache_tag", "LLM")
        out.add(f"\n  {medal}  {r['file']}")
        out.add(f"       Score  : {r['final_score']:5.2f} / 100  {bar}")
        out.add(f"       LLM    : {r['llm_score']:5.2f} / 60   Cosine : {r['cosine_score']:5.2f} / 40")
        out.add(f"       Name   : {r['name']}   Time : {r['latency_s']}s  [{ctag}]")

    # ── EVALUATION REPORT ─────────────────────────────────────────
    out.add(format_evaluation(ranked, all_details))

    # ── Write everything at once ──────────────────────────────────
    out.flush()


if __name__ == "__main__":
    main()