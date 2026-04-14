import json
import logging
import os
import tempfile
import time
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import util

from main import clean_sentences, get_nlp, get_skill_context_tagged, split_sentences
from matching.embedder import Embedder
from matching.llm_scorer import evaluate_all_skills_with_llm
from matching.score_calculator import compute_skill_score
from matching.similarity import compute_cosine_score_40, compute_similarity
from parser.entity_extractor import extract_structured_info
from preprocessing.text_cleaner import clean_text
from preprocessing.text_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid ATS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

_embedder = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
        get_nlp()
    return _embedder


def _parse_skill_weights(skills_json: str) -> Dict[str, int]:
    try:
        parsed = json.loads(skills_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid skills JSON: {exc}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise HTTPException(status_code=400, detail="skills_json must be a non-empty JSON object.")

    skill_weights: Dict[str, int] = {}
    for skill, weight in parsed.items():
        skill_name = str(skill).strip().lower()
        if not skill_name:
            continue
        try:
            weight_int = int(weight)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid weight for '{skill_name}'.") from exc
        if weight_int <= 0:
            raise HTTPException(status_code=400, detail=f"Weight for '{skill_name}' must be > 0.")
        skill_weights[skill_name] = weight_int

    if not skill_weights:
        raise HTTPException(status_code=400, detail="At least one valid skill is required.")
    return skill_weights


def _process_resume(
    filename: str,
    raw_text: str,
    jd_text: str,
    jd_embedding,
    skill_weights: Dict[str, int],
    skill_embeddings,
    embedder: Embedder,
):
    started = time.time()

    info = extract_structured_info(raw_text)
    cleaned_resume_text = clean_text(raw_text)
    resume_embedding = embedder.get_embedding(cleaned_resume_text)
    cosine_score = compute_cosine_score_40(compute_similarity(resume_embedding, jd_embedding))

    sentences = clean_sentences(split_sentences(raw_text))
    sent_embeddings = embedder.get_batch_embeddings(sentences) if sentences else []

    min_context_similarity = 0.28
    skill_context_map = {}
    skill_max_sims = {}
    skill_keyword_hits = {}
    for skill in skill_weights:
        tagged_ctx, max_sim, kw_found = get_skill_context_tagged(
            skill, raw_text, sentences, embedder, sent_embeddings
        )
        skill_context_map[skill] = tagged_ctx if (kw_found or max_sim >= min_context_similarity) else []
        skill_max_sims[skill] = max_sim
        skill_keyword_hits[skill] = kw_found

    llm_results = evaluate_all_skills_with_llm(
        list(skill_weights.keys()),
        skill_context_map,
        resume_text=raw_text,
        jd_text=jd_text,
    )

    for detail in llm_results:
        skill = detail["skill"]
        skill_emb = skill_embeddings.get(skill)
        if skill_emb is not None:
            broad_sim = float(util.cos_sim(skill_emb, resume_embedding).item())
            sent_max_sim = skill_max_sims.get(skill, 0.0)
            detail["per_skill_cosine"] = max(0.0, min(1.0, max(broad_sim, sent_max_sim)))
            detail["keyword_in_resume"] = skill_keyword_hits.get(skill, False)
        else:
            detail["per_skill_cosine"] = 0.0
            detail["keyword_in_resume"] = False

    llm_score = compute_skill_score(llm_results, skill_weights)
    final_score = round(llm_score + cosine_score, 2)

    return {
        "file": filename,
        "name": info.get("name", "Unknown"),
        "email": info.get("email", "N/A"),
        "llm_score": llm_score,
        "cosine_score": cosine_score,
        "final_score": final_score,
        "latency_s": round(time.time() - started, 2),
        "skill_details": llm_results,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_file):
        return {"message": "Frontend not found. Open /docs for API usage."}
    return FileResponse(index_file)


@app.post("/api/screen")
async def screen_resumes(
    job_description: str = Form(...),
    skills_json: str = Form(...),
    resumes: List[UploadFile] = File(...),
):
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set.")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required.")
    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume PDF is required.")

    skill_weights = _parse_skill_weights(skills_json)
    jd_text = clean_text(job_description)
    embedder = get_embedder()
    jd_embedding = embedder.get_embedding(jd_text)
    skill_embeddings = {skill: embedder.get_embedding(skill) for skill in skill_weights}

    results = []
    skipped = []

    for upload in resumes:
        filename = upload.filename or "resume.pdf"
        if not filename.lower().endswith(".pdf"):
            skipped.append({"file": filename, "reason": "Only PDF files are supported."})
            continue

        payload = await upload.read()
        if not payload:
            skipped.append({"file": filename, "reason": "Uploaded file is empty."})
            continue

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(payload)
                temp_path = tmp.name
            raw_text = extract_text_from_pdf(temp_path)
            if not raw_text or not raw_text.strip():
                skipped.append({"file": filename, "reason": "Could not extract readable text from PDF."})
                continue

            result = _process_resume(
                filename=filename,
                raw_text=raw_text,
                jd_text=jd_text,
                jd_embedding=jd_embedding,
                skill_weights=skill_weights,
                skill_embeddings=skill_embeddings,
                embedder=embedder,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Failed processing resume: %s", filename)
            skipped.append({"file": filename, "reason": f"Processing failed: {exc}"})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    ranked = sorted(results, key=lambda item: item["final_score"], reverse=True)

    return {
        "job_description_length": len(job_description),
        "skill_weights": skill_weights,
        "processed_count": len(ranked),
        "skipped_count": len(skipped),
        "ranked_results": ranked,
        "skipped_files": skipped,
    }
