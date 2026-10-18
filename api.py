"""
FastAPI Backend Service for AI Job Description–Driven Assessment Blueprint Builder
Fulfills Enterprise Project Guideline: 'Backend APIs: FastAPI'.
"""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import core

app = FastAPI(
    title="AI Assessment Blueprint Builder API",
    description="Multi-Agent GenAI REST API for Job Description Analysis, Adaptive Blueprint Generation, Question Retrieval, and Candidate Simulation.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Enable CORS for frontend clients (e.g., Streamlit, React, Gradio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------
class TestKeyRequest(BaseModel):
    api_key: str = Field(..., description="API Key to validate")
    provider: str = Field("Gemini", description="AI Provider: Google Gemini")
    model: str = Field("gemini-3.8-flash", description="Target model name")


class TestKeyResponse(BaseModel):
    is_valid: bool
    message: str


class JDExtractRequest(BaseModel):
    jd_text: str = Field(..., description="Raw text of the job description")
    use_ai: bool = Field(False, description="Whether to invoke AI or local rule-based extractor")
    api_key: Optional[str] = Field(None, description="API Key if use_ai is True")
    provider: Optional[str] = Field("Gemini", description="AI Provider: Google Gemini")
    model: Optional[str] = Field("gemini-3.8-flash", description="Model identifier")


class BlueprintRequest(BaseModel):
    competencies: List[Dict[str, Any]] = Field(..., description="List of confirmed competencies")
    seniority: str = Field("Mid", description="Seniority level: Entry, Junior, Mid, Senior, Lead")
    total_questions: int = Field(10, ge=1, le=50)
    duration: int = Field(60, ge=10, le=240)
    easy_pct: int = Field(30, ge=0, le=100)
    medium_pct: int = Field(50, ge=0, le=100)
    hard_pct: int = Field(20, ge=0, le=100)


class QuestionGenerateRequest(BaseModel):
    competency: str
    difficulty: str = Field("Medium", pattern="^(Easy|Medium|Hard)$")
    jd_text: str
    api_key: Optional[str] = Field("", description="API Key if available")
    provider: Optional[str] = Field("Gemini", description="AI Provider: Google Gemini")
    model: str = "gemini-3.8-flash"


class CandidateSimRequest(BaseModel):
    blueprint: List[Dict[str, Any]]
    duration: int = 60


class HighlightRequest(BaseModel):
    jd_text: str
    competencies: List[Dict[str, Any]]


class RAGSearchRequest(BaseModel):
    query: str
    competency: Optional[str] = None
    difficulty: Optional[str] = None
    top_k: int = 5


class TagAuditRequest(BaseModel):
    question_text: str
    assigned_competency: str


class RecommendTagsRequest(BaseModel):
    question_text: str
    top_k: int = 3


class ExplainabilityRequest(BaseModel):
    jd_text: str
    competencies: List[Dict[str, Any]]
    blueprint: List[Dict[str, Any]]
    seniority: str = "Mid"
    duration: int = 60
    difficulty_mix: Optional[Dict[str, int]] = None


class FeedbackEntryRequest(BaseModel):
    item_type: str = Field(..., description="competency_tag, question_tag, difficulty, rubric, blueprint")
    target_id: str
    original_value: Any
    corrected_value: Any
    notes: Optional[str] = ""
    reviewer: Optional[str] = "Lead Assessment Architect"


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "AI Assessment Blueprint Builder API",
        "status": "online",
        "documentation": "FastAPI REST API (Swagger UI disabled per project specification)",
        "guideline_compliance": "FastAPI Backend Enterprise Architecture Spec",
    }


@app.get("/api/health")
def health_check():
    gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    return {
        "status": "healthy",
        "provider": "Google Gemini",
        "taxonomy_competency_count": len(core.TAXONOMY),
        "question_bank_size": len(core.QUESTION_BANK),
        "gemini_configured": bool(gemini_key and gemini_key != "your_openai_api_key_here"),
    }


@app.post("/api/test-key", response_model=TestKeyResponse)
def test_key_endpoint(req: TestKeyRequest):
    is_valid, msg = core.test_ai_key(req.api_key, req.provider, req.model)
    return TestKeyResponse(is_valid=is_valid, message=msg)


@app.get("/api/taxonomy")
def get_taxonomy():
    return core.TAXONOMY


@app.get("/api/questions")
def get_questions(
    competency: Optional[str] = Query(None, description="Filter by competency"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
):
    questions = core.QUESTION_BANK
    if competency:
        questions = [q for q in questions if q["competency"].lower() == competency.lower()]
    if difficulty:
        questions = [q for q in questions if q["difficulty"].lower() == difficulty.lower()]
    return {
        "count": len(questions),
        "questions": questions,
    }


@app.post("/api/extract-jd")
def extract_job_description(req: JDExtractRequest):
    if not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")
    
    try:
        if req.use_ai and req.api_key:
            try:
                result = core.ai_based_extraction(req.jd_text, req.api_key, req.model or "gemini-3.8-flash", provider=req.provider or "Gemini")
            except Exception as ai_err:
                result = core.rule_based_extraction(req.jd_text)
                result["extraction_warning"] = f"AI extraction notice: {str(ai_err)}. Fallback to rule-based parser applied."
        else:
            result = core.rule_based_extraction(req.jd_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/api/adaptive-mix/{seniority}")
def get_adaptive_mix(seniority: str):
    return core.get_adaptive_difficulty_mix(seniority)


@app.post("/api/build-blueprint")
def generate_blueprint(req: BlueprintRequest):
    blueprint, warnings = core.build_adaptive_blueprint(
        competencies=req.competencies,
        seniority=req.seniority,
        total_questions=req.total_questions,
        duration=req.duration,
        easy_pct=req.easy_pct,
        medium_pct=req.medium_pct,
        hard_pct=req.hard_pct,
    )
    return {
        "blueprint": blueprint,
        "warnings": warnings,
        "seniority": req.seniority,
        "target_duration": req.duration,
        "total_questions": req.total_questions,
    }


@app.post("/api/generate-question")
def generate_draft_question(req: QuestionGenerateRequest):
    try:
        draft = core.generate_gap_question(req.competency, req.jd_text, req.difficulty, req.api_key or "", req.model, provider=req.provider or "Gemini")
        return draft
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {str(e)}")


@app.post("/api/simulate-candidates")
def simulate_candidates(req: CandidateSimRequest):
    sim = core.run_candidate_simulation(req.blueprint, req.duration)
    if not sim:
        raise HTTPException(status_code=400, detail="Blueprint has no selected questions to simulate.")
    return sim


@app.post("/api/highlight-evidence")
def highlight_evidence(req: HighlightRequest):
    html = core.highlight_jd_text(req.jd_text, req.competencies)
    return {"highlighted_html": html}


# ---------------------------------------------------------
# RAG, Explainability & Feedback Endpoints
# ---------------------------------------------------------
@app.get("/api/rag/status")
def get_rag_status():
    rag = core.init_rag_system()
    if not rag:
        return {"status": "unavailable", "message": "RAG engine not initialized"}
    return {"status": "active", **rag.status()}


@app.post("/api/rag/search")
def rag_search_questions(req: RAGSearchRequest):
    rag = core.init_rag_system()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    results = rag.retrieve_questions(
        query=req.query,
        competency=req.competency,
        difficulty=req.difficulty,
        top_k=req.top_k,
    )
    return {"count": len(results), "results": results}


@app.post("/api/rag/recommend-tags")
def recommend_tags_endpoint(req: RecommendTagsRequest):
    rag = core.init_rag_system()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    tags = rag.recommend_tags(req.question_text, top_k=req.top_k)
    return {"question_text": req.question_text, "recommended_tags": tags}


@app.post("/api/rag/audit-question")
def audit_question_tag(req: TagAuditRequest):
    rag = core.init_rag_system()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    audit = rag.detect_wrong_tags(req.question_text, req.assigned_competency)
    return audit


@app.post("/api/explainability")
def explainability_report_endpoint(req: ExplainabilityRequest):
    report = core.generate_explainability_report(
        jd_text=req.jd_text,
        competencies=req.competencies,
        blueprint=req.blueprint,
        seniority=req.seniority,
        duration=req.duration,
        difficulty_mix=req.difficulty_mix,
    )
    return report


@app.get("/api/feedback")
def get_feedback_history():
    history = core.load_reviewer_feedback()
    return {"count": len(history), "feedback": history}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackEntryRequest):
    updated = core.save_reviewer_feedback(req.dict())
    return {"status": "success", "message": "Feedback recorded", "total_records": len(updated)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


