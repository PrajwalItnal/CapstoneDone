"""
Streamlit Frontend for AI Job Description–Driven Assessment Blueprint Builder
Connects to FastAPI Backend Service (http://127.0.0.1:8000) with automatic local fallback.
Conforms to Enterprise Project Guidelines:
- Frontend: Streamlit
- Backend APIs: FastAPI
"""

import os
import re
import json
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

import core

APP_TITLE = "AI Job Description–Driven Assessment Blueprint Builder"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------
# FastAPI Client Integration Wrappers
# ---------------------------------------------------------
@st.cache_data(ttl=15)
def get_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=1.5)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}


def client_extract_jd(jd_text, use_ai, api_key, model, provider="Gemini"):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/extract-jd", json={
                "jd_text": jd_text, "use_ai": use_ai, "api_key": api_key, "model": model, "provider": provider
            }, timeout=90)
            if r.ok:
                return r.json(), f"FastAPI Backend ({provider})"
        except Exception:
            pass
    if use_ai and api_key:
        try:
            res = core.ai_based_extraction(jd_text, api_key, model, provider=provider)
            return res, f"Local Core Engine ({provider})"
        except Exception as ex:
            res = core.rule_based_extraction(jd_text)
            res["extraction_warning"] = f"AI extraction notice ({ex}). Fallback applied."
            return res, f"Local Core Engine (Rule-based Fallback)"
    res = core.rule_based_extraction(jd_text)
    return res, f"Local Core Engine (Rule-based)"


def client_build_blueprint(competencies, seniority, total_questions, duration, easy_pct, medium_pct, hard_pct):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/build-blueprint", json={
                "competencies": competencies, "seniority": seniority, "total_questions": total_questions,
                "duration": duration, "easy_pct": easy_pct, "medium_pct": medium_pct, "hard_pct": hard_pct
            }, timeout=30)
            if r.ok:
                data = r.json()
                return data["blueprint"], data["warnings"], "FastAPI Backend"
        except Exception:
            pass
    b, w = core.build_adaptive_blueprint(competencies, seniority, total_questions, duration, easy_pct, medium_pct, hard_pct)
    return b, w, "Local Core Engine"


def client_generate_question(competency, jd_text, difficulty, api_key, model, provider="Gemini"):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/generate-question", json={
                "competency": competency, "difficulty": difficulty, "jd_text": jd_text,
                "api_key": api_key, "model": model, "provider": provider
            }, timeout=90)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return core.generate_gap_question(competency, jd_text, difficulty, api_key, model, provider=provider)


def client_simulate_candidates(blueprint, duration):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/simulate-candidates", json={
                "blueprint": blueprint, "duration": duration
            }, timeout=10)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return core.run_candidate_simulation(blueprint, duration)


def client_highlight_evidence(jd_text, competencies):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/highlight-evidence", json={
                "jd_text": jd_text, "competencies": competencies
            }, timeout=10)
            if r.ok:
                return r.json()["highlighted_html"]
        except Exception:
            pass
    return core.highlight_jd_text(jd_text, competencies)


# ---------------------------------------------------------
# Streamlit Application Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .main-header { font-size: 2.1rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-caption { font-size: 1.0rem; color: #64748b; margin-bottom: 0.8rem; }
    .api-badge-ok { background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; border: 1px solid #86efac; display: inline-block; margin-bottom: 12px; }
    .api-badge-warn { background-color: #fef9c3; color: #854d0e; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; border: 1px solid #fde047; display: inline-block; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AI & Engine Configuration (Auto-detected with seamless local fallback)
# ---------------------------------------------------------
provider_choice = "Google Gemini"
provider_code = "Gemini"

gemini_env = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
api_key_input = gemini_env.strip() if gemini_env else ""
ai_toggle = bool(api_key_input)
model_choice = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

st.markdown(f'<div class="main-header">🎯 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Enterprise Multi-Agent Assessment Pipeline • Streamlit UI + FastAPI Backend Architecture</div>', unsafe_allow_html=True)

# Backend & RAG Engine status indicators
backend_online, backend_meta = get_api_health()
@st.cache_resource
def get_cached_rag():
    return core.init_rag_system()

rag_inst = get_cached_rag()
rag_status = rag_inst.status() if rag_inst else {}

c_st1, c_st2, c_st3 = st.columns([1.5, 2.2, 4])
with c_st1:
    if backend_online:
        st.markdown('<div class="api-badge-ok">🟢 FastAPI Backend Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-badge-warn">🟡 Local Core Engine</div>', unsafe_allow_html=True)
with c_st2:
    if rag_status.get("faiss") and rag_status.get("chromadb"):
        st.markdown('<div class="api-badge-ok">🧠 RAG Engine Active (FAISS + Chroma)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-badge-warn">⚪ Keyword Matcher Active</div>', unsafe_allow_html=True)
with c_st3:
    if ai_toggle and (api_key_input and api_key_input.strip()):
        st.markdown(f'<div class="api-badge-ok">🔑 Gemini AI Active ({model_choice})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-badge-ok">⚡ Local Rule Engine Active</div>', unsafe_allow_html=True)

# Session State Initialization
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "competencies" not in st.session_state:
    st.session_state.competencies = []
if "blueprint" not in st.session_state:
    st.session_state.blueprint = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []
if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = []

def log_audit(event):
    st.session_state.audit_logs.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event
    })

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 🤖 JD Analysis & Evidence Map",
    "2. 🧭 Competency Mapping & Review",
    "3. 📐 Adaptive Blueprint",
    "4. 🔍 Questions & Gap Generator",
    "5. 🛡️ Candidate Simulation & Audit"
])


# ---------------------------------------------------------
# TAB 1: Job Description Ingestion & Parsing
# ---------------------------------------------------------
with tab1:
    st.subheader("Stage 1: Job Description Ingestion & Analysis Agent")
    
    c_up1, c_up2 = st.columns([1, 1])
    with c_up1:
        uploaded_doc = st.file_uploader("Upload Job Description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if uploaded_doc:
            try:
                parsed_text = core.extract_text_from_bytes(uploaded_doc.name, uploaded_doc.getvalue())
                if parsed_text.strip():
                    st.session_state.jd_text = parsed_text
                    st.success(f"Successfully extracted {len(parsed_text)} characters from {uploaded_doc.name}")
                    log_audit(f"Parsed uploaded file: {uploaded_doc.name}")
            except Exception as err:
                st.error(f"Failed to read file: {err}")

    with c_up2:
        default_sample = (
            "Senior Backend Engineer\n\n"
            "Responsibilities:\n"
            "- Design and implement scalable REST APIs and microservices in Python (FastAPI/Django).\n"
            "- Optimize PostgreSQL databases, queries, and complex joins.\n"
            "- Deploy containerized workloads using Docker and Kubernetes on AWS.\n"
            "- Lead system design discussions for high-throughput distributed systems.\n"
            "- Mentor junior engineers, conduct code reviews, and champion unit testing with Pytest.\n\n"
            "Requirements:\n"
            "- 5+ years experience in Python and SQL.\n"
            "- Must have strong problem solving and data structures background.\n"
            "- Mandatory experience in Git collaboration, CI/CD, and secure OAuth2 authentication."
        )
        if st.button("Load Sample Senior Backend JD"):
            st.session_state.jd_text = default_sample

    jd_editor = st.text_area(
        "Job Description Content:",
        value=st.session_state.jd_text,
        height=220,
        placeholder="Paste JD text here..."
    )

    if st.button("🚀 Run Analysis Agent", type="primary", use_container_width=True):
        st.session_state.jd_text = jd_editor
        if not jd_editor.strip():
            st.error("Please provide a job description first.")
        else:
            with st.spinner("Analyzing Job Description via FastAPI Agent..."):
                try:
                    analysis, engine_name = client_extract_jd(jd_editor, ai_toggle, api_key_input, model_choice, provider=provider_code)
                    log_audit(f"Executed JD extraction via {engine_name}.")
                    st.session_state.analysis = analysis
                    st.session_state.competencies = analysis.get("competencies", [])
                    st.session_state.blueprint = []
                    st.success(f"Extracted {len(st.session_state.competencies)} competencies across role: '{analysis.get('role_title', 'Engineer')}' (Processed via {engine_name})")
                except Exception as ex:
                    st.error(f"Analysis failed: {ex}")

    if st.session_state.analysis:
        st.divider()
        st.markdown("### 📊 Extracted Role Summary")
        a = st.session_state.analysis
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Identified Title", a.get("role_title", "Unknown"))
        m2.metric("Detected Seniority", a.get("seniority", "Mid"))
        m3.metric("Competencies Found", len(st.session_state.competencies))

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### 🎯 Key Responsibilities")
            for r in a.get("responsibilities", []):
                st.markdown(f"- {r}")
            st.markdown("#### 🛠️ Tools & Technologies")
            st.write(", ".join(a.get("tools", [])) if a.get("tools") else "None explicitly isolated.")

        with col_right:
            st.markdown("#### 🤝 Observable Behaviors")
            for b in a.get("observable_behaviors", []):
                st.markdown(f"- {b}")
            if a.get("ambiguities"):
                st.warning("⚠️ **Ambiguities Detected in JD:**\n" + "\n".join(f"- {amb}" for amb in a["ambiguities"]))

        # Requirement 2: Missing or Ambiguous Requirements for Reviewer Confirmation
        c_qs = a.get("clarification_questions", [])
        if c_qs:
            st.divider()
            st.markdown("### ❓ Missing & Ambiguous Requirements: Clarification Questions")
            st.caption("The AI Agent identified missing or ambiguous requirements in the JD. Answer these questions to confirm specifications:")
            
            for q_idx, cq in enumerate(c_qs):
                with st.expander(f"Question {q_idx+1}: {cq['topic']} — {cq['question'][:75]}...", expanded=True):
                    st.markdown(f"**{cq['question']}**")
                    chosen_opt = st.radio(
                        "Reviewer Decision:",
                        cq["options"],
                        index=0,
                        key=f"cq_opt_{q_idx}"
                    )
                    if st.button(f"Apply Action: {chosen_opt[:35]}...", key=f"cq_btn_{q_idx}"):
                        act_comp = cq.get("action_competency")
                        if act_comp:
                            existing_names = [c["competency"] for c in st.session_state.competencies]
                            if act_comp not in existing_names:
                                st.session_state.competencies.append({
                                    "competency": act_comp,
                                    "category": "Technical",
                                    "mandatory": "Mandatory" in chosen_opt or "Yes" in chosen_opt,
                                    "proficiency": "Intermediate",
                                    "importance": 4,
                                    "confidence": 0.90,
                                    "matched_terms": [act_comp],
                                    "evidence": ["Added via Reviewer Clarification Answer"],
                                })
                            else:
                                for c in st.session_state.competencies:
                                    if c["competency"] == act_comp:
                                        c["mandatory"] = "Mandatory" in chosen_opt or "Yes" in chosen_opt
                            log_audit(f"Reviewer answered clarification question {cq['id']}: '{chosen_opt}'.")
                            st.success(f"Requirement confirmed and applied: {act_comp}")
                            st.rerun()

        # Innovative Workflow #1: Interactive JD Evidence Map
        st.divider()
        st.markdown("### 🗺️ Innovative Workflow 1: JD Evidence Map")
        st.caption("Visual phrase highlighting mapping detected skills to raw text context:")
        highlighted_html = client_highlight_evidence(st.session_state.jd_text, st.session_state.competencies)
        st.markdown(
            f'<div style="background-color: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #cbd5e1; color: #0f172a; line-height: 1.8; max-height: 300px; overflow-y: auto;">{highlighted_html}</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# TAB 2: Competency Mapping & Reviewer Confirmation
# ---------------------------------------------------------
with tab2:
    st.subheader("Stage 2: Competency Normalization & Human-in-the-Loop Review Agent")
    st.caption("Review extracted competencies, confirm mandatory flags, and adjust proficiency requirements.")

    if not st.session_state.competencies:
        st.info("Run the JD Analysis Agent in Tab 1 to populate competencies.")
    else:
        edited_comps = st.data_editor(
            st.session_state.competencies,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "mandatory": st.column_config.CheckboxColumn("Mandatory?", default=False),
                "proficiency": st.column_config.SelectboxColumn("Proficiency", options=["Beginner", "Intermediate", "Advanced"]),
                "confidence": st.column_config.ProgressColumn("Confidence Score", min_value=0.0, max_value=1.0, format="%.2f"),
                "importance": st.column_config.NumberColumn("Importance (1-5)", min_value=1, max_value=5),
            },
            key="competency_data_editor"
        )
        if hasattr(edited_comps, "to_dict"):
            st.session_state.competencies = edited_comps.to_dict(orient="records")
        elif isinstance(edited_comps, list):
            st.session_state.competencies = list(edited_comps)

        if st.button("💾 Confirm Competency Mapping"):
            log_audit("Reviewer confirmed competency taxonomy mapping.")
            st.success("Competency mapping confirmed. Proceed to Tab 3 to configure the blueprint.")

        # Innovative Workflow #6: Human-in-the-Loop Feedback Loop
        st.divider()
        st.markdown("#### 🔄 Innovative Workflow 6: Human-in-the-Loop Feedback Loop")
        st.caption("Store expert corrections to fine-tune competency taxonomy mapping and future extractions.")
        c_fb1, c_fb2, c_fb3 = st.columns([1.5, 1.5, 2])
        comp_options = [c.get("competency", "") for c in st.session_state.competencies if c.get("competency")]
        with c_fb1:
            fb_target = st.selectbox("Select Extracted Skill", comp_options, key="fb_target_comp") if comp_options else None
        with c_fb2:
            all_taxonomy_names = list(core.TAXONOMY.keys())
            fb_correct = st.selectbox("Map to Canonical Competency", all_taxonomy_names, key="fb_correct_comp")
        with c_fb3:
            fb_notes = st.text_input("Reviewer Notes / Rationale", placeholder="e.g. Map FastAPI to Python Core & Frameworks", key="fb_notes_comp")

        if st.button("📤 Submit Correction to Feedback Loop", disabled=not fb_target):
            core.save_reviewer_feedback({
                "item_type": "competency_tag",
                "target_id": fb_target,
                "original_value": fb_target,
                "corrected_value": fb_correct,
                "notes": fb_notes,
                "reviewer": "Lead Assessment Architect"
            })
            for c in st.session_state.competencies:
                if c.get("competency") == fb_target:
                    c["competency"] = fb_correct
                    c["reviewer_corrected"] = True
            log_audit(f"Reviewer submitted feedback correction: '{fb_target}' -> '{fb_correct}'.")
            st.success(f"Feedback stored! '{fb_target}' normalized to '{fb_correct}' in feedback loop.")
            st.rerun()


# ---------------------------------------------------------
# TAB 3: Adaptive Blueprint Design Agent
# ---------------------------------------------------------
with tab3:
    st.subheader("Stage 3: Adaptive Blueprint Design Agent")
    st.caption("Configures assessment weight, duration, and dynamically adapts difficulty mix to role seniority.")

    if not st.session_state.competencies:
        st.info("Analyze a JD first to configure its blueprint.")
    else:
        detected_seniority = st.session_state.analysis.get("seniority", "Mid") if st.session_state.analysis else "Mid"
        
        c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
        with c_cfg1:
            valid_seniorities = ["Entry", "Junior", "Mid", "Senior", "Lead"]
            seniority_idx = valid_seniorities.index(detected_seniority) if detected_seniority in valid_seniorities else 2
            selected_seniority = st.selectbox(
                "Role Seniority",
                valid_seniorities,
                index=seniority_idx
            )
        with c_cfg2:
            target_q_count = st.number_input("Total Questions", min_value=2, max_value=50, value=10)
        with c_cfg3:
            target_duration = st.number_input("Target Duration (minutes)", min_value=10, max_value=180, value=60)

        # Innovative Workflow #2: Adaptive Blueprint Difficulty Mix
        st.markdown("#### 🎚️ Seniority-Adaptive Difficulty Allocation")
        recommended_mix = core.get_adaptive_difficulty_mix(selected_seniority)
        
        c_d1, c_d2, c_d3, c_d4 = st.columns([1, 1, 1, 1])
        with c_d1:
            easy_val = st.number_input("Easy %", min_value=0, max_value=100, value=recommended_mix["Easy"])
        with c_d2:
            med_val = st.number_input("Medium %", min_value=0, max_value=100, value=recommended_mix["Medium"])
        with c_d3:
            hard_val = st.number_input("Hard %", min_value=0, max_value=100, value=recommended_mix["Hard"])
        with c_d4:
            st.write("")
            st.write("")
            if st.button("Reset to Adaptive Seniority Mix"):
                st.info(f"Recommended for {selected_seniority}: Easy {recommended_mix['Easy']}%, Med {recommended_mix['Medium']}%, Hard {recommended_mix['Hard']}%")

        if st.button("⚡ Generate Assessment Blueprint", type="primary", use_container_width=True):
            blueprint, warnings, engine_used = client_build_blueprint(
                st.session_state.competencies,
                selected_seniority,
                int(target_q_count),
                int(target_duration),
                int(easy_val),
                int(med_val),
                int(hard_val)
            )
            st.session_state.blueprint = blueprint
            st.session_state.warnings = warnings
            log_audit(f"Generated blueprint via {engine_used} ({target_q_count} questions for {selected_seniority} role).")

        if st.session_state.warnings:
            for w in st.session_state.warnings:
                if "must equal 100%" in w or "No competencies" in w:
                    st.error(f"⚠️ {w}")
                else:
                    st.warning(f"⚠️ {w}")

        if st.session_state.blueprint:
            st.divider()
            st.markdown("### 📋 Generated Assessment Blueprint Preview")
            
            preview_rows = []
            for b in st.session_state.blueprint:
                preview_rows.append({
                    "Competency": b["competency"],
                    "Category": b["category"],
                    "Mandatory": b["mandatory"],
                    "Weight (%)": b["weight_pct"],
                    "Allocated Questions": b["allocated_questions"],
                    "Questions Retrieved": len(b["selected_questions"]),
                    "Coverage Status": "Full" if len(b["selected_questions"]) >= b["allocated_questions"] else "Gap Detected"
                })
            
            st.dataframe(preview_rows, use_container_width=True, hide_index=True)

            # AI Explainability & Bias Guardrail Audit Report
            st.divider()
            st.markdown("### 🛡️ AI Explainability & Bias Guardrail Audit Report")
            exp_report = core.generate_explainability_report(
                jd_text=st.session_state.jd_text,
                competencies=st.session_state.competencies,
                blueprint=st.session_state.blueprint,
                seniority=selected_seniority,
                duration=int(target_duration),
            )
            
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                st.metric("Fairness Score", f"{exp_report['fairness_score']}%", delta="Passed" if exp_report['is_fairness_passed'] else "Warning")
            with c_f2:
                st.metric("Target Seniority", exp_report["seniority"])
            with c_f3:
                st.metric("Total Items", f"{exp_report['total_questions']} questions")
            with c_f4:
                st.metric("Total Time", f"{exp_report['total_minutes']} mins")

            with st.expander("📌 Seniority & Difficulty Allocation Rationale", expanded=True):
                st.markdown(f"**Target Role:** `{exp_report['seniority']}`")
                st.markdown(f"**Adaptive Rationale:** {exp_report['seniority_rationale']}")
                st.markdown(f"**Difficulty Mix:** Easy: `{exp_report['difficulty_mix']['Easy']}%` | Medium: `{exp_report['difficulty_mix']['Medium']}%` | Hard: `{exp_report['difficulty_mix']['Hard']}%`")

            with st.expander("🏷️ Competency Weighting & JD Evidence Rationale"):
                for cr in exp_report["competency_rationales"]:
                    st.markdown(f"**{cr['competency']}** (`{cr['category']}` • `{cr['weight_pct']}%` Weight)")
                    st.caption(cr["rationale"])

            with st.expander("⚖️ Bias & Fairness Guardrail Scan"):
                if exp_report["bias_guardrails"]:
                    for bg in exp_report["bias_guardrails"]:
                        st.error(f"**[{bg['severity']} Severity] {bg['category']}**: {bg['issue']}")
                        st.info(f"💡 Recommendation: {bg['recommendation']}")
                else:
                    st.success("✅ Clean Scan: No exclusionary terminology, age-restrictive phrasing, or degree inflation detected in the job description.")

            export_data = {
                "role": selected_seniority,
                "duration_minutes": target_duration,
                "explainability_report": exp_report,
                "blueprint": st.session_state.blueprint,
                "warnings": st.session_state.warnings,
            }
            st.download_button(
                "📥 Download Blueprint Specification (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name="assessment_blueprint.json",
                mime="application/json"
            )


# ---------------------------------------------------------
# TAB 4: Questions & AI Gap Generator
# ---------------------------------------------------------
with tab4:
    st.subheader("Stage 4: Question Retrieval, Tagging & Gap Synthesis Agent")
    st.caption("Inspect retrieved questions with full multi-tagging (Subskill, Cognitive Level, Job Relevance). Fill gaps with AI scenario questions.")

    if not st.session_state.blueprint:
        st.info("Generate a blueprint in Tab 3 to view questions.")
    else:
        st.markdown("### 📚 Assigned Assessment Questions")
        total_time = 0
        for b in st.session_state.blueprint:
            with st.expander(f"🧩 {b['competency']} ({b['weight_pct']}% Weight • {len(b['selected_questions'])}/{b['allocated_questions']} Questions)"):
                if not b["selected_questions"]:
                    st.error("No suitable question found in local bank! Use the AI Gap Generator below.")
                for i, q in enumerate(b["selected_questions"], 1):
                    total_time += q.get("minutes", 5)
                    st.markdown(f"**{i}. {q['question']}**")
                    st.caption(
                        f"🏷️ **Subskill:** {q.get('subskill', 'General')} | "
                        f"📊 **Difficulty:** {q.get('difficulty')} | "
                        f"🧠 **Cognitive Level:** {q.get('cognitive_level', 'Apply')} | "
                        f"⏱️ **Time:** {q.get('minutes')} mins | "
                        f"🎯 **Relevance:** {q.get('job_relevance', 'Direct')}"
                    )
                    st.markdown(f"*Expected Answer:* {q.get('expected', '')}")
                    st.markdown(f"*Scoring Rubric:* {q.get('rubric', 'Standard 10-point scale')}")
                    st.divider()

        st.markdown(f"**Total Estimated Assessment Time:** `{total_time}` minutes")

        # Gap Synthesis Module
        st.divider()
        st.markdown("### 💡 Fill Coverage Gaps: Scenario Question Generator")
        c_gap1, c_gap2 = st.columns(2)
        with c_gap1:
            all_comp_names = [b["competency"] for b in st.session_state.blueprint]
            gap_comp = st.selectbox("Select Competency to Generate Question For", all_comp_names)
        with c_gap2:
            gap_diff = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=1)

        if st.button("✨ Synthesize Scenario Question", type="primary", use_container_width=True):
            with st.spinner("Synthesizing scenario question..."):
                try:
                    new_q = client_generate_question(gap_comp, st.session_state.jd_text, gap_diff, api_key_input, model_choice, provider=provider_code)
                    st.session_state.generated_questions.append(new_q)
                    log_audit(f"Generated new draft question for {gap_comp} ({gap_diff}).")
                    st.success("Draft question created. Human review is required before publication.")
                except Exception as ex:
                    st.error(f"Generation failed: {ex}")

        if st.session_state.generated_questions:
            st.markdown("#### 📝 Pending Draft Questions for Human Review")
            for idx, gq in enumerate(st.session_state.generated_questions):
                with st.expander(f"Draft #{idx+1}: {gq.get('competency')} ({gq.get('difficulty')}) — {gq.get('review_status')}"):
                    st.write(gq.get("question"))
                    st.write("**Subskill:**", gq.get("subskill"))
                    st.write("**Expected Answer:**", gq.get("expected"))
                    st.write("**Rubric:**", gq.get("rubric"))
                    col_app1, col_app2 = st.columns(2)
                    with col_app1:
                        if st.button(f"✅ Approve & Add to Blueprint #{idx+1}", key=f"app_{idx}"):
                            for b in st.session_state.blueprint:
                                if b["competency"] == gq["competency"]:
                                    b["selected_questions"].append(gq)
                                    break
                            gq["review_status"] = "Approved by Reviewer"
                            log_audit(f"Approved draft question #{idx+1} for {gq['competency']}.")
                            st.rerun()

        # Innovative Workflow #4: Semantic Tagging & QA Audit Tool
        st.divider()
        st.markdown("### 🔬 Innovative Workflow 4: RAG Semantic Tagging & QA Audit Tool")
        st.caption("Audit assigned questions against the 15-competency vector taxonomy using embeddings to verify classification accuracy and detect wrong tags.")
        
        all_assigned = [q for b in st.session_state.blueprint for q in b["selected_questions"]]
        if all_assigned:
            q_map = {f"[{q.get('id', 'Q')}] {q['competency']} ({q.get('difficulty')}): {q['question'][:70]}...": q for q in all_assigned}
            chosen_q_key = st.selectbox("Select Question to Audit with Vector Taxonomy", list(q_map.keys()))
            target_q = q_map[chosen_q_key]

            if st.button("🧪 Run RAG Vector Tag Audit", type="secondary"):
                rag_engine = get_cached_rag()
                if rag_engine:
                    audit_res = rag_engine.detect_wrong_tags(target_q["question"], target_q["competency"])
                    tag_recs = rag_engine.recommend_tags(target_q["question"], top_k=3)
                    
                    c_aud1, c_aud2 = st.columns(2)
                    with c_aud1:
                        st.markdown("**Assigned Competency Tag:**")
                        st.info(f"🏷️ `{target_q['competency']}`")
                        if audit_res.get("is_wrong"):
                            st.warning(f"⚠️ Tag Mismatch Detected! Recommended: `{audit_res.get('suggested_competency')}` (Confidence: {int(audit_res.get('confidence', 0)*100)}%)")
                        else:
                            st.success("✅ Tag Validated: Question content closely matches assigned canonical competency.")
                    with c_aud2:
                        st.markdown("**Top Recommended Vector Taxonomy Tags:**")
                        for tr in tag_recs:
                            st.markdown(f"- **{tr['competency']}** (`{tr['category']}`): `{int(tr['confidence']*100)}% Match`")
                else:
                    st.info("RAG engine is operating in fallback mode.")


# ---------------------------------------------------------
# TAB 5: Candidate Simulation & Quality Review Agent
# ---------------------------------------------------------
with tab5:
    st.subheader("Stage 5: Quality Review, Coverage Heatmap & Candidate Simulation Agent")
    st.caption("Validate duration constraints, coverage heatmaps, and run synthetic candidate profiles to verify proficiency separation.")

    if not st.session_state.blueprint:
        st.info("Generate a blueprint in Tab 3 to review assessment quality.")
    else:
        # Innovative Workflow #3: Visual Coverage Heatmap
        st.markdown("### 🌡️ Innovative Workflow 3: Competency Coverage Heatmap")
        
        heatmap_cols = st.columns(len(st.session_state.blueprint))
        for col, b in zip(heatmap_cols, st.session_state.blueprint):
            found_count = len(b["selected_questions"])
            target_count = b["allocated_questions"]
            
            if found_count == 0:
                status_color = "#fecaca"  # Red (Missing)
                status_label = "Missing"
            elif found_count < target_count:
                status_color = "#fef08a"  # Amber (Under-assessed)
                status_label = "Under-assessed"
            else:
                status_color = "#bbf7d0"  # Green (Assessed)
                status_label = "Assessed"

            with col:
                st.markdown(
                    f"""
                    <div style="background-color: {status_color}; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid #cbd5e1; min-height: 110px;">
                        <div style="font-weight: 700; font-size: 0.85rem;">{b['competency']}</div>
                        <div style="font-size: 1.2rem; font-weight: 800; margin: 5px 0;">{found_count}/{target_count}</div>
                        <span style="font-size: 0.75rem; font-weight: 600;">{status_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Innovative Workflow #5: Synthetic Candidate Simulation
        st.divider()
        st.markdown("### 🧪 Innovative Workflow 5: Synthetic Candidate Simulation Engine")
        st.caption("Simulates real-world testing dynamics across distinct proficiency tiers to verify duration and scoring spread.")

        sim_button = st.button("🎲 Run Candidate Simulation", type="primary")
        if sim_button or "sim_data" in st.session_state:
            if sim_button:
                sim_res = client_simulate_candidates(st.session_state.blueprint, duration=60)
                st.session_state.sim_data = sim_res
                log_audit("Ran synthetic candidate simulation across 3 candidate profiles.")
            
            sim_data = st.session_state.get("sim_data")
            if sim_data:
                res_df = pd.DataFrame(sim_data["results"])
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                c_idx1, c_idx2 = st.columns(2)
                with c_idx1:
                    st.metric("Proficiency Separation Index", f"{sim_data['separation_index']}%")
                with c_idx2:
                    if sim_data["is_discriminating"]:
                        st.success("✅ Assessment Quality Passed: Effective score discrimination between Senior and Junior tiers (>35% spread).")
                    else:
                        st.warning("⚠️ Insufficient Separation: The difficulty mix does not adequately differentiate senior and junior candidates.")

        # Explainability & Audit Log
        st.divider()
        st.markdown("### 🔍 Explainability Report & Audit Log")
        
        with st.expander("View Job Description Evidence Traces"):
            for b in st.session_state.blueprint:
                st.markdown(f"**{b['competency']} (Weight: {b['weight_pct']}%)**")
                if b["evidence"]:
                    for ev in b["evidence"]:
                        st.markdown(f"- *\"{ev}\"*")
                else:
                    st.caption("Inferred from role title and related technology stack.")
        
        with st.expander("View Agent Orchestration Audit Logs"):
            if st.session_state.audit_logs:
                st.dataframe(pd.DataFrame(st.session_state.audit_logs), use_container_width=True, hide_index=True)
            else:
                st.write("No audit events recorded yet.")

        with st.expander("🔄 View Human-in-the-Loop Reviewer Feedback History"):
            fb_records = core.load_reviewer_feedback()
            if fb_records:
                st.dataframe(pd.DataFrame(fb_records), use_container_width=True, hide_index=True)
            else:
                st.write("No reviewer corrections saved yet. Use Tab 2 to submit feedback corrections.")

        # Final Assessment Export
        full_assessment_export = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app": APP_TITLE,
                "backend": API_BASE_URL,
            },
            "blueprint": st.session_state.blueprint,
            "audit_logs": st.session_state.audit_logs,
        }
        st.download_button(
            "📦 Export Complete Assessment Package (JSON)",
            data=json.dumps(full_assessment_export, indent=2),
            file_name="complete_assessment_package.json",
            mime="application/json"
        )

st.divider()
st.caption("AI Job Description–Driven Assessment Blueprint Builder • Enterprise AI Blueprint Architecture")


