"""
streamlit_app.py

Professional medical dashboard GUI for the Intelligent Healthcare
Diagnostic Assistant. Designed to resemble a clinical EMR interface
rather than an AI/tech demo.

DESIGN NOTE: this file contains NO diagnostic logic of its own -- it
imports build_system(), process_patient(), get_test_patients(), and
build_nlp_patient() directly from app.py, and reuses the exact same
pipeline the CLI uses. The CLI (app.py) remains the manual-aligned
source of truth; this file is purely a presentation layer on top of it.

Run with:
    streamlit run streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from app import (
    build_system, process_patient, load_patient_records,
    get_test_patients, build_nlp_patient,
)
from modules.agent import PatientPercept
from modules.nlp_processor import extract_symptoms_from_text

st.set_page_config(
    page_title="MedAssist — Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# NLTK DATA (best-effort, NON-blocking)
#
# nlp_processor.extract_symptoms_from_text() already falls back to a plain
# regex tokenizer when punkt isn't available (see modules/nlp_processor.py,
# _tokenize()). Downloading punkt can hit a slow network / SSL failure on a
# fresh Streamlit Cloud instance, so it must never block first paint -- run
# it in a daemon thread instead. If the download finishes, later reruns get
# full NLTK tokenization; if not, the regex fallback is used.
# ---------------------------------------------------------------------------
try:
    import nltk
    import threading

    def _download_nltk_data():
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

    threading.Thread(target=_download_nltk_data, daemon=True).start()
except Exception:
    pass

# ---------------------------------------------------------------------------
# DESIGN TOKENS — professional clinical palette
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_raised": "#F8FAFC",
    "ink": "#111827",
    "muted": "#6B7280",
    "accent": "#1A56DB",
    "accent_light": "#E8EEF8",
    "critical": "#DC2626",
    "critical_bg": "#FEF2F2",
    "high": "#D97706",
    "high_bg": "#FFFBEB",
    "medium": "#2563EB",
    "medium_bg": "#EFF6FF",
    "low": "#059669",
    "low_bg": "#ECFDF5",
    "border": "#E5E7EB",
}

URGENCY_COLORS = {
    "CRITICAL": PALETTE["critical"],
    "HIGH": PALETTE["high"],
    "MEDIUM": PALETTE["medium"],
    "LOW": PALETTE["low"],
}

URGENCY_BG = {
    "CRITICAL": PALETTE["critical_bg"],
    "HIGH": PALETTE["high_bg"],
    "MEDIUM": PALETTE["medium_bg"],
    "LOW": PALETTE["low_bg"],
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background: {PALETTE['surface']};
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: {PALETTE['surface_raised']};
    border-right: 1px solid {PALETTE['border']};
}}

section[data-testid="stSidebar"] .sidebar-logo {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {PALETTE['accent']};
    padding: 0.5rem 0 0.25rem 0;
    letter-spacing: -0.01em;
}}

section[data-testid="stSidebar"] .sidebar-sub {{
    font-size: 0.7rem;
    color: {PALETTE['muted']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {PALETTE['border']};
}}

section[data-testid="stSidebar"] .nav-label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {PALETTE['muted']};
    padding: 1.5rem 0 0.3rem 0;
}}

/* Custom sidebar nav buttons */
div[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    transition: background 0.15s;
    font-size: 0.88rem;
}}

div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: {PALETTE['accent_light']};
}}

/* ---- Top header bar ---- */
.top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 0.75rem 0;
    border-bottom: 1px solid {PALETTE['border']};
    margin-bottom: 1.25rem;
}}

.top-bar-left {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

.top-bar-icon {{
    width: 34px;
    height: 34px;
    background: {PALETTE['accent']};
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 1.1rem;
    font-weight: 700;
}}

.top-bar-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {PALETTE['ink']};
    margin: 0;
}}

.top-bar-sub {{
    font-size: 0.72rem;
    color: {PALETTE['muted']};
    margin: 0;
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 500;
    color: {PALETTE['low']};
    background: {PALETTE['low_bg']};
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
}}

.status-badge::before {{
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {PALETTE['low']};
}}

/* ---- Section headers ---- */
.section-header {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {PALETTE['ink']};
    margin: 0 0 1rem 0;
    padding: 0;
}}

.section-desc {{
    font-size: 0.8rem;
    color: {PALETTE['muted']};
    margin: -0.5rem 0 1rem 0;
}}

/* ---- Cards ---- */
.card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}

.card-header {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {PALETTE['muted']};
    margin-bottom: 0.5rem;
}}

/* ---- Vitals row ---- */
.vitals-row {{
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}}

.vital-item {{
    text-align: center;
    min-width: 90px;
}}

.vital-label {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {PALETTE['muted']};
}}

.vital-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: {PALETTE['ink']};
    line-height: 1.3;
}}

.vital-unit {{
    font-size: 0.7rem;
    color: {PALETTE['muted']};
}}

/* ---- Diagnosis result card ---- */
.dx-card {{
    border: 1px solid {PALETTE['border']};
    border-left: 4px solid var(--dx-color, {PALETTE['accent']});
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    background: {PALETTE['surface']};
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}

.dx-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {PALETTE['ink']};
    text-transform: capitalize;
    margin: 0.1rem 0;
}}

.dx-meta {{
    font-size: 0.85rem;
    color: {PALETTE['muted']};
    margin: 0.25rem 0 0.5rem 0;
}}

.urgency-tag {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}}

/* ---- Module rows ---- */
.module-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid {PALETTE['border']};
    font-size: 0.85rem;
}}

.module-row:last-child {{ border-bottom: none; }}

.module-name {{
    color: {PALETTE['muted']};
}}

.module-result {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: {PALETTE['ink']};
}}

/* ---- Treatment plan ---- */
.plan-step {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid {PALETTE['border']};
    font-size: 0.85rem;
}}

.plan-step:last-child {{ border-bottom: none; }}

.plan-step-num {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: {PALETTE['accent']};
    min-width: 1.6rem;
}}

.plan-step-duration {{
    margin-left: auto;
    font-size: 0.78rem;
    color: {PALETTE['muted']};
}}

/* ---- RL note ---- */
.rl-note {{
    font-size: 0.78rem;
    color: {PALETTE['muted']};
    border-left: 2px solid {PALETTE['accent_light']};
    padding-left: 0.6rem;
    margin-top: 0.5rem;
}}

/* ---- About section ---- */
.about-card {{
    background: {PALETTE['surface_raised']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

.about-card h4 {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {PALETTE['ink']};
    margin: 0 0 0.25rem 0;
}}

.about-card p {{
    font-size: 0.82rem;
    color: {PALETTE['muted']};
    margin: 0 0 0.75rem 0;
}}

.about-card .tag {{
    display: inline-block;
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    background: {PALETTE['accent_light']};
    color: {PALETTE['accent']};
    font-weight: 500;
    margin: 0.15rem;
}}

/* ---- Footer ---- */
.app-footer {{
    text-align: center;
    font-size: 0.7rem;
    color: {PALETTE['muted']};
    border-top: 1px solid {PALETTE['border']};
    padding-top: 1rem;
    margin-top: 2rem;
}}

/* Hide Streamlit branding */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SYSTEM INITIALIZATION
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading diagnostic system (first run only)...")
def get_system():
    agent, fuzzy_assessor, planner, rl_agent = build_system()
    patient_history = load_patient_records()
    return agent, fuzzy_assessor, planner, rl_agent, patient_history


agent, fuzzy_assessor, planner, rl_agent, patient_history = get_system()


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">✦ MedAssist</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Clinical Decision Support</div>', unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["Patient Intake", "Test Suite", "Evaluation", "About"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("CCS 3101 · AI Capstone")
    st.caption("v1.0")


# ---------------------------------------------------------------------------
# RENDER HELPERS
# ---------------------------------------------------------------------------
def render_vitals(patient: PatientPercept):
    st.markdown(f"""
    <div class="card">
        <div class="card-header">Patient Vitals</div>
        <div class="vitals-row">
            <div class="vital-item">
                <div class="vital-label">Patient</div>
                <div class="vital-value">{patient.patient_id}</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">Age</div>
                <div class="vital-value">{patient.age}</div>
                <div class="vital-unit">years</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">Temperature</div>
                <div class="vital-value">{patient.temperature}&#176;</div>
                <div class="vital-unit">Celsius</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">Heart Rate</div>
                <div class="vital-value">{patient.heart_rate}</div>
                <div class="vital-unit">bpm</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">Blood Pressure</div>
                <div class="vital-value">{patient.blood_pressure}</div>
                <div class="vital-unit">mmHg</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_report(report: dict):
    urgency = report["urgency"]
    uc = URGENCY_COLORS.get(urgency, PALETTE["accent"])
    ubg = URGENCY_BG.get(urgency, PALETTE["accent_light"])

    st.markdown(f"""
    <div class="dx-card" style="--dx-color: {uc};">
        <div class="card-header">Aggregated Diagnosis</div>
        <div class="dx-title">{report['diagnosis'].replace('_', ' ')}</div>
        <div class="dx-meta">Confidence: {report['confidence']:.1%}</div>
        <span class="urgency-tag" style="background: {ubg}; color: {uc};">{urgency} URGENCY</span>
    </div>
    """, unsafe_allow_html=True)

    if report.get("excluded_symptoms"):
        st.warning(
            f"Unrecognized symptom(s) ignored: "
            f"{', '.join(report['excluded_symptoms'])}. "
            f"Diagnosis based only on recognized symptoms."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="card-header">Module Breakdown</div>', unsafe_allow_html=True)
        rows = ""
        for name, result in report["module_results"].items():
            dx = result.get("diagnosis", "-") or "-"
            conf = result.get("confidence")
            conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else "-"
            dx_clean = dx.replace("_suspected", "").replace("_confirmed", "").replace("_", " ")
            rows += (f'<div class="module-row"><span class="module-name">{name}</span>'
                     f'<span class="module-result">{dx_clean} ({conf_str})</span></div>')
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

        if report.get("severity_score") is not None:
            st.markdown(f'<div class="card-header">Fuzzy Severity</div>', unsafe_allow_html=True)
            st.progress(
                min(report["severity_score"] / 100, 1.0),
                text=f"{report['severity_label']} — {report['severity_score']}/100",
            )

        if report.get("rl_recommended_action"):
            match = report["rl_recommended_action"] == report["next_action"]
            status = "MATCH" if match else "DIFFERS"
            st.markdown(
                f'<div class="rl-note">RL Policy: {report["rl_recommended_action"]} '
                f'&nbsp;[{status} vs. rule-based]</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(f'<div class="card-header">Treatment Plan</div>', unsafe_allow_html=True)
        if report["treatment_plan"]:
            steps_html = ""
            for step in report["treatment_plan"]:
                steps_html += (f'<div class="plan-step">'
                                f'<span class="plan-step-num">{step["step"]:02d}</span>'
                                f'<span>{step["action"]}</span>'
                                f'<span class="plan-step-duration">{step["duration"]}</span>'
                                f'</div>')
            st.markdown(f'<div class="card">{steps_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No treatment plan generated for this case.")

        st.markdown(f'<div class="card-header">Recommendations</div>', unsafe_allow_html=True)
        recs = "".join(f"<div>• {rec}</div>" for rec in report["recommendations"])
        st.markdown(f'<div class="card" style="font-size:0.85rem;">{recs}</div>', unsafe_allow_html=True)

        if report.get("similar_cases"):
            st.markdown(f'<div class="card-header">Similar Historical Cases</div>', unsafe_allow_html=True)
            for c in report["similar_cases"]:
                st.caption(f"{c['patient_id']} — {c['diagnosis']} (similarity {c['similarity']:.0%})")


# ---------------------------------------------------------------------------
# TOP HEADER BAR
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <div class="top-bar-icon">✦</div>
        <div>
            <p class="top-bar-title">MedAssist</p>
            <p class="top-bar-sub">Clinical Decision Support &middot; Dedan Kimathi University</p>
        </div>
    </div>
    <div class="status-badge">System Ready</div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE: PATIENT INTAKE
# ===========================================================================
if nav == "Patient Intake":
    st.markdown(f'<div class="section-header">Patient Intake & Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">Describe symptoms and vitals to generate a clinical assessment.</div>', unsafe_allow_html=True)

    intake_col, _ = st.columns([2, 1])

    with intake_col:
        with st.container():
            raw_complaint = st.text_area(
                "Patient Complaint",
                placeholder="e.g. Fever and a bad cough for two days, whole body aches, chills.",
                height=90,
            )

            cols = st.columns(4)
            age = cols[0].number_input("Age", min_value=0, max_value=120, value=30)
            temperature = cols[1].number_input("Temp (C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
            heart_rate = cols[2].number_input("Heart Rate", min_value=30, max_value=250, value=80)
            blood_pressure = cols[3].text_input("BP", value="120/80")

            diagnose_btn = st.button("Run Diagnosis", type="primary", use_container_width=True)

    if diagnose_btn:
        if not raw_complaint.strip():
            st.error("Please describe at least one symptom.")
        else:
            symptoms = extract_symptoms_from_text(raw_complaint)
            if not symptoms:
                st.error("No recognizable symptoms found. Try different wording.")
            else:
                st.caption(f"Recognized: {', '.join(symptoms)}")
                patient = PatientPercept(
                    patient_id=f"LIVE-{len(patient_history) + 1:03d}",
                    symptoms=symptoms, age=age, temperature=temperature,
                    heart_rate=heart_rate, blood_pressure=blood_pressure,
                )
                with st.spinner("Running diagnostic pipeline..."):
                    report = process_patient(agent, fuzzy_assessor, planner,
                                              rl_agent, patient, patient_history)
                render_vitals(patient)
                render_report(report)

    if not diagnose_btn:
        st.info("Enter patient data above and click **Run Diagnosis** to begin.")


# ===========================================================================
# PAGE: TEST SUITE
# ===========================================================================
elif nav == "Test Suite":
    st.markdown(f'<div class="section-header">Built-in Test Suite</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">Run all 7 test patients through the full diagnostic pipeline.</div>', unsafe_allow_html=True)

    if st.button("Run All Tests", type="primary"):
        patients = get_test_patients()
        patients.append(None)

        for marker in patients:
            patient = build_nlp_patient() if marker is None else marker
            with st.spinner(f"Processing {patient.patient_id}..."):
                report = process_patient(agent, fuzzy_assessor, planner,
                                          rl_agent, patient, patient_history)
            with st.expander(f"**{patient.patient_id}** — {report['diagnosis'].replace('_', ' ')} ({report['urgency']})", expanded=False):
                render_vitals(patient)
                render_report(report)
    else:
        st.info("Click **Run All Tests** to evaluate all built-in patients.")


# ===========================================================================
# PAGE: EVALUATION
# ===========================================================================
elif nav == "Evaluation":
    st.markdown(f'<div class="section-header">Module Evaluation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">Compare each diagnostic module against 15 ground-truth patients.</div>', unsafe_allow_html=True)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating modules..."):
            from evaluation.metrics import load_seed_patients, evaluate_all_modules
            from evaluation.visualizations import generate_all_evaluation_plots

            seed_patients = load_seed_patients()
            if not seed_patients:
                st.error("No seed patients found. Ensure data/patient_records.csv exists.")
            else:
                disease_list = sorted(set(d for _, d in seed_patients))
                modules = {
                    "KnowledgeBase": agent._modules.get("KnowledgeBase"),
                    "BayesianNet": agent._modules.get("BayesianNet"),
                    "MLClassifier": agent._modules.get("MLClassifier"),
                    "NeuralNetwork": agent._modules.get("NeuralNetwork"),
                }
                results = evaluate_all_modules(modules, seed_patients, disease_list)

                cols = st.columns(4)
                for col, (name, r) in zip(cols, results.items()):
                    col.metric(name, f"{r['accuracy']:.1%}", f"F1: {r['f1_macro']:.2f}")

                generate_all_evaluation_plots(results)
                cm_col, bar_col = st.columns(2)
                cm_col.image("evaluation/confusion_matrices.png", caption="Confusion Matrices")
                bar_col.image("evaluation/module_comparison.png", caption="Module Comparison")
    else:
        st.info("Click **Run Evaluation** to assess module accuracy against ground truth.")


# ===========================================================================
# PAGE: ABOUT
# ===========================================================================
elif nav == "About":
    st.markdown(f'<div class="section-header">About MedAssist</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card">
        <h4>Intelligent Healthcare Diagnostic Assistant</h4>
        <p>An end-to-end AI system integrating intelligent agents, probabilistic reasoning, machine learning, deep learning, NLP, fuzzy logic, reinforcement learning, and automated planning into a unified clinical decision-support platform.</p>
    </div>

    <div class="about-card">
        <h4>AI Modules</h4>
        <p>
            <span class="tag">Intelligent Agent</span>
            <span class="tag">Knowledge Base & Logic</span>
            <span class="tag">Bayesian Network</span>
            <span class="tag">ML Classifier</span>
            <span class="tag">Neural Network</span>
            <span class="tag">Fuzzy Logic</span>
            <span class="tag">STRIPS Planning</span>
            <span class="tag">NLP Processing</span>
            <span class="tag">Search Algorithms</span>
            <span class="tag">Q-Learning (RL)</span>
        </p>
    </div>

    <div class="about-card">
        <h4>Diagnostic Coverage</h4>
        <p>15 diseases · 26 symptoms — All relationships are synthetic and medically plausible, not sourced from real clinical data.</p>
    </div>

    <div class="about-card">
        <h4>Course</h4>
        <p>CCS 3101 —  Artificial Intelligence · 13-Week Capstone · Dedan Kimathi University of Science and Technology</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="app-footer">
    MedAssist Clinical Decision Support &middot; CCS 3101 AI Capstone &middot; {__import__('datetime').datetime.now().year}
</div>
""", unsafe_allow_html=True)
