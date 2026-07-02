import streamlit as st
from journal_model import load_model, predict_entry
from decision_logic import decide_next_step, score_phq9, score_gad7

st.set_page_config(
    page_title="MindJournal · AI Support",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_SOURCE = "paras9o9/journal-distilbert-model"


@st.cache_resource
def get_model():
    tokenizer, model = load_model(
        model_path=MODEL_SOURCE,
        tokenizer_name=MODEL_SOURCE,
    )
    return tokenizer, model


tokenizer, model = get_model()

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "analysis_done": False,
    "pred_label": None,
    "prob_dict": None,
    "decision": None,
    "user_text": "",
    "q_submitted": False,
    "phq_total": None,
    "phq_severity": None,
    "gad_total": None,
    "gad_severity": None,
    "final_decision": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_app():
    # For widget-bound keys, DELETE instead of setting — Streamlit resets them on next run
    widget_keys = {"user_text"}
    
    for k, v in DEFAULTS.items():
        if k in widget_keys:
            if k in st.session_state:
                del st.session_state[k]   # ← delete widget keys, don't assign to them
        else:
            st.session_state[k] = v       # ← normal keys are fine to assign
    st.rerun()


# ── Helpers ────────────────────────────────────────────────────────────────────
BADGE_MAP = {
    "NEU": ("Neutral", "#16a34a", "#f0fdf4", "#166534", "🟢"),
    "HUMOR": ("Humor / Positive", "#2563eb", "#eff6ff", "#1e40af", "😄"),
    "MH": ("Mental health concern", "#d97706", "#fef3c7", "#92400e", "🟡"),
    "SI": ("Suicidal ideation signal", "#dc2626", "#fef2f2", "#991b1b", "🔴"),
}

LABEL_ORDER = [
    ("NEU", "Neutral", "#16a34a"),
    ("HUMOR", "Humor / Positive", "#2563eb"),
    ("MH", "Mental health concern", "#d97706"),
    ("SI", "Suicidal ideation signal", "#dc2626"),
]


def get_badge(label):
    return BADGE_MAP.get(label, (label, "#6366f1", "#eef2ff", "#3730a3", "•"))


def get_support_message(decision):
    rec = decision.get("recommendation", "general_positive_feedback")
    if decision.get("showcrisiscard") or decision.get("show_crisis_card"):
        return (
            "crisis",
            "This entry may reflect significant distress. If you are in immediate "
            "danger, contact emergency services or iCall India: 9152987821",
        )
    if rec == "seek_professional_help":
        return ("warn", "Speaking with a mental health professional would be a helpful next step.")
    if rec == "consider_professional_help":
        return ("info", "Talking to a counsellor, therapist, or trusted person may be beneficial.")
    if rec in ("monitorandselfcare", "monitor_and_self_care"):
        return (
            "soft",
            "You may be going through a difficult period. Rest, self-care, and checking in "
            "with someone you trust can help.",
        )
    return ("ok", "No high-concern signal detected. Your feelings still matter — keep journalling.")


# ── Styles ────────────────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #f8fafc; min-height: 100vh; }
.block-container {
    max-width: 700px !important;
    padding: 1.75rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* ── Top bar ── */
.mj-topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.mj-logo { display: flex; align-items: center; gap: 10px; }
.mj-logo-icon {
    width: 34px; height: 34px; background: #4f46e5;
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 15px;
}
.mj-logo-name { font-size: 15px; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
.mj-logo-name em { color: #4f46e5; font-style: normal; }
.mj-emergency-chip {
    font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 999px;
    background: #fff7ed; color: #9a3412;
    border: 1px solid #fed7aa; letter-spacing: 0.01em;
}

/* ── Step tabs ── */
.mj-tabs {
    display: flex; gap: 4px; background: #f1f5f9;
    border-radius: 11px; padding: 3px; margin-bottom: 1.5rem;
}
.mj-tab {
    flex: 1; text-align: center; padding: 7px 4px;
    border-radius: 8px; font-size: 12px; font-weight: 600;
    color: #64748b;
}
.mj-tab.active {
    background: #fff; color: #4f46e5;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08), 0 0 0 0.5px #e2e8f0;
}

/* ── Cards ── */
.mj-card {
    background: #fff;
    border: 1px solid #e8edf3;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.mj-card-label {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #6366f1; margin-bottom: 6px;
}
.mj-card-title {
    font-size: 17px; font-weight: 700;
    color: #0f172a; margin-bottom: 5px; letter-spacing: -0.01em;
}
.mj-card-body {
    font-size: 14px; color: #64748b; line-height: 1.6; margin: 0;
}

/* ── Word count ── */
.mj-wc {
    display: flex; align-items: center; gap: 7px;
    font-size: 12px; color: #94a3b8; margin: 8px 0 14px; font-weight: 500;
}

/* ── Textarea ── */
div[data-testid="stTextArea"] { margin-top: 10px; }
div[data-testid="stTextArea"] textarea {
    border-radius: 11px !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    caret-color: #4f46e5 !important;
    padding: 13px 14px !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    resize: vertical !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important; font-style: italic !important;
}

/* ── Buttons ── */
# div[data-testid="stButton"] > button,
# div[data-testid="stFormSubmitButton"] > button {
#     border-radius: 10px !important;
#     min-height: 42px !important;
#     font-size: 13.5px !important;
#     font-weight: 700 !important;
#     font-family: 'Inter', system-ui, sans-serif !important;
#     letter-spacing: 0.01em !important;
#     transition: all 0.14s ease !important;
#     cursor: pointer !important;
# }
# div[data-testid="stBaseButton-primary"] > button,
# div[data-testid="stFormSubmitButton"] > button {
#     background: #4f46e5 !important;
#     color: #fff !important;
#     border: none !important;
#     box-shadow: 0 1px 2px rgba(79,70,229,0.25) !important;
# }
# div[data-testid="stBaseButton-primary"] > button:hover,
# div[data-testid="stFormSubmitButton"] > button:hover {
#     background: #4338ca !important;
#     box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
#     transform: translateY(-1px) !important;
# }
# div[data-testid="stBaseButton-secondary"] > button {
#     background: #fff !important;
#     color: #475569 !important;
#     border: 1.5px solid #e2e8f0 !important;
#     box-shadow: 0 1px 2px rgba(15,23,42,0.05) !important;
# }
# div[data-testid="stBaseButton-secondary"] > button:hover {
#     background: #43ff64d9 !important;
#     border-color: #cbd5e1 !important;
#     color: #0f172a !important;
# }

/* ── Result badge ── */
# .mj-badge {
#     display: inline-flex; align-items: center; gap: 8px;
#     padding: 6px 15px; border-radius: 999px;
#     font-size: 13px; font-weight: 700; margin-bottom: 14px;
# }

/* ── Metric pills ── */
# .mj-pills { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 18px; }
# .mj-pill {
#     background: #f8fafc; border: 1px solid #e2e8f0;
#     border-radius: 8px; padding: 6px 12px;
#     font-size: 12px; font-weight: 600; color: #334155;
# }
# .mj-pill-key { color: #94a3b8; font-weight: 500; margin-right: 4px; }

/* ── Confidence bars ── */
# .mj-conf-head {
#     font-size: 11px; font-weight: 700; text-transform: uppercase;
#     letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 11px;
# }
# .mj-conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
# .mj-conf-name {
#     font-size: 12.5px; font-weight: 600; color: #334155;
#     width: 175px; flex-shrink: 0;
# }
# .mj-conf-track {
#     flex: 1; height: 7px; background: #f1f5f9;
#     border-radius: 99px; overflow: hidden;
# }
# .mj-conf-fill { height: 100%; border-radius: 99px; }
# .mj-conf-pct { font-size: 12px; font-weight: 700; color: #64748b; width: 34px; text-align: right; }

/* ── Message boxes ── */
.mj-msg {
    border-radius: 11px; padding: 13px 15px;
    font-size: 13.5px; line-height: 1.6; font-weight: 500;
    display: flex; gap: 11px; align-items: flex-start; margin-top: 2px;
}
.mj-msg-icon { font-size: 16px; margin-top: 1px; flex-shrink: 0; }
.mj-crisis { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
.mj-warn   { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.mj-info   { background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; }
.mj-soft   { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.mj-ok     { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

/* ── Questionnaire results ── */
.mj-q-sec-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6366f1;
    padding-bottom: 8px; border-bottom: 1px solid #f1f5f9; margin-bottom: 12px;
}
.mj-q-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }
.mj-q-pill { border-radius: 9px; padding: 7px 13px; font-size: 13px; font-weight: 700; }
.mj-q-min  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.mj-q-mild { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.mj-q-mod  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.mj-q-sev  { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Expander ── */
div[data-testid="stExpander"] details {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] summary {
    font-weight: 700 !important; 
    color: black;
    font-size: 14px !important; 
    padding: 14px 16px !important;
    transition: background 0.12s !important;
}

div[data-testid="stExpander"] summary:hover { background: #fafafa !important;}
# div[data-testid="stExpander"] details > div { padding: 0 16px 16px !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 9px !important;
    border-color: #e2e8f0 !important;
    background: #f8fafc !important;
    font-size: 13.5px !important;

.st-emotion-cache-1weic72 {
color: black;
}

# /* ── Footer ── */
# .mj-footer {
#     text-align: center; font-size: 12px; color: #94a3b8;
#     padding-top: 1.25rem; border-top: 1px solid #e2e8f0;
#     margin-top: 2rem; line-height: 1.7;
# }
# .mj-footer strong { color: #64748b; }
# </style>
# """)

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-topbar">
  <div class="mj-logo">
    <div class="mj-logo-icon">🧠</div>
    <div class="mj-logo-name">Mind<em>Journal</em></div>
  </div>
  <div class="mj-emergency-chip">Not for emergency use</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Step tabs ──────────────────────────────────────────────────────────────────
if st.session_state.analysis_done and st.session_state.q_submitted:
    active = 2
elif st.session_state.analysis_done:
    active = 1
else:
    active = 0

tab_classes = ["", "", ""]
tab_classes[active] = " active"

st.markdown(
    f"""
<div class="mj-tabs">
  <div class="mj-tab{tab_classes[0]}">1 · Entry</div>
  <div class="mj-tab{tab_classes[1]}">2 · Analysis</div>
  <div class="mj-tab{tab_classes[2]}">3 · Guidance</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-card">
  <div class="mj-card-label">Step 1 of 3 · Your entry</div>
  <div class="mj-card-title">What's on your mind today?</div>
  <div class="mj-card-body">Write freely, your thoughts, your day, anything weighing on you. Processed locally.</div>
</div>
""",
    unsafe_allow_html=True,
)

user_text = st.text_area(
    "Journal entry",
    height=170,
    placeholder=(
        "e.g. I've been feeling emotionally drained lately. Some days I manage well, "
        "but other times everything feels heavier than usual…"
    ),
    key="user_text",
    label_visibility="collapsed",
)

word_count = len(user_text.split()) if user_text.strip() else 0

if word_count == 0:
    wc_color, wc_dot, wc_hint = "#94a3b8", "#cbd5e1", "Try writing at least 2–3 sentences for better feedback."
elif word_count < 10:
    wc_color, wc_dot, wc_hint = "#d97706", "#d97706", "A bit short — a few more words helps the model."
elif word_count <= 150:
    wc_color, wc_dot, wc_hint = "#16a34a", "#16a34a", "Good length."
else:
    wc_color, wc_dot, wc_hint = "#d97706", "#d97706", "Long entry — the model reads up to ~512 tokens."

st.markdown(
    f'<div class="mj-wc">'
    f'<span style="width:7px;height:7px;border-radius:50%;background:{wc_dot};display:inline-block;flex-shrink:0"></span>'
    f'<span style="color:{wc_color};font-weight:700">{word_count} words</span>'
    f'<span style="color:#e2e8f0">·</span>'
    f'<span>{wc_hint}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

col_analyse, col_reset = st.columns([5, 1])
with col_analyse:
    analyse = st.button("Analyse my entry", type="primary", use_container_width=True)
with col_reset:
    if st.button("Reset", use_container_width=True):
        reset_app()

# ── Run analysis ───────────────────────────────────────────────────────────────
if analyse:
    if not user_text.strip():
        st.warning("Please write something before analysing.")
    elif word_count < 3:
        st.warning("Your entry is very short. Write a few more words for useful feedback.")
    else:
        with st.spinner("Analysing your entry…"):
            pred_label, prob_dict = predict_entry(user_text, tokenizer, model)
            decision = decide_next_step(pred_label, prob_dict)
        st.session_state.update(
            analysis_done=True,
            pred_label=pred_label,
            prob_dict=prob_dict,
            decision=decision,
            q_submitted=False,
            phq_total=None,
            phq_severity=None,
            gad_total=None,
            gad_severity=None,
            final_decision=None,
        )

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    pred_label = st.session_state.pred_label
    prob_dict = st.session_state.prob_dict
    decision = st.session_state.decision

    _, label_color, label_bg, label_text_color, label_icon = get_badge(pred_label)
    label_name = BADGE_MAP.get(pred_label, (pred_label,))[0]
    msg_type, msg_text = get_support_message(decision)
    risk_tier = decision.get("risk_tier", "low").capitalize()

    conf_bars_html = ""
    for code, name, bar_color in LABEL_ORDER:
        p = float(prob_dict.get(code, 0.0))
        conf_bars_html += (
            f'<div class="mj-conf-row">'
            f'  <div class="mj-conf-name">{name}</div>'
            f'  <div class="mj-conf-track">'
            f'    <div class="mj-conf-fill" style="width:{p*100:.1f}%;background:{bar_color}"></div>'
            f'  </div>'
            f'  <div class="mj-conf-pct">{p:.0%}</div>'
            f'</div>'
        )

    st.markdown(
        f"""
<div class="mj-card">
  <div class="mj-card-label">Step 2 of 3 · Analysis result</div>
  <div class="mj-card-title">Model interpretation</div>

  <div class="mj-badge"
       style="background:{label_bg};color:{label_text_color};border:1px solid {label_color}55">
    {label_icon}&nbsp; {label_name}
  </div>

  <div class="mj-pills">
    <div class="mj-pill"><span class="mj-pill-key">Risk tier</span>{risk_tier}</div>
    <div class="mj-pill"><span class="mj-pill-key">Top signal</span>{pred_label}</div>
  </div>

  <div class="mj-conf-head">Confidence across classes</div>
  {conf_bars_html}
</div>
""",
        unsafe_allow_html=True,
    )

    MSG_ICONS = {"crisis": "⛑️", "warn": "⚠️", "info": "💬", "soft": "🌿", "ok": "✅"}
    MSG_CLASSES = {
        "crisis": "mj-crisis",
        "warn": "mj-warn",
        "info": "mj-info",
        "soft": "mj-soft",
        "ok": "mj-ok",
    }
    icon = MSG_ICONS.get(msg_type, "•")
    cls = MSG_CLASSES.get(msg_type, "mj-ok")

    st.markdown(
        f"""
<div class="mj-card">
  <div class="mj-card-label">Step 3 of 3 · Guidance</div>
  <div class="mj-card-title">Suggested next step</div>
  <div class="mj-msg {cls}">
    <span class="mj-msg-icon">{icon}</span>
    <span>{msg_text}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    show_prompt = decision.get("showphqgadprompt") or decision.get("show_phq_gad_prompt")
    if show_prompt:
        with st.expander("📋  Optional: PHQ-9 and GAD-7 check-in", expanded=False):
            st.markdown(
                '<p style="font-size:13.5px;color:#64748b;margin:4px 0 16px;line-height:1.6">'
                "These brief validated questionnaires give a more structured check-in "
                "on mood and anxiety.</p>",
                unsafe_allow_html=True,
            )

            RESP = {
                0: "Not at all",
                1: "Several days",
                2: "More than half the days",
                3: "Nearly every day",
            }
            PHQ_QS = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep, or sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself, or feeling like a failure",
                "Trouble concentrating on things",
                "Moving or speaking unusually slowly or being unusually fidgety",
                "Thoughts that you would be better off dead, or of hurting yourself",
            ]
            GAD_QS = [
                "Feeling nervous, anxious, or on edge",
                "Not being able to stop or control worrying",
                "Worrying too much about different things",
                "Trouble relaxing",
                "Being so restless that it is hard to sit still",
                "Becoming easily annoyed or irritable",
                "Feeling afraid as if something awful might happen",
            ]

            with st.form("questionnaire_form"):
                st.markdown(
                    '<div class="mj-q-sec-label">PHQ-9 · Depression screen</div>',
                    unsafe_allow_html=True,
                )
                phq_answers = [
                    st.selectbox(
                        f"{i + 1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: RESP[x],
                        key=f"phq_{i}",
                    )
                    for i, q in enumerate(PHQ_QS)
                ]
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="mj-q-sec-label">GAD-7 · Anxiety screen</div>',
                    unsafe_allow_html=True,
                )
                gad_answers = [
                    st.selectbox(
                        f"{i + 1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: RESP[x],
                        key=f"gad_{i}",
                    )
                    for i, q in enumerate(GAD_QS)
                ]
                submitted = st.form_submit_button(
                    "Submit questionnaires", use_container_width=True
                )

            if submitted:
                phq_total, phq_severity = score_phq9(phq_answers)
                gad_total, gad_severity = score_gad7(gad_answers)
                final_decision = decide_next_step(
                    pred_label,
                    prob_dict,
                    phq9_items=phq_answers,
                    gad7_items=gad_answers,
                )
                st.session_state.update(
                    q_submitted=True,
                    phq_total=phq_total,
                    phq_severity=phq_severity,
                    gad_total=gad_total,
                    gad_severity=gad_severity,
                    final_decision=final_decision,
                )

            if st.session_state.q_submitted:
                fd = st.session_state.final_decision
                ph_sev = (st.session_state.phq_severity or "").lower()
                ga_sev = (st.session_state.gad_severity or "").lower()

                def sev_class(s):
                    if "severe" in s:
                        return "mj-q-sev"
                    if "moderate" in s:
                        return "mj-q-mod"
                    if "mild" in s:
                        return "mj-q-mild"
                    return "mj-q-min"

                st.markdown(
                    f"""
<p style="font-size:14px;font-weight:700;color:#0f172a;margin:16px 0 4px">
  Questionnaire results
</p>
<div class="mj-q-pills">
  <div class="mj-q-pill {sev_class(ph_sev)}">
    PHQ-9: {st.session_state.phq_total} &nbsp;·&nbsp; {ph_sev.capitalize()}
  </div>
  <div class="mj-q-pill {sev_class(ga_sev)}">
    GAD-7: {st.session_state.gad_total} &nbsp;·&nbsp; {ga_sev.capitalize()}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                if fd.get("suicidalityflag") or fd.get("suicidality_flag"):
                    st.error(
                        "You indicated possible self-harm thoughts on PHQ-9 item 9. "
                        "Please seek immediate support. iCall India: **9152987821**"
                    )
                elif fd.get("recommendation") == "seek_professional_help":
                    st.warning(
                        "These results suggest professional mental health support would be advisable."
                    )
                elif fd.get("recommendation") == "consider_professional_help":
                    st.info(
                        "These results suggest that talking to a counsellor or therapist may help."
                    )
                else:
                    st.success(
                        "Thank you for completing the check-in. Keep taking care of yourself."
                    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-footer">
  Research prototype &nbsp;·&nbsp; Not a clinical tool &nbsp;·&nbsp; Not for emergency use<br>
  If you are in crisis, contact <strong>iCall India: 9152987821</strong> or local emergency services.
</div>
""",
    unsafe_allow_html=True,
)
