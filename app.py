"""
HealthShield AI — Health Insurance Policy Assistant
Streamlit front-end for the policy_assistant RAG pipeline.
"""

import os
import html
import time
import datetime as dt

import streamlit as st

from policy_assistant import config
from policy_assistant.pipeline import ask, build_health_assistant
from policy_assistant.vector_store import vector_store_exists
from policy_assistant.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="HealthShield AI · Policy Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BOT_AVATAR = "🛡️"

# --------------------------------------------------------------------------- #
# THEME / CSS
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0B1120;
    --surface:   #111C2E;
    --surface-2: #16233A;
    --border:    rgba(148, 163, 184, 0.14);
    --border-hi: rgba(45, 212, 191, 0.38);
    --text:      #E6EDF7;
    --muted:     #8FA3BF;
    --accent:    #2DD4BF;
    --accent-2:  #38BDF8;
    --good:      #34D399;
    --warn:      #FBBF24;
    --bad:       #F87171;
    --r-lg: 20px; --r-md: 14px; --r-sm: 9px;
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1100px 600px at 12% -10%, rgba(45, 212, 191, 0.10), transparent 60%),
        radial-gradient(900px 500px at 92% 0%, rgba(56, 189, 248, 0.09), transparent 55%),
        var(--bg);
    color: var(--text);
}
.block-container { padding-top: 1.6rem; padding-bottom: 5rem; max-width: 1080px; }
#MainMenu, footer { visibility: hidden; }
code { font-family: 'JetBrains Mono', monospace !important; }

/* Accessible focus for every interactive element */
button:focus-visible, textarea:focus-visible, a:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* =========================== HERO =========================== */
.hero {
    position: relative; overflow: hidden;
    border-radius: var(--r-lg);
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #0E1B2E 0%, #122A3E 45%, #0D3B3B 100%);
    border: 1px solid var(--border);
    box-shadow: 0 30px 70px -35px rgba(0, 0, 0, 0.95);
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 2.4rem;
    align-items: center;
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: radial-gradient(circle at 80% 30%, black, transparent 70%);
}
.hero > * { position: relative; z-index: 1; }

.hero-live {
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--accent); font-size: 0.78rem; font-weight: 600;
    margin-bottom: 1rem;
}
.dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
    animation: pulse 2.2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(45,212,191,0.55); }
    70%  { box-shadow: 0 0 0 9px rgba(45,212,191,0); }
    100% { box-shadow: 0 0 0 0 rgba(45,212,191,0); }
}
.hero-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.08; margin: 0;
    background: linear-gradient(100deg, #FFFFFF 20%, #9FF3E6 95%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 0.97rem; color: var(--muted); margin: 0.8rem 0 0;
    max-width: 52ch; line-height: 1.62;
}
.hero-facts { display: flex; gap: 1.6rem; margin-top: 1.5rem; flex-wrap: wrap; }
.fact { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--muted); }
.fact b { color: var(--text); font-weight: 600; }
.fact-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.fact-dot.ok  { background: var(--good); box-shadow: 0 0 10px rgba(52,211,153,0.6); }
.fact-dot.bad { background: var(--bad);  box-shadow: 0 0 10px rgba(248,113,113,0.6); }

/* The policy card — the one bold element on the page */
.pcard-wrap { perspective: 1200px; display: flex; justify-content: flex-end; }
.pcard {
    width: 100%; max-width: 360px; aspect-ratio: 1.6 / 1;
    border-radius: 18px; padding: 1.25rem 1.4rem;
    position: relative; overflow: hidden;
    background:
        linear-gradient(120deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02) 40%, transparent 60%),
        linear-gradient(160deg, #123A44 0%, #0F2A3E 45%, #0B1A2C 100%);
    border: 1px solid rgba(159, 243, 230, 0.28);
    box-shadow:
        0 30px 60px -25px rgba(0,0,0,0.9),
        inset 0 1px 0 rgba(255,255,255,0.12);
    transform: rotateY(-9deg) rotateX(5deg);
    transition: transform 0.5s cubic-bezier(.2,.7,.2,1);
    display: flex; flex-direction: column; justify-content: space-between;
}
.pcard-wrap:hover .pcard { transform: rotateY(0) rotateX(0); }
.pcard::before {
    content: ""; position: absolute; top: -40%; right: -30%; width: 75%; height: 160%;
    background: radial-gradient(closest-side, rgba(45,212,191,0.28), transparent);
    filter: blur(6px);
}
.pcard-top { display: flex; justify-content: space-between; align-items: flex-start; }
.pcard-brand { font-size: 0.82rem; font-weight: 700; letter-spacing: -0.01em; color: #EAFBF7; }
.pcard-brand small { display: block; font-weight: 500; color: rgba(234,251,247,0.6); font-size: 0.68rem; margin-top: 2px; }
.chip {
    width: 38px; height: 28px; border-radius: 6px;
    background: linear-gradient(135deg, #E9D8A6, #B9974E 60%, #E4CB8A);
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.25);
    position: relative;
}
.chip::before, .chip::after {
    content: ""; position: absolute; left: 0; right: 0; height: 1px; background: rgba(0,0,0,0.28);
}
.chip::before { top: 9px; } .chip::after { bottom: 9px; }
.pcard-num {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem; letter-spacing: 0.06em; color: #DFF7F2;
    margin: 0.4rem 0; word-break: break-all;
}
.pcard-bottom { display: flex; justify-content: space-between; gap: 1rem; }
.pcard-field small { display: block; font-size: 0.62rem; color: rgba(223,247,242,0.55); margin-bottom: 2px; }
.pcard-field span  { font-size: 0.78rem; font-weight: 600; color: #EAFBF7; }

@media (max-width: 860px) {
    .hero { grid-template-columns: 1fr; padding: 1.8rem 1.5rem; }
    .pcard-wrap { justify-content: flex-start; }
    .pcard { transform: none; }
    .hero-title { font-size: 1.9rem; }
}

/* =========================== SECTION HEADINGS =========================== */
.sec {
    display: flex; align-items: baseline; gap: 12px;
    margin: 1.5rem 0 0.75rem;
}
.sec h3 { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 0; letter-spacing: -0.01em; }
.sec span { font-size: 0.78rem; color: var(--muted); }
.sec::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }

/* =========================== SIDEBAR =========================== */
[data-testid="stSidebar"] { background: #0A0F1C; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }
.sb-brand { display: flex; align-items: center; gap: 11px; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
.sb-logo {
    width: 38px; height: 38px; border-radius: 11px; flex: none; font-size: 1.15rem;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(45,212,191,0.22), rgba(56,189,248,0.18));
    border: 1px solid var(--border-hi);
}
.sb-name { font-size: 0.98rem; font-weight: 700; }
.sb-tag  { font-size: 0.72rem; color: var(--muted); }

.sb-h { font-size: 0.8rem; font-weight: 700; color: var(--text); margin: 1.2rem 0 0.5rem; }

.readiness {
    border-radius: var(--r-md); padding: 0.85rem 0.95rem; margin-bottom: 0.6rem;
    border: 1px solid var(--border); background: var(--surface);
}
.readiness.ok  { border-color: rgba(52,211,153,0.35);  background: linear-gradient(180deg, rgba(52,211,153,0.10), var(--surface)); }
.readiness.bad { border-color: rgba(248,113,113,0.35); background: linear-gradient(180deg, rgba(248,113,113,0.10), var(--surface)); }
.readiness-title { font-size: 0.86rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
.readiness-sub   { font-size: 0.74rem; color: var(--muted); margin-top: 3px; }

.check { display: flex; align-items: flex-start; gap: 10px; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.check:last-child { border-bottom: none; }
.check-ic {
    width: 20px; height: 20px; border-radius: 50%; flex: none; font-size: 0.7rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.check-ic.ok  { background: rgba(52,211,153,0.15);  color: var(--good); }
.check-ic.bad { background: rgba(248,113,113,0.15); color: var(--bad); }
.check-t { font-size: 0.8rem; font-weight: 600; }
.check-d { font-size: 0.72rem; color: var(--muted); word-break: break-all; margin-top: 1px; }

.spec { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.spec div {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-sm);
    padding: 0.55rem 0.7rem;
}
.spec small { display: block; font-size: 0.66rem; color: var(--muted); }
.spec b { display: block; font-size: 0.78rem; font-weight: 600; margin-top: 2px; word-break: break-all; }

/* =========================== BUTTONS =========================== */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    border-radius: var(--r-md); padding: 0.7rem 0.95rem;
    font-size: 0.83rem; font-weight: 500; text-align: left; line-height: 1.45;
    transition: border-color 0.15s ease, background 0.15s ease; box-shadow: none;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: var(--surface-2); border-color: var(--border-hi); color: #FFFFFF;
}
div[data-testid="stButton"] > button:focus:not(:active) { border-color: var(--border-hi); color: #FFFFFF; }

/* Suggested-question chips */
.st-key-sample_0 button, .st-key-sample_1 button, .st-key-sample_2 button, .st-key-sample_3 button {
    position: relative; padding-left: 2.4rem !important; min-height: 64px;
}
.st-key-sample_0 button::before, .st-key-sample_1 button::before,
.st-key-sample_2 button::before, .st-key-sample_3 button::before {
    content: "?"; position: absolute; left: 0.9rem; top: 50%; transform: translateY(-50%);
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 800; color: var(--accent);
    background: rgba(45,212,191,0.12); border: 1px solid var(--border-hi);
}

/* Sidebar toggle */
.st-key-sidebar_toggle button {
    padding: 0.4rem 0.85rem !important; font-size: 0.76rem !important; font-weight: 600 !important;
    border-radius: 10px !important; text-align: center !important;
    color: var(--muted) !important; background: rgba(17, 28, 46, 0.65) !important;
}
.st-key-sidebar_toggle button:hover { color: var(--accent) !important; border-color: var(--border-hi) !important; }

/* =========================== CHAT =========================== */
/* User message: right-aligned bubble rendered as HTML */
.u-row { display: flex; justify-content: flex-end; margin: 0.4rem 0 0.9rem; }
.u-bubble {
    max-width: 72%; padding: 0.8rem 1.05rem;
    border-radius: 16px 16px 4px 16px;
    background: linear-gradient(135deg, rgba(45,212,191,0.18), rgba(56,189,248,0.14));
    border: 1px solid var(--border-hi);
    color: var(--text); font-size: 0.93rem; line-height: 1.6;
    box-shadow: 0 12px 30px -18px rgba(45,212,191,0.5);
}

/* Assistant message: native chat container, dressed */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px 16px 16px 16px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.9rem;
}
[data-testid="stChatMessage"] p { color: var(--text); line-height: 1.7; font-size: 0.93rem; }
[data-testid="stChatMessage"] li { color: var(--text); }
[data-testid="stChatMessage"] code {
    background: rgba(45,212,191,0.10); color: var(--accent);
    padding: 1px 6px; border-radius: 5px; font-size: 0.85em;
}
[data-testid="stChatMessage"] table { font-size: 0.85rem; border-collapse: collapse; }
[data-testid="stChatMessage"] th { background: var(--surface-2); color: var(--text); }
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] th {
    border: 1px solid var(--border); padding: 6px 10px;
}
[data-testid="stChatMessage"] blockquote { border-left: 2px solid var(--border-hi); color: var(--muted); }

/* Answer footer (elapsed time, chunk count) */
.a-meta {
    display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
    margin-top: 0.75rem; padding-top: 0.6rem; border-top: 1px dashed var(--border);
    font-size: 0.72rem; color: var(--muted);
}
.a-meta b { color: var(--text); font-weight: 600; }

/* Typing indicator */
.typing { display: inline-flex; align-items: center; gap: 10px; color: var(--muted); font-size: 0.85rem; }
.typing i { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); display: inline-block; animation: blink 1.2s infinite; }
.typing i:nth-child(2) { animation-delay: 0.15s; } .typing i:nth-child(3) { animation-delay: 0.3s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--border-hi); box-shadow: 0 0 0 4px rgba(45,212,191,0.10); }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stBottomBlockContainer"] { background: linear-gradient(180deg, transparent, var(--bg) 40%); }

/* =========================== SOURCES =========================== */
[data-testid="stExpander"] details {
    background: rgba(11,17,32,0.6); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stExpander"] summary { font-size: 0.8rem; color: var(--muted); font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: var(--accent); }

.cite { display: grid; grid-template-columns: 28px 1fr; gap: 12px; padding: 0.7rem 0; border-bottom: 1px solid var(--border); }
.cite:last-child { border-bottom: none; }
.cite-n {
    width: 26px; height: 26px; border-radius: 8px; font-size: 0.72rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    color: var(--accent); background: rgba(45,212,191,0.12); border: 1px solid var(--border-hi);
}
.cite-t { font-size: 0.76rem; font-weight: 600; color: var(--text); margin-bottom: 3px; }
.cite-b { font-size: 0.79rem; color: var(--muted); line-height: 1.62; }

/* Feedback widget */
[data-testid="stFeedback"] button { border-color: transparent !important; background: transparent !important; }

/* =========================== FOOTER NOTE =========================== */
.note {
    margin-top: 2rem; padding: 0.9rem 1.1rem; border-radius: var(--r-md);
    border: 1px solid var(--border); background: rgba(17,28,46,0.5);
    font-size: 0.76rem; color: var(--muted); line-height: 1.6;
    display: flex; gap: 10px; align-items: flex-start;
}
.note b { color: var(--text); font-weight: 600; }

hr { border-color: var(--border); }
.stAlert { border-radius: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# SIDEBAR VISIBILITY
# --------------------------------------------------------------------------- #
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

if not st.session_state.sidebar_open:
    _SIDEBAR_CSS = "<style>[data-testid='stSidebar'] { display: none !important; }</style>"
else:
    _SIDEBAR_CSS = """
        <style>
            [data-testid="stSidebar"] {
                display: flex !important; visibility: visible !important; opacity: 1 !important;
                transform: none !important; margin-left: 0 !important; left: 0 !important;
                width: 310px !important; min-width: 310px !important; max-width: 310px !important;
            }
            [data-testid="stSidebar"] > div { visibility: visible !important; }
        </style>
    """
st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_assistant():
    """Initialise and cache the health assistant agent."""
    try:
        config.check_api_key()
        return build_health_assistant(), None
    except Exception as exc:  # noqa: BLE001
        logger.error("Error loading health assistant: %s", exc, exc_info=True)
        return None, str(exc)


def _clean_sources(raw):
    """Normalise whatever the pipeline hands back into [{'title', 'text'}]."""
    if not raw:
        return []
    if isinstance(raw, (str, bytes)):
        raw = [raw]

    out = []
    for i, item in enumerate(raw, start=1):
        title, text = f"Clause excerpt {i}", ""
        page_content = getattr(item, "page_content", None) or getattr(item, "text", None)
        metadata = getattr(item, "metadata", None)

        if isinstance(item, dict):
            text = item.get("page_content") or item.get("text") or item.get("content") or ""
            metadata = item.get("metadata") or item
        elif page_content is not None:
            text = page_content
        else:
            text = str(item)

        if isinstance(metadata, dict):
            src = metadata.get("source") or metadata.get("file_name") or metadata.get("file_path")
            page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_label")
            bits = []
            if src:
                bits.append(os.path.basename(str(src)))
            if page is not None:
                bits.append(f"page {page}")
            if bits:
                title = ", ".join(bits)

        text = " ".join(str(text).split())
        if text:
            out.append({"title": title, "text": text})
    return out


def normalise_answer(raw):
    """Return (answer_text, sources) for a variety of pipeline return shapes."""
    if isinstance(raw, str):
        return raw, []
    if isinstance(raw, tuple) and len(raw) == 2:
        return str(raw[0]), _clean_sources(raw[1])
    if isinstance(raw, dict):
        text = (raw.get("answer") or raw.get("output") or raw.get("result")
                or raw.get("response") or raw.get("content") or "")
        sources = (raw.get("sources") or raw.get("source_documents")
                   or raw.get("context") or raw.get("documents") or [])
        return (str(text) if text else str(raw)), _clean_sources(sources)
    text = getattr(raw, "answer", None) or getattr(raw, "content", None) or getattr(raw, "output", None)
    sources = getattr(raw, "sources", None) or getattr(raw, "source_documents", None)
    if text is not None:
        return str(text), _clean_sources(sources)
    return str(raw), []


def stream_text(text, delay=0.012):
    for token in text.split(" "):
        yield token + " "
        time.sleep(delay)


def render_user(text):
    st.markdown(
        f'<div class="u-row"><div class="u-bubble">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Show the {len(sources)} policy excerpt(s) this answer is based on"):
        for i, s in enumerate(sources, start=1):
            snippet = html.escape(s["text"][:900]) + ("…" if len(s["text"]) > 900 else "")
            st.markdown(
                f'<div class="cite"><div class="cite-n">{i}</div>'
                f'<div><div class="cite-t">{html.escape(s["title"])}</div>'
                f'<div class="cite-b">{snippet}</div></div></div>',
                unsafe_allow_html=True,
            )


def render_meta(msg):
    bits = []
    if msg.get("elapsed") is not None:
        bits.append(f"Answered in <b>{msg['elapsed']:.1f}s</b>")
    if msg.get("sources"):
        bits.append(f"<b>{len(msg['sources'])}</b> excerpts retrieved")
    bits.append(f"Model <b>{html.escape(str(config.LLM_MODEL_NAME))}</b>")
    st.markdown('<div class="a-meta">' + "".join(f"<span>{b}</span>" for b in bits) + "</div>",
                unsafe_allow_html=True)


def render_feedback(idx):
    """Thumbs up/down under an answer (only on Streamlit builds that ship st.feedback)."""
    if not hasattr(st, "feedback"):
        return
    key = f"fb_{idx}"
    val = st.feedback("thumbs", key=key)
    if val is not None and st.session_state.get(f"{key}_logged") != val:
        st.session_state[f"{key}_logged"] = val
        logger.info("Feedback on message %s: %s", idx, "up" if val == 1 else "down")


def transcript_markdown(messages):
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# HealthShield AI — Conversation\n\n_Exported {stamp}_\n"]
    for m in messages:
        who = "You" if m["role"] == "user" else "HealthShield AI"
        lines.append(f"\n### {who}\n\n{m['content']}\n")
        for i, s in enumerate(m.get("sources", []), start=1):
            lines.append(f"> [{i}] **{s['title']}** — {s['text'][:500]}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# STATE
# --------------------------------------------------------------------------- #
GREETING = (
    "Hello — I'm your policy advisor. Ask about coverage limits, exclusions, waiting periods, "
    "room-rent caps or claim procedures, and I'll answer from your policy document and show "
    "you the exact excerpts I relied on."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING, "sources": []}]
if "pending" not in st.session_state:
    st.session_state.pending = None

# --------------------------------------------------------------------------- #
# SYSTEM CHECKS
# --------------------------------------------------------------------------- #
pdf_ok = os.path.exists(config.DATA_FILE_PATH)
try:
    qdrant_ok = bool(vector_store_exists())
except Exception:  # noqa: BLE001
    qdrant_ok = False
all_ok = pdf_ok and qdrant_ok
pdf_name = os.path.basename(config.DATA_FILE_PATH)

# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">🛡️</div>'
        '<div><div class="sb-name">HealthShield AI</div>'
        '<div class="sb-tag">Policy assistant console</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="readiness {"ok" if all_ok else "bad"}">'
        f'<div class="readiness-title">{"✓ Ready to answer" if all_ok else "⚠ Needs attention"}</div>'
        f'<div class="readiness-sub">{"Document indexed and vector store reachable." if all_ok else "One or more components are unavailable — see below."}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-h">Components</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="check"><div class="check-ic {"ok" if pdf_ok else "bad"}">{"✓" if pdf_ok else "!"}</div>'
        f'<div><div class="check-t">Policy document</div><div class="check-d">{html.escape(pdf_name)}</div></div></div>'
        f'<div class="check"><div class="check-ic {"ok" if qdrant_ok else "bad"}">{"✓" if qdrant_ok else "!"}</div>'
        f'<div><div class="check-t">Vector store</div><div class="check-d">Collection {html.escape(str(config.QDRANT_COLLECTION_NAME))}</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-h">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="spec">'
        f'<div><small>Language model</small><b>{html.escape(str(config.LLM_MODEL_NAME))}</b></div>'
        f'<div><small>Embeddings</small><b>{html.escape(str(config.EMBEDDING_MODEL_NAME))}</b></div>'
        f'<div><small>Excerpts per answer</small><b>{config.TOP_K_RESULTS}</b></div>'
        f'<div><small>Chunk size</small><b>{config.CHUNK_SIZE} chars</b></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-h">Preferences</div>', unsafe_allow_html=True)
    show_sources = st.toggle("Show policy excerpts under answers", value=True)
    typing_fx = st.toggle("Animate answers as they arrive", value=True)

    st.markdown('<div class="sb-h">This conversation</div>', unsafe_allow_html=True)
    turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"{turns} question{'s' if turns != 1 else ''} asked")
    st.download_button(
        "Download conversation (.md)",
        data=transcript_markdown(st.session_state.messages),
        file_name=f"healthshield-chat-{dt.datetime.now():%Y%m%d-%H%M}.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=turns == 0,
    )
    if st.button("Clear conversation", use_container_width=True, disabled=turns == 0):
        st.session_state.messages = [{"role": "assistant", "content": GREETING, "sources": []}]
        st.toast("Conversation cleared")
        st.rerun()

# --------------------------------------------------------------------------- #
# TOGGLE + HERO
# --------------------------------------------------------------------------- #
_tc, _ = st.columns([1, 5])
with _tc:
    st.button(
        "◀  Hide console" if st.session_state.sidebar_open else "☰  Show console",
        key="sidebar_toggle", on_click=toggle_sidebar, use_container_width=True,
        help="Collapse or expand the console panel",
    )

st.markdown(
    f"""
<div class="hero">
  <div>
    <div class="hero-live"><span class="dot"></span> Answers grounded in your policy document</div>
    <h1 class="hero-title">Know exactly what<br/>your policy covers</h1>
    <p class="hero-sub">
      Ask about coverage limits, exclusions, waiting periods and room-rent caps.
      Every answer is drawn from the document and comes with the clauses it relied on.
    </p>
    <div class="hero-facts">
      <div class="fact"><span class="fact-dot {'ok' if pdf_ok else 'bad'}"></span>Document <b>{'indexed' if pdf_ok else 'missing'}</b></div>
      <div class="fact"><span class="fact-dot {'ok' if qdrant_ok else 'bad'}"></span>Search <b>{'online' if qdrant_ok else 'offline'}</b></div>
      <div class="fact"><span class="fact-dot ok"></span><b>{config.TOP_K_RESULTS}</b>&nbsp;excerpts per answer</div>
    </div>
  </div>
  <div class="pcard-wrap">
    <div class="pcard">
      <div class="pcard-top">
        <div class="pcard-brand">HealthShield AI<small>Policy knowledge card</small></div>
        <div class="chip"></div>
      </div>
      <div class="pcard-num">{html.escape(pdf_name)}</div>
      <div class="pcard-bottom">
        <div class="pcard-field"><small>Index</small><span>{html.escape(str(config.QDRANT_COLLECTION_NAME))}</span></div>
        <div class="pcard-field"><small>Model</small><span>{html.escape(str(config.LLM_MODEL_NAME))}</span></div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# AGENT BOOT
# --------------------------------------------------------------------------- #
with st.spinner("Connecting to the vector store and loading the agent…"):
    agent, error_msg = load_assistant()

if error_msg:
    st.error(f"**The assistant couldn't start.** {error_msg}")
    st.info(
        "Check that `.env` contains a valid `GOOGLE_API_KEY`, `JINA_API_KEY` and the "
        "`QDRANT_*` connection settings, then reload this page."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# SUGGESTED QUESTIONS (only before the first question)
# --------------------------------------------------------------------------- #
SAMPLES = [
    "How do room rent limits differ between Optima Select and Optima Secure?",
    "What are the waiting periods for pre-existing diseases?",
    "Are day care procedures and AYUSH treatments covered?",
    "Explain the restoration benefit clause.",
]

if len(st.session_state.messages) <= 1:
    st.markdown('<div class="sec"><h3>Try one of these</h3><span>or type your own below</span></div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SAMPLES):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

st.markdown('<div class="sec"><h3>Conversation</h3></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# HISTORY
# --------------------------------------------------------------------------- #
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        render_user(msg["content"])
        continue
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(msg["content"])
        if show_sources:
            render_sources(msg.get("sources"))
        if idx > 0:  # no footer/feedback on the greeting
            render_meta(msg)
            render_feedback(idx)

# --------------------------------------------------------------------------- #
# INPUT + RESPONSE
# --------------------------------------------------------------------------- #
typed = st.chat_input("Ask anything about your health insurance policy…")
query = typed or st.session_state.pending
st.session_state.pending = None

if query:
    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    render_user(query)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing">Reading the policy and matching clauses <i></i><i></i><i></i></div>',
            unsafe_allow_html=True,
        )
        t0 = time.perf_counter()
        try:
            raw = ask(agent, query)
            elapsed = time.perf_counter() - t0
            answer, sources = normalise_answer(raw)
            placeholder.empty()

            if typing_fx:
                st.write_stream(stream_text(answer))
            else:
                st.markdown(answer)

            record = {"role": "assistant", "content": answer, "sources": sources, "elapsed": elapsed}
            st.session_state.messages.append(record)

            if show_sources:
                render_sources(sources)
            render_meta(record)
            render_feedback(len(st.session_state.messages) - 1)

        except Exception as exc:  # noqa: BLE001
            placeholder.empty()
            logger.error("Query failed: %s", exc, exc_info=True)
            err = (
                "I couldn't retrieve an answer for that question. "
                f"The pipeline reported: `{exc}`. Try rephrasing, or check the console for component status."
            )
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})

# --------------------------------------------------------------------------- #
# FOOTER NOTE
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="note"><span>ⓘ</span><span><b>For guidance only.</b> Answers are generated from the '
    'uploaded policy wording and may omit context. Confirm benefits, limits and exclusions with your '
    'insurer before making a claim decision.</span></div>',
    unsafe_allow_html=True,
)