from __future__ import annotations
import json
import re
from datetime import timedelta
from typing import Any, Dict, List
import requests
import streamlit as st
from dateutil import parser as dateparser
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PERSIST_DIR = "mydb_insurance"
COLLECTION_NAME = "idoi_insurance"

DEFAULT_MODEL = "huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 4

INS_KWS = [
    "insurance", "policy", "premium", "claim", "deductible", "coverage",
    "liability", "auto", "home", "health", "medicare", "medicaid",
    "uninsured", "underinsured", "idoi", "department of insurance",
    "complaint", "appeal", "denied", "denial", "cancel", "cancellation",
    "lapse", "renewal", "rate", "agent", "adjuster"
]

def get_llm(mode: str) -> ChatOllama:
    mode = (mode or "normal").strip().lower()
    temperature = {"strict": 0.1, "normal": 0.4, "creative": 0.8}.get(mode, 0.4)
    return ChatOllama(model=DEFAULT_MODEL, temperature=temperature)


def _safe_request_json(url: str, params: Dict[str, Any] | None = None, timeout: int = 15) -> Dict[str, Any]:
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_resource(show_spinner=False)
def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )

def tool_search_indiana_insurance_docs(query: str) -> str:
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=TOP_K)
    if not docs:
        return "No relevant Indiana insurance document snippets were found in the local database."

    out = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        source = meta.get("source", meta.get("file_path", "unknown_source"))
        page = meta.get("page", meta.get("page_number", ""))
        header = f"[{i}] Source: {source}" + (f" | Page: {page}" if page != "" else "")
        snippet = (d.page_content or "").strip()
        out.append(header + "\n" + snippet)

    return "\n\n---\n\n".join(out)


def tool_get_weather_open_meteo(location: str) -> str:
    location = (location or "").strip()
    if not location:
        return "Please provide a location, like 'Bloomington, IN' or 'Indianapolis'."

    geo = _safe_request_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "language": "en", "format": "json"},
    )
    results = geo.get("results") or []
    if not results:
        return f"Could not find that location: {location}"

    best = results[0]
    lat = best.get("latitude")
    lon = best.get("longitude")
    name = best.get("name", location)
    admin1 = best.get("admin1", "")
    country = best.get("country", "")

    wx = _safe_request_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,weather_code",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        },
    )
    current = wx.get("current") or {}

    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")
    code = current.get("weather_code")

    return (
        f"Weather for {name}{(', ' + admin1) if admin1 else ''}{(', ' + country) if country else ''}:\n"
        f"- Temperature: {temp} °F\n"
        f"- Wind: {wind} mph\n"
        f"- Weather code: {code}\n"
        f"(Source: Open-Meteo)"
    )


def tool_lookup_us_zip(zip_code: str) -> str:
    z = re.sub(r"[^\d]", "", zip_code or "")
    if len(z) != 5:
        return "Please provide a 5-digit US ZIP code (example: 47401)."

    url = f"https://api.zippopotam.us/us/{z}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return f"ZIP code lookup failed for {z}."

    data = r.json()
    places = data.get("places") or []
    if not places:
        return f"No place data found for {z}."

    p0 = places[0]
    city = p0.get("place name", "")
    state = p0.get("state", "")
    state_abbr = p0.get("state abbreviation", "")
    lat = p0.get("latitude", "")
    lon = p0.get("longitude", "")

    return (
        f"ZIP {z}:\n"
        f"- City: {city}\n"
        f"- State: {state} ({state_abbr})\n"
        f"- Latitude/Longitude: {lat}, {lon}\n"
        f"(Source: Zippopotam.us)"
    )


def tool_insurance_deadline_helper(start_date: str, days: int) -> str:
    try:
        base = dateparser.parse(start_date).date()
    except Exception:
        return "I couldn't parse that start_date. Try formats like '2025-12-01' or 'Dec 1, 2025'."

    try:
        days_int = int(days)
    except Exception:
        return "Days must be an integer (example: 30)."

    deadline = base + timedelta(days=days_int)

    reminders = []
    for n in [days_int, max(days_int - 7, 0), max(days_int - 3, 0), max(days_int - 1, 0)]:
        d = base + timedelta(days=n)
        label = "Deadline day" if n == days_int else f"{days_int - n} day(s) before deadline"
        reminders.append(f"- {d.isoformat()} ({label})")

    return (
        f"Start date: {base.isoformat()}\n"
        f"Window: {days_int} day(s)\n"
        f"Deadline: {deadline.isoformat()}\n"
        f"Suggested reminders:\n" + "\n".join(reminders)
    )


TOOLS = {
    "search_indiana_insurance_docs": tool_search_indiana_insurance_docs,
    "get_weather_open_meteo": tool_get_weather_open_meteo,
    "lookup_us_zip": tool_lookup_us_zip,
    "insurance_deadline_helper": tool_insurance_deadline_helper,
}


ROUTER_SYSTEM = """
You are a tool-routing controller for a chatbot.

Choose which tool(s) to call to answer the user's last message.

Available tools:
1) search_indiana_insurance_docs(query: str) - use for Indiana insurance rules/complaints/claims/consumer rights
2) get_weather_open_meteo(location: str) - use for weather questions
3) lookup_us_zip(zip_code: str) - use for US ZIP code -> city/state lookup
4) insurance_deadline_helper(start_date: str, days: int) - use for deadline calculations

Return ONLY valid JSON in this exact format:
{
  "subtasks": ["...", "..."],
  "calls": [
    {"tool": "tool_name", "args": {...}},
    ...
  ]
}

Rules:
- If no tool is needed, return "calls": [].
- For insurance questions, you should almost always call search_indiana_insurance_docs.
- Keep subtasks 2-3 items.
"""

FINAL_SYSTEM = """
You are an Indiana consumer and insurance assistant who can also answer general questions.
You are NOT a lawyer and you do not provide legal advice. Provide educational information only.

When tool results are provided, use them and cite them:
- For the insurance retrieval tool, cite like: Local IDOI snippet [#]
- For APIs, cite the source name (Open-Meteo, Zippopotam.us)

Output format:
1) Key issues (2–3 bullets)
2) Answer (structured, bullets/headings)
3) If insurance-related: "Not legal advice" reminder.
"""


def llm_json_router(llm: ChatOllama, user_text: str) -> Dict[str, Any]:
    prompt = ROUTER_SYSTEM + "\n\nUSER:\n" + user_text.strip()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Try to extract JSON even if the model wrapped it
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"subtasks": ["Understand the question", "Answer directly"], "calls": []}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {"subtasks": ["Understand the question", "Answer directly"], "calls": []}


def run_tools(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for c in calls:
        tool_name = c.get("tool")
        args = c.get("args", {}) or {}
        fn = TOOLS.get(tool_name)
        if not fn:
            results.append({"tool": tool_name, "ok": False, "result": f"Unknown tool: {tool_name}"})
            continue
        try:
            result = fn(**args)
            results.append({"tool": tool_name, "ok": True, "result": result})
        except Exception as e:
            results.append({"tool": tool_name, "ok": False, "result": f"Tool error: {e}"})
    return results


def build_final_answer(llm: ChatOllama, user_text: str, subtasks: List[str], tool_results: List[Dict[str, Any]]) -> str:
    tool_block = json.dumps(tool_results, indent=2)
    subtasks_block = "\n".join([f"- {s}" for s in (subtasks or [])])

    prompt = (
        FINAL_SYSTEM
        + "\n\nUSER QUESTION:\n"
        + user_text.strip()
        + "\n\nKEY ISSUES:\n"
        + subtasks_block
        + "\n\nTOOL RESULTS (JSON):\n"
        + tool_block
        + "\n\nNow write the final response."
    )
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


st.set_page_config(page_title="Indiana Insurance Assistant (Tool-Calling)")
st.title("Indiana Insurance & General Chat Assistant")
st.caption("Final Project: Tool-calling upgrade (RAG + external APIs)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mode" not in st.session_state:
    st.session_state.mode = "normal"

mode = st.radio(
    "Answer style:",
    options=["strict", "normal", "creative"],
    index=["strict", "normal", "creative"].index(st.session_state.mode),
    horizontal=True,
)
st.session_state.mode = mode

with st.expander("Project notes (quick)"):
    st.write(
        "Tool-calling routing is handled by the LLM producing JSON calls. "
        "Insurance uses your local Chroma DB. Weather uses Open-Meteo. ZIP lookup uses Zippopotam.us. "
        "Deadline helper computes dates locally."
    )

user_input = st.text_input("Ask me anything (Indiana insurance, weather, ZIP codes, or general questions):")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Send")
with col2:
    clear_btn = st.button("Clear chat")

if clear_btn:
    st.session_state.chat_history = []
    st.rerun()

if run_btn and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

    llm = get_llm(st.session_state.mode)

    with st.spinner("Thinking..."):
        router = llm_json_router(llm, user_input)
        subtasks = router.get("subtasks", [])
        calls = router.get("calls", [])

        tool_results = run_tools(calls)
        answer = build_final_answer(llm, user_input, subtasks, tool_results)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.success("Response generated.")

st.subheader("Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
        st.markdown("---")
