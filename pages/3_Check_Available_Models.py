import os
import time
import concurrent.futures

import streamlit as st
import google.generativeai as genai
from indictrans_utils import translate_indictrans2

st.set_page_config(page_title="Check Available Models", layout="wide")
st.title("Gemini model check (gRPC)")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
    st.stop()

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Models with higher free-tier request limits (from the AI Studio Rate Limit page)
CANDIDATES = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
]


def list_models():
    genai.configure(api_key=API_KEY, transport="grpc")
    return [
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]


def test_one(name: str, timeout: int = 30):
    genai.configure(api_key=API_KEY, transport="grpc")

    def _call():
        model = genai.GenerativeModel(name)
        resp = model.generate_content(
            "Reply with the single word: ready",
            request_options={"timeout": timeout},
        )
        return resp.text.strip()

    t0 = time.time()
    fut = _pool.submit(_call)
    try:
        text = fut.result(timeout=timeout + 10)
        return round(time.time() - t0, 1), True, text
    except concurrent.futures.TimeoutError:
        return round(time.time() - t0, 1), False, "TIMEOUT (no reply)"
    except Exception as e:
        return round(time.time() - t0, 1), False, str(e)


st.subheader("1. Models your key can use")
if st.button("List models"):
    try:
        st.session_state["models"] = list_models()
    except Exception as e:
        st.error(f"Could not list models: {e}")

models = st.session_state.get("models", [])
if models:
    st.write(f"{len(models)} models support generateContent.")
    with st.expander("Show full list"):
        st.code("\n".join(models), language="text")

st.subheader("2. Test the candidate models (one tiny prompt each)")
extra = st.text_input("Optional extra model name to test (e.g. models/gemma-4-26b...)", "")
if st.button("Test candidate models"):
    available = set(st.session_state.get("models") or list_models())
    to_test = [m for m in CANDIDATES + ([extra.strip()] if extra.strip() else []) if m in available]
    skipped = [m for m in CANDIDATES if m not in available]
    if skipped:
        st.info("Not in your model list, skipped: " + ", ".join(skipped))

    summary = []
    for i, name in enumerate(to_test):
        with st.spinner(f"Testing {name} ({i + 1}/{len(to_test)})..."):
            secs, ok, detail = test_one(name)
        summary.append({"Model": name, "Seconds": secs, "Result": "OK" if ok else "FAILED"})
        if ok:
            st.success(f"{name} | {secs}s | reply: {detail}")
        else:
            st.error(f"{name} | {secs}s | {detail}")
        time.sleep(3)
    if summary:
        st.table(summary)

st.divider()
st.subheader("IndicTrans2 Test (temporary)")
test_text = st.text_input("English text", "The Tenant shall pay Rs. 4,29,000 as deposit.")
if st.button("Test IndicTrans2"):
    st.write(translate_indictrans2(test_text))
