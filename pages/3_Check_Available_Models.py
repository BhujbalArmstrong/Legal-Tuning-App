import os
import re
import time
import concurrent.futures

import streamlit as st
import google.generativeai as genai
from indictrans_utils import translate_indictrans2

st.set_page_config(page_title="Check Available Models", layout="wide")
st.title("Gemini model and transport check")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
    st.stop()

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def configure(transport: str):
    genai.configure(api_key=API_KEY, transport=transport)


def list_models():
    configure("rest")
    return [
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]


def test_one(name: str, transport: str, timeout: int = 30):
    configure(transport)

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
        return round(time.time() - t0, 1), "OK: " + text[:40]
    except concurrent.futures.TimeoutError:
        return round(time.time() - t0, 1), "TIMEOUT (no reply)"
    except Exception as e:
        return round(time.time() - t0, 1), "ERROR: " + str(e)[:160]


st.subheader("1. Models your key can use")
if st.button("List models"):
    try:
        st.session_state["models"] = list_models()
    except Exception as e:
        st.error(f"Could not list models: {e}")

models = st.session_state.get("models", [])
if models:
    st.write(f"{len(models)} models support generateContent.")
    st.code("\n".join(models), language="text")

    flash = sorted([m for m in models if "flash" in m], key=_version, reverse=True)
    chosen = st.multiselect("Models to test", models, default=flash[:2])
    transports = st.multiselect("Transports", ["rest", "grpc"], default=["rest", "grpc"])

    st.subheader("2. Latency test (one tiny prompt each, 12 s apart)")
    if st.button("Run latency test"):
        rows = []
        table = st.empty()
        total = len(chosen) * len(transports)
        done = 0
        for name in chosen:
            for tr in transports:
                done += 1
                with st.spinner(f"Testing {name} via {tr} ({done}/{total})..."):
                    secs, result = test_one(name, tr)
                rows.append({"Model": name, "Transport": tr, "Seconds": secs, "Result": result})
                table.table(rows)
                if done < total:
                    time.sleep(12)
        st.success("Done. Screenshot the table.")

st.divider()
st.subheader("IndicTrans2 Test (temporary)")
test_text = st.text_input("English text", "The Tenant shall pay Rs. 4,29,000 as deposit.")
if st.button("Test IndicTrans2"):
    st.write(translate_indictrans2(test_text))
