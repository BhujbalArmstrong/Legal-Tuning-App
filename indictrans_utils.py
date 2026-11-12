import streamlit as st
from gradio_client import Client

SPACE_ID = "manjunathmvbhat/Indic_Translation"
TIMEOUT_SECONDS = 60


@st.cache_resource
def get_client():
    token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
    try:
        if token:
            return Client(SPACE_ID, hf_token=token)
        return Client(SPACE_ID)
    except TypeError:
        return Client(SPACE_ID)


def translate_indictrans2(text: str, tgt_lang: str = "hin_Deva") -> str:
    try:
        client = get_client()
        job = client.submit(text=text, tgt_lang=tgt_lang, api_name="/predict")
        try:
            return job.result(timeout=TIMEOUT_SECONDS)
        except Exception as e:
            try:
                job.cancel()
            except Exception:
                pass
            if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower() or not str(e):
                return f"[IndicTrans2 Error: timed out after {TIMEOUT_SECONDS}s]"
            return f"[IndicTrans2 Error: {e}]"
    except Exception as e:
        return f"[IndicTrans2 Error: {e}]"
