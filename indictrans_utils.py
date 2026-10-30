import streamlit as st
from gradio_client import Client

SPACE_ID = "manjunathmvbhat/Indic_Translation"

_client = None

def get_client():
    global _client
    if _client is None:
        hf_token = st.secrets.get("HF_TOKEN", None)
        _client = Client(SPACE_ID, hf_token=hf_token)
    return _client

def translate_indictrans2(text, tgt_lang="hin_Deva"):
    try:
        client = get_client()
        result = client.predict(
            text=text,
            tgt_lang=tgt_lang,
            api_name="/predict"
        )
        return result
    except Exception as e:
        return f"[IndicTrans2 Error: {e}]"
