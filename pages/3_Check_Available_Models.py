import streamlit as st
import google.generativeai as genai
import os
from indictrans_utils import translate_indictrans2

st.title("🔍 Check Available Gemini Models")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.write("Models available to your API key that support `generateContent`:")

try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            st.write(f"✅ `{m.name}`")
except Exception as e:
    st.error(f"Error listing models: {e}")

st.subheader("IndicTrans2 Test (temporary)")
test_text = st.text_input("Test English text", "The tenant shall pay rent every month.")
if st.button("Translate with IndicTrans2"):
    output = translate_indictrans2(test_text)
    st.write(output)
