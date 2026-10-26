import streamlit as st
import google.generativeai as genai
import os

st.title("🔍 Check Available Gemini Models")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.write("Models available to your API key that support `generateContent`:")

try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            st.write(f"✅ `{m.name}`")
except Exception as e:
    st.error(f"Error listing models: {e}")
