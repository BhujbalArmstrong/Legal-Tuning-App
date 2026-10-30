import streamlit as st
import pandas as pd

from ocr_utils import extract_text
from segment_utils import segment_clauses
from translate_compare import translate_google, translate_mymemory, back_translate
from llm_pipeline import run_proposed_pipeline, run_single_prompt_baseline
from evaluate_utils import number_mismatch_flag, round_trip_consistency
from indictrans_utils import translate_indictrans2

st.set_page_config(page_title="Legal Tuning — Simplification & Translation Comparator", layout="wide")

st.title("⚖️ Legal Tuning: Document Simplification & Hindi Translation Comparator")
st.caption("Upload a legal document → OCR → clause segmentation → compare pipelines → see inaccuracy flags")

uploaded_file = st.file_uploader("Upload a PDF, image, or TXT legal document", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded_file:
    with st.spinner("Running OCR / reading text..."):
        raw_text = extract_text(uploaded_file)

    st.subheader("1. OCR / Extracted Text")
    with st.expander("View raw extracted text"):
        st.text_area("Raw text", raw_text, height=250)

    st.subheader("2. Clause Segmentation")
    clauses = segment_clauses(raw_text)
    st.write(f"Detected **{len(clauses)}** clauses.")

    clause_options = [f"{c['clause_id']} — {c['raw_text'][:60]}..." for c in clauses]
    selected_idx = st.selectbox("Select a clause to analyse", range(len(clauses)), format_func=lambda i: clause_options[i])
    selected_clause = clauses[selected_idx]["raw_text"]

    st.text_area("Selected clause (source text)", selected_clause, height=120)

    if st.button("Run comparison across pipelines"):
        results = {}

        with st.spinner("Baseline 1: Direct translation (Google)..."):
            results["Baseline 1 (Google Translate, direct)"] = translate_google(selected_clause)

        with st.spinner("Baseline 1b: Direct translation (MyMemory)..."):
            results["Baseline 1b (MyMemory, direct)"] = translate_mymemory(selected_clause)

        with st.spinner("Baseline 2: Single-prompt LLM simplify+translate..."):
            results["Baseline 2 (Single-prompt LLM)"] = run_single_prompt_baseline(selected_clause)

        with st.spinner("Proposed: Structured extraction -> simplify -> translate..."):
            proposed = run_proposed_pipeline(selected_clause)
            results["Proposed (Structured pipeline)"] = proposed["hindi_output"]

        with st.spinner("Baseline 3: IndicTrans2 (ai4bharat, direct)..."):
            results["Baseline 3 (IndicTrans2)"] = translate_indictrans2(selected_clause)

        st.subheader("3. Side-by-Side Output Comparison")
        for name, output in results.items():
            st.markdown(f"**{name}**")
            st.write(output)
            st.markdown("---")

        st.subheader("4. Inaccuracy Checks")

        rows = []
        for name, output in results.items():
            mismatch = number_mismatch_flag(selected_clause, output)
            back = back_translate(output) if output and not output.startswith("[") else ""
            if back.startswith("["):
                back = ""  # back-translation itself failed; don't score a garbage string
            consistency = round_trip_consistency(selected_clause, back) if back else {"f1": None}

            rows.append({
                "Pipeline": name,
                "Numbers in source": len(mismatch["source_numbers"]),
                "Numbers missing in output": len(mismatch["missing_numbers"]),
                "Number mismatch flag": "⚠️ YES" if mismatch["flag_mismatch"] else "OK",
                "Back-translation (round-trip) BERTScore F1": round(consistency["f1"], 3) if consistency["f1"] is not None else None,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.subheader("5. Structured Extraction (Proposed System Internals)")
        with st.expander("View extracted z_i fields + intermediate English"):
            st.text(proposed["structured_fields"])
            st.write("**Simplified English (pre-translation):**")
            st.write(proposed["simplified_english"])

        st.download_button(
            "Download comparison table as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="pipeline_comparison.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a rental agreement / employment letter (PDF, image, or TXT) to begin.")
