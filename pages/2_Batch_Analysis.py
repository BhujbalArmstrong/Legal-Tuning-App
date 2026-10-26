import streamlit as st
import pandas as pd
import time

from ocr_utils import extract_text
from segment_utils import segment_clauses
from translate_compare import translate_google, back_translate
from llm_pipeline import run_proposed_pipeline, run_single_prompt_baseline
from evaluate_utils import number_mismatch_flag, round_trip_consistency, aggregate_summary

st.set_page_config(page_title="Batch Analysis — Legal Tuning", layout="wide")

st.title("📊 Batch Analysis: Multi-Document, Multi-Clause Comparison")
st.caption("Upload multiple documents, process many clauses automatically, and get one aggregated accuracy report.")

st.warning("⚠️ Free-tier translation/LLM services are rate-limited. Recommended: process no more than 15-20 clauses total per run.")

uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF, image, or TXT)",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    accept_multiple_files=True
)

max_clauses_per_doc = st.slider("Max clauses to process PER document", 1, 15, 5)
delay_seconds = st.slider("Delay between clauses (seconds) — increase if you hit rate-limit errors", 5, 30, 15)

if uploaded_files and st.button("Run Batch Analysis"):
    all_rows = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    doc_clause_pairs = []
    for f in uploaded_files:
        try:
            text = extract_text(f)
            clauses = segment_clauses(text)[:max_clauses_per_doc]
            for c in clauses:
                doc_clause_pairs.append((f.name, c))
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")

    total = len(doc_clause_pairs)
    if total == 0:
        st.error("No clauses could be extracted from the uploaded files.")
    else:
        for idx, (doc_name, clause) in enumerate(doc_clause_pairs):
            progress_text.text(f"Processing {doc_name} — clause {clause['clause_id']} ({idx+1}/{total})")
            clause_text = clause["raw_text"]

            pipeline_outputs = {}

            try:
                pipeline_outputs["Baseline 1 (Google Translate, direct)"] = translate_google(clause_text)
            except Exception as e:
                pipeline_outputs["Baseline 1 (Google Translate, direct)"] = f"[ERROR: {e}]"

            try:
                pipeline_outputs["Baseline 2 (Single-prompt LLM)"] = run_single_prompt_baseline(clause_text)
            except Exception as e:
                pipeline_outputs["Baseline 2 (Single-prompt LLM)"] = f"[ERROR: {e}]"

            try:
                proposed = run_proposed_pipeline(clause_text)
                pipeline_outputs["Proposed (Structured pipeline)"] = proposed["hindi_output"]
            except Exception as e:
                pipeline_outputs["Proposed (Structured pipeline)"] = f"[ERROR: {e}]"

            for pipeline_name, output in pipeline_outputs.items():
                mismatch = number_mismatch_flag(clause_text, output)
                is_error = isinstance(output, str) and output.startswith("[")
                back = "" if is_error else back_translate(output)
                consistency = round_trip_consistency(clause_text, back) if back and not str(back).startswith("[") else {"f1": None}

                all_rows.append({
                    "Document": doc_name,
                    "Clause ID": clause["clause_id"],
                    "Pipeline": pipeline_name,
                    "Source Clause (truncated)": clause_text[:100],
                    "Output (truncated)": str(output)[:150],
                    "Numbers in source": len(mismatch["source_numbers"]),
                    "Numbers missing": len(mismatch["missing_numbers"]),
                    "Number mismatch flag": "⚠️ YES" if mismatch["flag_mismatch"] else "OK",
                    "Back-translation (round-trip) BERTScore F1": round(consistency["f1"], 3) if consistency["f1"] else "N/A",
                })

            progress_bar.progress((idx + 1) / total)
            time.sleep(delay_seconds)

        progress_text.text("Done.")
        df = pd.DataFrame(all_rows)

        st.subheader("1. Detailed Clause-Level Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("2. Aggregated Summary (per pipeline)")
        summary_df = aggregate_summary(df)
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download detailed results CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="batch_detailed_results.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download summary report CSV",
            summary_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_summary_report.csv",
            mime="text/csv",
                                        )
