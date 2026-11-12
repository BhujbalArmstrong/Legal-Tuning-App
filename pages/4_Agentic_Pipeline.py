import json
import time
import streamlit as st
import agent_graph
from agent_graph import build_graph, SAMPLE_CLAUSE, call_gemini

st.set_page_config(page_title="Agentic Pipeline (LangGraph)", layout="wide")
st.title("Agentic Legal Tuning (LangChain + LangGraph)")
st.caption(
    "Extract → English → Hindi → Verify → (Reflect → regenerate) → Finalize. "
    "The verifier checks numbers and meaning, and loops back with feedback if it fails."
)


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

st.subheader("Step 0: Gemini connection test (1 cheap call)")
if st.button("Test Gemini connection"):
    t0 = time.time()
    try:
        reply = call_gemini("Reply with the single word: ready")
        st.success(
            f"Gemini works. Model: {agent_graph._WORKING_MODEL} | "
            f"{time.time() - t0:.1f}s | reply: {reply}"
        )
    except Exception as e:
        st.error(f"Gemini failed after {time.time() - t0:.0f}s: {e}")

st.subheader("Step 1: Run the agent")
clause = st.text_area("English clause", value=SAMPLE_CLAUSE, height=170)
c1, c2 = st.columns(2)
max_attempts = c1.slider("Max attempts", 1, 3, 3)
use_judge = c2.checkbox("Use LLM faithfulness judge (one extra API call per attempt)", value=True)

with st.expander("Graph structure (Mermaid text, usable in the report)"):
    try:
        st.code(graph.get_graph().draw_mermaid(), language="text")
    except Exception as e:
        st.write(f"Could not draw graph: {e}")

if st.button("Run agentic pipeline"):
    if not clause.strip():
        st.warning("Enter a clause first.")
        st.stop()

    state = {
        "clause": clause,
        "attempt": 1,
        "max_attempts": max_attempts,
        "use_judge": use_judge,
        "reflections": [],
        "history": [],
    }

    st.subheader("Live trace")
    try:
        with st.spinner("Running graph..."):
            for step in graph.stream(state, stream_mode="updates"):
                for node, update in step.items():
                    state.update(update)
                    with st.expander(f"✅ {node}"):
                        st.json(update)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    final = state.get("final", {})
    st.subheader("Result")
    if final.get("status") == "passed":
        st.success(f"Passed verification after {final.get('attempts_used')} attempt(s).")
    else:
        st.warning("Verification still failing after the maximum attempts. Needs human review.")

    st.markdown("**Hindi output**")
    st.write(final.get("hindi", ""))
    st.markdown("**English explanation**")
    st.write(final.get("english", ""))

    st.markdown("**Attempt history (number omissions per attempt)**")
    st.table([
        {
            "Attempt": h["attempt"],
            "Missing numbers": ", ".join(h["missing_numbers"]) or "none",
            "Judge OK": h["judge_ok"],
            "Passed": h["passed"],
        }
        for h in state.get("history", [])
    ])

    if state.get("reflections"):
        st.markdown("**Reflections (memory carried into regeneration)**")
        for r in state["reflections"]:
            st.write("- " + r)

    st.download_button(
        "Download run (JSON)",
        data=json.dumps(state, ensure_ascii=False, indent=2),
        file_name="agentic_run.json",
        mime="application/json",
    )
