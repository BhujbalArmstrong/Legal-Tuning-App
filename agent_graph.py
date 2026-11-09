import os
import re
import json
import time
from typing import TypedDict, List

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END

SAMPLE_CLAUSE = (
    "The Tenant shall pay to the Landlord a refundable security deposit of "
    "Rs. 4,29,000 on or before the date of execution of this Agreement. The "
    "Landlord shall return the deposit within 30 days of the Tenant vacating "
    "the premises, after deducting any amounts due towards unpaid rent or "
    "damage beyond normal wear and tear."
)


# ---------------------------------------------------------------- Gemini
_MODEL_NAME = None


def _get_model_name():
    global _MODEL_NAME
    if _MODEL_NAME:
        return _MODEL_NAME
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    names = [
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]
    preferred = [
        n for n in names
        if "flash" in n and not any(x in n for x in ("exp", "preview", "lite", "thinking", "image", "tts"))
    ]
    _MODEL_NAME = (preferred or names)[0]
    return _MODEL_NAME


def call_gemini(prompt_text: str) -> str:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(_get_model_name())
    last_err = None
    for _ in range(3):
        try:
            resp = model.generate_content(
                prompt_text,
                generation_config=genai.GenerationConfig(temperature=0.2),
            )
            return resp.text.strip()
        except Exception as e:  # rate limit etc.
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "exhausted" in msg:
                time.sleep(12)
            else:
                raise
    raise last_err


def _llm(prompt_value) -> str:
    text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
    return call_gemini(text)


def make_chain(template: str):
    """LangChain chain: prompt | model | string parser."""
    return PromptTemplate.from_template(template) | RunnableLambda(_llm) | StrOutputParser()


# ---------------------------------------------------------------- prompts
EXTRACT_T = (
    "You are a legal analyst. Read the contract clause below and return ONLY a "
    "JSON object (no markdown, no commentary) with these keys: parties, "
    "obligation, amount, deadline, condition, consequence. Use null for anything "
    "not stated. Copy numbers, dates and amounts exactly as written.\n\n"
    "Clause:\n{clause}"
)

EXPLAIN_T = (
    "Explain the contract clause below in plain English for a person with no "
    "legal training. Use short sentences. Keep EVERY number, amount, date and "
    "time period exactly as in the clause. Do not add facts that are not in the "
    "clause.\n\nClause:\n{clause}\n\nStructured summary:\n{structured}\n\n"
    "Feedback from earlier attempts (follow it):\n{feedback}\n\n"
    "Plain English explanation:"
)

HINDI_T = (
    "Write the explanation below in simple, everyday Hindi (Devanagari) for a "
    "person with no legal training. Keep EVERY number, amount, date and time "
    "period from the original clause. Write numbers with ASCII digits exactly as "
    "in the clause (for example 4,29,000). Output only the Hindi text.\n\n"
    "Original clause:\n{clause}\n\nEnglish explanation:\n{english}\n\n"
    "Feedback from earlier attempts (follow it):\n{feedback}\n\n"
    "Hindi:"
)

JUDGE_T = (
    "You are a strict reviewer. Compare the Hindi text with the original English "
    "clause. Reply with exactly the word OK if the Hindi keeps every fact "
    "(parties, amounts, dates, time periods, conditions, consequences). "
    "Otherwise reply with PROBLEM: followed by a one-line list of the facts that "
    "are missing or changed.\n\nClause:\n{clause}\n\nHindi:\n{hindi}"
)

extract_chain = make_chain(EXTRACT_T)
explain_chain = make_chain(EXPLAIN_T)
hindi_chain = make_chain(HINDI_T)
judge_chain = make_chain(JUDGE_T)


# ---------------------------------------------------------------- tools
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def extract_numbers(text: str) -> List[str]:
    t = (text or "").translate(_DEV)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 4,29,000 -> 429000
    return re.findall(r"\d+(?:\.\d+)?", t)


def check_numbers(source: str, output: str) -> dict:
    src = extract_numbers(source)
    out = set(extract_numbers(output))
    missing = [n for n in dict.fromkeys(src) if n not in out]
    return {"source_numbers": list(dict.fromkeys(src)), "missing_numbers": missing}


def _parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_parse_error": text[:200]}


# ---------------------------------------------------------------- state
class GraphState(TypedDict, total=False):
    clause: str
    structured: dict
    english: str
    hindi: str
    verification: dict
    attempt: int
    max_attempts: int
    use_judge: bool
    reflections: List[str]
    history: List[dict]
    final: dict


def _feedback(state) -> str:
    return "\n".join(state.get("reflections", [])) or "None"


# ---------------------------------------------------------------- nodes
# NOTE: node names must differ from state keys, hence the "_node" suffix.
def extract_node(state: GraphState):
    raw = extract_chain.invoke({"clause": state["clause"]})
    return {"structured": _parse_json(raw)}


def english_node(state: GraphState):
    out = explain_chain.invoke({
        "clause": state["clause"],
        "structured": json.dumps(state.get("structured", {}), ensure_ascii=False),
        "feedback": _feedback(state),
    })
    return {"english": out}


def hindi_node(state: GraphState):
    out = hindi_chain.invoke({
        "clause": state["clause"],
        "english": state["english"],
        "feedback": _feedback(state),
    })
    return {"hindi": out}


def verify_node(state: GraphState):
    nums = check_numbers(state["clause"], state["hindi"])
    judge = ""
    judge_ok = True
    if state.get("use_judge", True):
        judge = judge_chain.invoke({"clause": state["clause"], "hindi": state["hindi"]})
        judge_ok = judge.strip().upper().startswith("OK")
    passed = (len(nums["missing_numbers"]) == 0) and judge_ok
    verification = {**nums, "judge": judge, "judge_ok": judge_ok, "passed": passed}
    history = list(state.get("history", [])) + [{
        "attempt": state.get("attempt", 1),
        "missing_numbers": nums["missing_numbers"],
        "judge_ok": judge_ok,
        "passed": passed,
    }]
    return {"verification": verification, "history": history}


def reflect_node(state: GraphState):
    v = state["verification"]
    parts = []
    if v["missing_numbers"]:
        parts.append("these numbers from the clause were missing: " + ", ".join(v["missing_numbers"]))
    if v.get("judge") and not v.get("judge_ok", True):
        parts.append("reviewer said: " + v["judge"])
    note = f"Attempt {state['attempt']} was rejected - " + "; ".join(parts) + ". Fix this."
    return {
        "reflections": list(state.get("reflections", [])) + [note],
        "attempt": state["attempt"] + 1,
    }


def finalize_node(state: GraphState):
    clause_l = state["clause"].lower()
    grounding = {}
    for k, v in (state.get("structured") or {}).items():
        if isinstance(v, str) and v.strip():
            grounding[k] = v.strip().lower() in clause_l
    passed = state["verification"]["passed"]
    return {"final": {
        "status": "passed" if passed else "failed_after_max_attempts",
        "attempts_used": state["attempt"],
        "hindi": state["hindi"],
        "english": state["english"],
        "field_found_verbatim_in_source": grounding,
    }}


def route_after_verify(state: GraphState) -> str:
    if state["verification"]["passed"]:
        return "finalize_node"
    if state["attempt"] >= state["max_attempts"]:
        return "finalize_node"
    return "reflect_node"


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("extract_node", extract_node)
    g.add_node("english_node", english_node)
    g.add_node("hindi_node", hindi_node)
    g.add_node("verify_node", verify_node)
    g.add_node("reflect_node", reflect_node)
    g.add_node("finalize_node", finalize_node)

    g.add_edge(START, "extract_node")
    g.add_edge("extract_node", "english_node")
    g.add_edge("english_node", "hindi_node")
    g.add_edge("hindi_node", "verify_node")
    g.add_conditional_edges(
        "verify_node",
        route_after_verify,
        {"finalize_node": "finalize_node", "reflect_node": "reflect_node"},
    )
    g.add_edge("reflect_node", "hindi_node")
    g.add_edge("finalize_node", END)
    return g.compile()
