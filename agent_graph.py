import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import TypedDict, List, Dict, Any

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------
# Gemini access (gRPC transport, quota-aware model fallback)
# ---------------------------------------------------------------
CALL_TIMEOUT = 60
MIN_GAP = 6
PREFERRED_MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.5-flash",
    "models/gemini-3.1-flash-lite",
]

_lock = threading.Lock()
_last_call = 0.0
_WORKING_MODEL = None


def _configure():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"), transport="grpc")


def _wait_gap():
    global _last_call
    with _lock:
        wait = MIN_GAP - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _one_call(model_name, prompt):
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2},
        request_options={"timeout": CALL_TIMEOUT},
    )
    return resp.text


def call_gemini(prompt):
    """Try models in order. 429 -> next model; 503/504/timeout -> retry once."""
    global _WORKING_MODEL
    _configure()

    order = []
    for m in [_WORKING_MODEL, os.environ.get("GEMINI_MODEL")] + PREFERRED_MODELS:
        if m and m not in order:
            order.append(m if m.startswith("models/") else "models/" + m)

    errors = []
    for name in order:
        for attempt in range(2):
            _wait_gap()
            ex = ThreadPoolExecutor(max_workers=1)
            try:
                fut = ex.submit(_one_call, name, prompt)
                text = fut.result(timeout=CALL_TIMEOUT + 10)
                ex.shutdown(wait=False)
                _WORKING_MODEL = name
                return text
            except FuturesTimeout:
                ex.shutdown(wait=False)
                errors.append(f"{name}: timed out")
                if attempt == 0:
                    time.sleep(5)
                    continue
                break
            except Exception as e:
                ex.shutdown(wait=False)
                msg = str(e).lower()
                errors.append(f"{name}: {str(e)[:120]}")
                if "429" in msg or "quota" in msg or "exhausted" in msg:
                    break  # next model immediately
                if any(k in msg for k in ["503", "504", "deadline", "timeout", "timed out", "unavailable"]):
                    if attempt == 0:
                        time.sleep(5)
                        continue
                    break
                break  # 404 or anything else: next model
    raise RuntimeError("All Gemini models failed -> " + " | ".join(errors[-4:]))


def _llm(prompt_value):
    return call_gemini(prompt_value.to_string())


def make_chain(template):
    return PromptTemplate.from_template(template) | RunnableLambda(_llm) | StrOutputParser()


# ---------------------------------------------------------------
# Prompts (LangChain chains)
# ---------------------------------------------------------------
EXTRACT_TEMPLATE = """You are a careful legal analyst. Read the clause and return ONLY a JSON object
with exactly these keys: parties, obligation, amount, deadline, condition, consequence.
Use short phrases copied from the clause where possible. Use "" for a missing field.
Do not add any text outside the JSON.

Clause:
{clause}
"""

EXPLAIN_TEMPLATE = """Explain the following legal clause in plain English for an ordinary person.
Keep EVERY number, amount, date and time period exactly as written. Do not add facts.
Use at most 5 short sentences.

Clause:
{clause}

Extracted fields (for guidance):
{structured}
"""

HINDI_TEMPLATE = """Translate the plain-English explanation below into simple, natural Hindi
that an ordinary person can understand.
Rules:
- Write every number with ASCII digits exactly as in the English (for example 4,29,000 and 30).
- Keep ALL amounts, dates, time periods, conditions and exceptions.
- Do not add or remove any facts.
{feedback}
English explanation:
{english}
"""

JUDGE_TEMPLATE = """You are checking a Hindi explanation against the original English legal clause.

Original clause:
{clause}

Hindi explanation:
{hindi}

Check whether every fact (parties, amounts, dates, time periods, conditions, exceptions)
in the clause is present and correct in the Hindi.

Reply in exactly ONE line and nothing else:
OK
or
PROBLEM: <one short sentence naming the missing or wrong fact>

Do not reason out loud. Do not write anything before or after that line.
"""

extract_chain = make_chain(EXTRACT_TEMPLATE)
explain_chain = make_chain(EXPLAIN_TEMPLATE)
hindi_chain = make_chain(HINDI_TEMPLATE)
judge_chain = make_chain(JUDGE_TEMPLATE)


# ---------------------------------------------------------------
# Deterministic tools
# ---------------------------------------------------------------
_DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")


def extract_numbers(text):
    t = (text or "").translate(_DEVANAGARI)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return re.findall(r"\d+(?:\.\d+)?", t)


def check_numbers(source, output):
    out_nums = set(extract_numbers(output))
    src_nums = extract_numbers(source)
    missing = [n for n in src_nums if n not in out_nums]
    return src_nums, missing


def _parse_json(text):
    try:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def parse_judge(reply):
    """Return 'OK', 'PROBLEM' or 'UNCLEAR'. Contradictory replies are UNCLEAR."""
    verdicts = re.findall(r"^\s*(?:VERDICT:\s*)?(OK|PROBLEM)\b", reply or "",
                          re.MULTILINE | re.IGNORECASE)
    kinds = {v.upper() for v in verdicts}
    if kinds == {"OK"}:
        return "OK"
    if kinds == {"PROBLEM"}:
        return "PROBLEM"
    return "UNCLEAR"


# ---------------------------------------------------------------
# Graph state and nodes
# ---------------------------------------------------------------
class GraphState(TypedDict, total=False):
    clause: str
    structured: Dict[str, Any]
    english: str
    hindi: str
    verification: Dict[str, Any]
    attempt: int
    max_attempts: int
    use_judge: bool
    reflections: List[str]
    history: List[Dict[str, Any]]
    final: Dict[str, Any]


def extract_node(state):
    raw = extract_chain.invoke({"clause": state["clause"]})
    return {"structured": _parse_json(raw)}


def english_node(state):
    structured = json.dumps(state.get("structured", {}), ensure_ascii=False)
    english = explain_chain.invoke({"clause": state["clause"], "structured": structured})
    return {"english": english.strip()}


def hindi_node(state):
    refl = state.get("reflections", [])
    feedback = ""
    if refl:
        feedback = ("\nA previous attempt was rejected. Reviewer note: "
                    + refl[-1] + "\nFix this issue in the new translation.\n")
    hindi = hindi_chain.invoke({"english": state["english"], "feedback": feedback})
    return {"hindi": hindi.strip()}


def verify_node(state):
    src_nums, missing = check_numbers(state["clause"], state["hindi"])
    judge_text, judge_status, judge_ok = "", "skipped", True

    if state.get("use_judge", True):
        try:
            judge_text = judge_chain.invoke({"clause": state["clause"], "hindi": state["hindi"]})
            judge_status = parse_judge(judge_text)
            judge_ok = judge_status != "PROBLEM"   # UNCLEAR does not block
        except Exception as e:
            judge_text = f"[judge unavailable: {str(e)[:100]}]"
            judge_status, judge_ok = "unavailable", True

    passed = (len(missing) == 0) and judge_ok
    verification = {
        "source_numbers": src_nums,
        "missing_numbers": missing,
        "judge": judge_text,
        "judge_ok": judge_ok,
        "judge_status": judge_status,
        "passed": passed,
    }
    history = list(state.get("history", []))
    history.append({
        "attempt": state.get("attempt", 1),
        "missing_numbers": missing,
        "judge_status": judge_status,
        "passed": passed,
    })
    return {"verification": verification, "history": history}


def reflect_node(state):
    v = state["verification"]
    parts = []
    if v["missing_numbers"]:
        parts.append("these numbers from the source are missing in the Hindi: "
                     + ", ".join(v["missing_numbers"]))
    if v["judge_status"] == "PROBLEM":
        j = " ".join((v["judge"] or "").split())[:300]
        parts.append("reviewer said: " + j)
    attempt = state.get("attempt", 1)
    note = f"Attempt {attempt} was rejected - " + "; ".join(parts)
    refl = list(state.get("reflections", []))
    refl.append(note)
    return {"reflections": refl, "attempt": attempt + 1}


def finalize_node(state):
    v = state["verification"]
    if v["passed"]:
        status = "passed" if v["judge_status"] == "OK" else "passed_numbers_only"
    else:
        status = "failed_after_max_attempts"
    structured = state.get("structured", {}) or {}
    clause_low = state["clause"].lower()
    found = {}
    for k in ["obligation", "amount", "deadline", "condition", "consequence"]:
        val = str(structured.get(k, "")).strip().lower()
        found[k] = bool(val) and val in clause_low
    final = {
        "status": status,
        "attempts_used": state.get("attempt", 1),
        "hindi": state["hindi"],
        "english": state["english"],
        "model_used": _WORKING_MODEL,
        "field_found_verbatim_in_source": found,
    }
    return {"final": final}


def route_after_verify(state):
    if state["verification"]["passed"]:
        return "finalize_node"
    if state.get("attempt", 1) >= state.get("max_attempts", 3):
        return "finalize_node"
    return "reflect_node"


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
        "verify_node", route_after_verify,
        {"reflect_node": "reflect_node", "finalize_node": "finalize_node"},
    )
    g.add_edge("reflect_node", "hindi_node")
    g.add_edge("finalize_node", END)
    return g.compile()


SAMPLE_CLAUSE = (
    "The Tenant shall pay to the Landlord a refundable security deposit of Rs. 4,29,000 "
    "on or before the date of execution of this Agreement. The Landlord shall return the "
    "deposit within 30 days of the Tenant vacating the premises, after deducting any "
    "amounts due towards unpaid rent or damage beyond normal wear and tear."
                            )
