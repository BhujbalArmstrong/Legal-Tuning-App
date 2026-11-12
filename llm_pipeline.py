import os
import time
import concurrent.futures
import google.generativeai as genai

CALL_TIMEOUT = 60                 # seconds allowed per Gemini call
MIN_SECONDS_BETWEEN_CALLS = 6     # model allows ~15 requests/minute
PREFERRED_MODELS = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.5-flash",
    "models/gemini-3.1-flash-lite",
]
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_last_call_time = [0.0]
WORKING_MODEL = None


def _configure():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"), transport="grpc")


def _wait_for_rate_limit():
    elapsed = time.time() - _last_call_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time[0] = time.time()


def _model_order():
    order = []
    for n in [WORKING_MODEL, os.environ.get("GEMINI_MODEL")] + PREFERRED_MODELS:
        if n and n not in order:
            order.append(n)
    return order


EXTRACTION_PROMPT = """You are a legal-clause structured information extractor.
Given a clause from a rental agreement or employment offer letter, extract the fields below.
If a field does not apply, write "NONE". Do not invent information not present in the text.

Clause text:
\"\"\"{clause}\"\"\"

Return strictly in this format:
PARTY: ...
ACTION: ...
OBJECT: ...
DATE_DEADLINE: ...
QUALIFICATION: ...
REFERENCE: ...
CATEGORY: ...
"""

SIMPLIFY_PROMPT = """You are simplifying a legal clause for a layperson with no legal background.
Use the structured extraction below to write ONE short, clear plain-English sentence (max 40 words)
explaining what this clause means in practice. Do NOT add any information not present in the fields.

Structured fields:
{fields}

Plain-English explanation:
"""

TRANSLATE_PROMPT = """Translate the following plain-English legal explanation into simple, natural,
everyday Hindi that a person with basic education can understand. Avoid overly literal/formal
legal Hindi. Do not add or remove any information.

English text:
\"\"\"{text}\"\"\"

Hindi translation:
"""


def call_llm(prompt: str) -> str:
    """Calls Gemini with timeout, pacing, retry and model fallback.
    Returns an error string starting with '[' on failure (never raises)."""
    global WORKING_MODEL
    try:
        _configure()
    except Exception as e:
        return f"[LLM ERROR: setup failed: {e}]"

    errors = []
    for name in _model_order():
        for attempt in (1, 2):
            _wait_for_rate_limit()

            def _call(n=name):
                m = genai.GenerativeModel(n)
                return m.generate_content(prompt, request_options={"timeout": CALL_TIMEOUT})

            try:
                resp = _pool.submit(_call).result(timeout=CALL_TIMEOUT + 10)
                WORKING_MODEL = name
                return resp.text.strip()
            except concurrent.futures.TimeoutError:
                errors.append(f"{name}: timed out after {CALL_TIMEOUT}s")
            except Exception as e:
                errors.append(f"{name}: {str(e)[:150]}")
                msg = str(e).lower()
                if any(k in msg for k in ("429", "quota", "exhausted")):
                    break  # quota used up: next model immediately
                if any(k in msg for k in ("503", "504", "deadline", "unavailable", "timed out", "timeout")):
                    if attempt == 1:
                        time.sleep(5)
                    continue  # retry once
                break  # e.g. 404: next model
    return "[LLM ERROR: all models failed -> " + " | ".join(errors) + "]"


def extract_structured(clause_text: str) -> str:
    return call_llm(EXTRACTION_PROMPT.format(clause=clause_text))


def simplify_from_fields(fields_text: str) -> str:
    return call_llm(SIMPLIFY_PROMPT.format(fields=fields_text))


def translate_to_hindi_llm(english_text: str) -> str:
    return call_llm(TRANSLATE_PROMPT.format(text=english_text))


def run_proposed_pipeline(clause_text: str) -> dict:
    fields = extract_structured(clause_text)
    simplified_en = simplify_from_fields(fields)
    hindi = translate_to_hindi_llm(simplified_en)
    return {
        "structured_fields": fields,
        "simplified_english": simplified_en,
        "hindi_output": hindi,
    }


def run_single_prompt_baseline(clause_text: str) -> str:
    prompt = f"""Read this legal clause and directly write a simple Hindi explanation 
a layperson can understand, in one or two short sentences. Do not add information not in the clause.

Clause:
\"\"\"{clause_text}\"\"\"

Simple Hindi explanation:
"""
    return call_llm(prompt)
