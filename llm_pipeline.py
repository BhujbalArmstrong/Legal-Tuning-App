import os
import re
import time
import concurrent.futures
import google.generativeai as genai

CALL_TIMEOUT = 90                 # seconds allowed per Gemini call
LIST_TIMEOUT = 20
MIN_SECONDS_BETWEEN_CALLS = 13    # free tier ~5 requests/minute
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_last_call_time = [0.0]
WORKING_MODEL = None


def _configure():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"), transport="rest")


def _wait_for_rate_limit():
    elapsed = time.time() - _last_call_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time[0] = time.time()


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _candidate_models():
    """Flash models available to this key, newest version first (max 4)."""
    _configure()

    def _list():
        return [
            m.name for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]

    try:
        names = _pool.submit(_list).result(timeout=LIST_TIMEOUT)
    except Exception:
        return ["models/gemini-flash-latest"]

    flash = [
        n for n in names
        if "flash" in n
        and not any(x in n for x in ("exp", "preview", "lite", "thinking", "image", "tts", "live", "audio"))
    ]
    flash.sort(key=_version, reverse=True)
    return (flash or names)[:4]


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


def call_llm(prompt: str, max_retries: int = 2) -> str:
    """Calls Gemini with timeout, pacing, retry, and model fallback.
    Returns an error string starting with '[' on failure (never raises)."""
    global WORKING_MODEL
    try:
        _configure()
        if WORKING_MODEL:
            candidates = [WORKING_MODEL]
        else:
            override = os.environ.get("GEMINI_MODEL")
            candidates = ([override] if override else []) + _candidate_models()
    except Exception as e:
        return f"[LLM ERROR: setup failed: {e}]"

    errors = []
    for name in candidates:
        for attempt in range(max_retries):
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
                retryable = any(k in msg for k in ("429", "quota", "exhausted", "timed out", "timeout"))
                if not retryable:
                    break  # e.g. 404: next model
                if attempt < max_retries - 1:
                    time.sleep(15 * (attempt + 1))
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
