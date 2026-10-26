import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.8-flash")

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
    response = model.generate_content(prompt)
    return response.text.strip()

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
