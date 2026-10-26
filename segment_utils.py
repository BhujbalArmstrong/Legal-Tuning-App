import re

CLAUSE_PATTERN = re.compile(
    r"(?:^|\n)\s*((?:\d+\.\d+|\d+\.|\(\w\)|ARTICLE\s+\d+|Clause\s+\d+))",
    re.IGNORECASE
)

def segment_clauses(text: str):
    matches = list(CLAUSE_PATTERN.finditer(text))

    if not matches:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [{"clause_id": f"P{i+1}", "raw_text": p} for i, p in enumerate(paras)]

    clauses = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        clause_text = text[start:end].strip()
        clause_id = m.group(1).strip()
        if len(clause_text) > 10:
            clauses.append({"clause_id": clause_id, "raw_text": clause_text})

    return clauses
