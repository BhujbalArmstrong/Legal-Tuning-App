import re
import pandas as pd
import evaluate

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

NUMBER_PATTERN = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")

def extract_numbers(text: str) -> set:
    return set(NUMBER_PATTERN.findall(text))

def number_mismatch_flag(source_text: str, output_text: str) -> dict:
    src_nums = extract_numbers(source_text)
    out_nums = extract_numbers(output_text)
    missing = src_nums - out_nums
    return {
        "source_numbers": src_nums,
        "output_numbers": out_nums,
        "missing_numbers": missing,
        "flag_mismatch": len(missing) > 0 and len(src_nums) > 0,
    }

def compute_rouge(reference: str, prediction: str) -> dict:
    result = rouge.compute(predictions=[prediction], references=[reference])
    return result

def compute_bertscore(reference: str, prediction: str, lang="en") -> dict:
    result = bertscore.compute(predictions=[prediction], references=[reference], lang=lang)
    return {"precision": result["precision"][0], "recall": result["recall"][0], "f1": result["f1"][0]}

def round_trip_consistency(original_en: str, back_translated_en: str) -> dict:
    bert = compute_bertscore(original_en, back_translated_en)
    return bert

def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces one summary row per pipeline:
    - total clauses processed
    - % of clauses with a number/date mismatch (critical error rate)
    - average back-translation consistency score
    """
    def f1_mean(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.mean()

    summary = df.groupby("Pipeline").agg(
        total_clauses=("Pipeline", "count"),
        mismatch_count=("Number mismatch flag", lambda x: (x == "⚠️ YES").sum()),
        avg_backtranslation_f1=("Back-translation (round-trip) BERTScore F1", f1_mean),
    ).reset_index()

    summary["mismatch_rate_%"] = (summary["mismatch_count"] / summary["total_clauses"] * 100).round(1)
    summary["avg_backtranslation_f1"] = summary["avg_backtranslation_f1"].round(3)

    return summary
