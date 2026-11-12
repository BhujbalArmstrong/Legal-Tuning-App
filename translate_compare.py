import concurrent.futures
from deep_translator import GoogleTranslator, MyMemoryTranslator

TIMEOUT_SECONDS = 30
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def _run(fn, label: str) -> str:
    fut = _pool.submit(fn)
    try:
        return fut.result(timeout=TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        return f"[{label} ERROR: timed out after {TIMEOUT_SECONDS}s]"
    except Exception as e:
        return f"[{label} ERROR: {e}]"


def translate_google(text: str, target="hi") -> str:
    return _run(lambda: GoogleTranslator(source="en", target=target).translate(text),
                "GoogleTranslator")


def translate_mymemory(text: str, target="hi") -> str:
    return _run(lambda: MyMemoryTranslator(source="en-GB", target="hi-IN").translate(text),
                "MyMemory")


def back_translate(hindi_text: str) -> str:
    return _run(lambda: MyMemoryTranslator(source="hi-IN", target="en-GB").translate(hindi_text),
                "BackTranslate")
