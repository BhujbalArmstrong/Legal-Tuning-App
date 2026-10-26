from deep_translator import GoogleTranslator, MyMemoryTranslator

def translate_google(text: str, target="hi") -> str:
    try:
        return GoogleTranslator(source="en", target=target).translate(text)
    except Exception as e:
        return f"[GoogleTranslator ERROR: {e}]"

def translate_mymemory(text: str, target="hi") -> str:
    try:
        return MyMemoryTranslator(source="en-GB", target="hi-IN").translate(text)
    except Exception as e:
        return f"[MyMemory ERROR: {e}]"

def back_translate(hindi_text: str) -> str:
    try:
        return GoogleTranslator(source="hi", target="en").translate(hindi_text)
    except Exception as e:
        return f"[BackTranslate ERROR: {e}]"
