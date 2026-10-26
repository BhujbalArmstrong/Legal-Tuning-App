import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile, os

def ocr_from_image(image_file) -> str:
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img, lang="eng")
    return text

def ocr_from_pdf(pdf_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    pages = convert_from_path(tmp_path, dpi=300)
    full_text = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        full_text.append(f"\n--- Page {i+1} ---\n{text}")

    os.remove(tmp_path)
    return "\n".join(full_text)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return ocr_from_pdf(uploaded_file)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        return ocr_from_image(uploaded_file)
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type. Use PDF, PNG, JPG, or TXT.")
