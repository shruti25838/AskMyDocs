import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

def load_pdf(file_path):
    try:
        # First try text extraction (works for non-scanned PDFs)
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            return text

        # If no text found, fall back to OCR
        print("⚠️ No extractable text found — using OCR...")
        images = convert_from_path(file_path, dpi=300)
        text = ""
        for i, image in enumerate(images):
            ocr_text = pytesseract.image_to_string(image)
            text += f"\n\n[Page {i+1}]\n{ocr_text}"
        return text.strip()
    except Exception as e:
        return f"❌ Error loading PDF: {e}"
