"""
resume_parser.py
----------------
Utilities for extracting text and candidate name from PDF resumes
using PyMuPDF (fitz).

Improvements over original:
- extract_text_from_pdf() falls back to raw dict extraction when
  standard "text" mode returns empty content (handles scanned/OCR PDFs
  or atypical PDF layouts gracefully)
- extract_largest_text() now ignores lines that look like URLs, emails,
  phone numbers, or consist only of digits — reduces mis-identification
  of the candidate name
- Both functions handle corrupt/password-protected PDFs with a clear error
"""

import re
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_name(text: str) -> bool:
    """
    Return True if `text` looks like a human name.
    Rejects lines containing URLs, emails, digits, or single words.
    """
    if not text or len(text.strip()) < 3:
        return False
    t = text.strip()
    # Reject if contains email / URL / phone
    if re.search(r'[@:/\\]', t):
        return False
    # Reject if mostly digits
    if re.search(r'\d', t):
        return False
    # Must be 2–5 words
    words = t.split()
    return 2 <= len(words) <= 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str:
    """
    Read all pages of a PDF upload and return the full extracted text.

    Strategy:
      1. Try PyMuPDF's standard text extraction (sort=True).
      2. If that yields nothing, fall back to dict-based span extraction
         (handles some unusual PDF layouts).

    Resets the file pointer before reading to prevent empty-read bugs.

    Parameters
    ----------
    pdf_file : file-like object
        Streamlit UploadedFile or any seekable binary stream.

    Returns
    -------
    str
        Full text content of the PDF.

    Raises
    ------
    ValueError
        If the PDF is password-protected or cannot be opened.
    """
    pdf_file.seek(0)
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    if pdf_document.is_encrypted:
        pdf_document.close()
        raise ValueError("PDF is password-protected and cannot be processed.")

    text_parts = []
    for page in pdf_document:
        page_text = page.get_text("text", sort=True).strip()
        if page_text:
            text_parts.append(page_text)
        else:
            # Fallback: collect spans from dict extraction
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            text_parts.append(span_text)

    pdf_document.close()
    return "\n".join(text_parts)


def extract_largest_text(pdf_file) -> str:
    """
    Scan the first page of a PDF and return the text span with the
    largest font size that looks like a human name.

    Falls back to the raw largest span if no clean name is found.
    Resets the file pointer before reading.

    Parameters
    ----------
    pdf_file : file-like object

    Returns
    -------
    str
        Candidate name (best-effort) or empty string.
    """
    pdf_file.seek(0)
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception:
        return ""

    if pdf_document.is_encrypted or len(pdf_document) == 0:
        pdf_document.close()
        return ""

    page = pdf_document[0]
    blocks = page.get_text("dict").get("blocks", [])

    candidates: list[tuple[float, str]] = []  # (font_size, text)

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0:
                    candidates.append((size, text))

    pdf_document.close()

    if not candidates:
        return ""

    # Sort by font size descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Prefer the largest span that passes the name heuristic
    for _, text in candidates[:10]:
        if _looks_like_name(text):
            return text.strip().title()

    # Fallback: return the largest span regardless
    return candidates[0][1].strip().title()
