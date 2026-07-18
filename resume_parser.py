"""
resume_parser.py
----------------
Utilities for extracting text and candidate name from PDF resumes
using PyMuPDF (fitz).
"""

import fitz


def extract_text_from_pdf(pdf_file) -> str:
    """
    Reads all pages of a PDF upload and returns the full extracted text.
    Resets the file pointer before reading to prevent empty-read bugs.
    """
    pdf_file.seek(0)  # BUG FIX: reset pointer before reading
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text("text", sort=True)
    pdf_document.close()
    return text


def extract_largest_text(pdf_file) -> str:
    """
    Scans the first page of a PDF and returns the text span with the
    largest font size — typically the candidate name in a resume.
    Resets the file pointer before reading.
    """
    pdf_file.seek(0)  # Reset pointer before second read
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = pdf_document[0]
    blocks = page.get_text("dict")["blocks"]

    max_size = 0
    candidate_text = ""

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                if span["size"] > max_size and len(text.split()) <= 5:
                    max_size = span["size"]
                    candidate_text = text

    pdf_document.close()
    return candidate_text.strip()
