"""
nlp_utils.py
------------
Text preprocessing utilities for TalentScreener AI.

Improvements over original:
- Stopword removal using a curated list (no external NLTK download required)
- Contraction expansion (e.g. "don't" → "do not")
- Preserves tech keywords like "c++", "c#" that plain regex would destroy
- Normalises whitespace and casing
"""

import re

# Common English stopwords (curated subset — avoids NLTK download dependency)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "we", "our", "you",
    "your", "he", "his", "she", "her", "they", "their", "i", "my",
    "us", "me", "him", "them", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "any", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "same", "so", "than", "too", "very", "just", "as", "also",
}

# Basic English contractions
_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "can't": "cannot", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "you're": "you are", "you've": "you have", "you'll": "you will",
}

# Tech terms with special chars that must survive cleaning.
# Tokens must contain only lowercase letters, digits, and underscores
# so they pass the [^a-z0-9_ ] regex filter unchanged.
_PROTECTED_TERMS = {
    "c++":          "xxtokenxcplusplus",
    "c#":           "xxtokenxcsharp",
    ".net":         "xxtokenxdotnet",
    "node.js":      "xxtokenxnodejs",
    "vue.js":       "xxtokenxvuejs",
    "react.js":     "xxtokenxreactjs",
    "next.js":      "xxtokenxnextjs",
    "express.js":   "xxtokenxexpressjs",
    "scikit-learn": "xxtokenxsklearn",
    "t-sql":        "xxtokenxtsql",
    "pl/sql":       "xxtokenxplsql",
    "ci/cd":        "xxtokenxcicd",
    "asp.net":      "xxtokenxaspnet",
}
_RESTORE_TERMS = {v: k for k, v in _PROTECTED_TERMS.items()}


def _protect_tech_terms(text: str) -> str:
    """Replace special tech terms with safe tokens before regex cleaning."""
    for term, token in _PROTECTED_TERMS.items():
        text = text.replace(term, token)
    return text


def _restore_tech_terms(text: str) -> str:
    """Restore tech term tokens back to their original form."""
    for token, term in _RESTORE_TERMS.items():
        text = text.replace(token, term)
    return text


def _expand_contractions(text: str) -> str:
    """Replace contractions with their expanded forms."""
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def remove_stopwords(text: str) -> str:
    """Remove common stopwords from a lowercased, space-separated text string."""
    tokens = text.split()
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(filtered)


def clean_text(text: str, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Protect special tech terms (c++, c#, .net, …)
      3. Expand contractions
      4. Remove non-alphanumeric characters (except spaces)
      5. Collapse whitespace
      6. Optionally remove stopwords
      7. Restore protected tech terms

    Parameters
    ----------
    text : str
        Raw input text (resume or job description).
    remove_stops : bool
        Set to False to skip stopword removal (useful for short snippets).

    Returns
    -------
    str
        Cleaned, normalised text string.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _protect_tech_terms(text)
    text = _expand_contractions(text)

    # Replace non-alphanumeric (preserve spaces and the __ tokens)
    text = re.sub(r'[^a-z0-9_ ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stops:
        text = remove_stopwords(text)

    text = _restore_tech_terms(text)
    return text.strip()
