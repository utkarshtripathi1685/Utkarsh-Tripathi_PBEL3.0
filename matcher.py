"""
matcher.py
----------
Text similarity engine for TalentScreener AI.

Improvements over original:
- calculate_similarity() is unchanged (stable public API)
- Added calculate_similarity_with_details() for richer scoring breakdown
- TF-IDF vectoriser now uses bigrams (ngram_range=(1,2)) for better phrase
  matching (e.g. "machine learning", "deep learning" score more accurately)
- sublinear_tf=True reduces the dominance of very common terms
- min_df=1, max_df=0.95 to drop near-universal terms
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cosine


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # log-normalise term frequency
        min_df=1,
        max_df=1.0,           # keep all terms — only 2 docs so 0.95 kills shared terms
        strip_accents="unicode",
    )


def calculate_similarity(resume: str, job: str) -> float:
    """
    Return the cosine similarity (0–1) between a resume and job description.

    This is the original public API — callers are unchanged.
    """
    if not resume.strip() or not job.strip():
        return 0.0

    vectorizer = _build_vectorizer()
    vectors = vectorizer.fit_transform([resume, job])
    score = _cosine(vectors[0], vectors[1])
    return float(score[0][0])


def calculate_similarity_with_details(resume: str, job: str) -> dict:
    """
    Extended version of calculate_similarity() that also returns the top
    keywords driving the score — useful for the UI's "Why this score?" panel.

    Returns
    -------
    dict with keys:
        score      : float  (0–1 cosine similarity)
        top_terms  : list[str]  (top 10 TF-IDF terms from the job description)
    """
    if not resume.strip() or not job.strip():
        return {"score": 0.0, "top_terms": []}

    vectorizer = _build_vectorizer()
    vectors = vectorizer.fit_transform([resume, job])
    score = float(_cosine(vectors[0], vectors[1])[0][0])

    # Extract the top terms from the job description vector
    feature_names = vectorizer.get_feature_names_out()
    job_vector = vectors[1].toarray()[0]

    # Sort indices by weight descending, take top 10
    top_indices = job_vector.argsort()[::-1][:10]
    top_terms = [feature_names[i] for i in top_indices if job_vector[i] > 0]

    return {"score": score, "top_terms": top_terms}
