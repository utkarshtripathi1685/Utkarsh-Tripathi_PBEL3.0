"""
skill_matcher.py
----------------
Skill extraction and resume strength analysis for TalentScreener AI.

Improvements over original:
- Skills list expanded from 25 → 100+ covering more domains
- Skills grouped by category for richer analysis
- extract_skills_with_categories() returns category-wise breakdown
- detect_experience_level() infers seniority from resume text
- analyze_resume_strength() unchanged API, but keyword lists improved
"""

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Master skill registry  (skill_name → category)
# ---------------------------------------------------------------------------
SKILL_REGISTRY: Dict[str, str] = {
    # --- Programming Languages ---
    "python": "Programming",
    "java": "Programming",
    "c": "Programming",
    "c++": "Programming",
    "c#": "Programming",
    "r": "Programming",
    "scala": "Programming",
    "go": "Programming",
    "rust": "Programming",
    "kotlin": "Programming",
    "swift": "Programming",
    "php": "Programming",
    "ruby": "Programming",
    "typescript": "Programming",
    "javascript": "Programming",
    "bash": "Programming",
    "shell": "Programming",

    # --- Web / Frontend ---
    "html": "Web",
    "css": "Web",
    "react": "Web",
    "react.js": "Web",
    "angular": "Web",
    "vue.js": "Web",
    "next.js": "Web",
    "express.js": "Web",
    "node.js": "Web",
    "bootstrap": "Web",
    "tailwind": "Web",
    "sass": "Web",
    "rest api": "Web",
    "graphql": "Web",
    "django": "Web",
    "flask": "Web",
    "fastapi": "Web",
    "spring boot": "Web",
    "asp.net": "Web",

    # --- Data / ML ---
    "pandas": "Data & ML",
    "numpy": "Data & ML",
    "matplotlib": "Data & ML",
    "seaborn": "Data & ML",
    "scikit-learn": "Data & ML",
    "machine learning": "Data & ML",
    "deep learning": "Data & ML",
    "nlp": "Data & ML",
    "natural language processing": "Data & ML",
    "tensorflow": "Data & ML",
    "keras": "Data & ML",
    "pytorch": "Data & ML",
    "opencv": "Data & ML",
    "computer vision": "Data & ML",
    "data analysis": "Data & ML",
    "data visualization": "Data & ML",
    "feature engineering": "Data & ML",
    "model deployment": "Data & ML",
    "xgboost": "Data & ML",
    "lightgbm": "Data & ML",
    "hugging face": "Data & ML",
    "bert": "Data & ML",
    "llm": "Data & ML",
    "reinforcement learning": "Data & ML",
    "time series": "Data & ML",

    # --- Databases ---
    "sql": "Database",
    "mysql": "Database",
    "postgresql": "Database",
    "mongodb": "Database",
    "redis": "Database",
    "sqlite": "Database",
    "oracle": "Database",
    "cassandra": "Database",
    "elasticsearch": "Database",
    "pl/sql": "Database",
    "t-sql": "Database",

    # --- Cloud / DevOps ---
    "aws": "Cloud & DevOps",
    "azure": "Cloud & DevOps",
    "gcp": "Cloud & DevOps",
    "docker": "Cloud & DevOps",
    "kubernetes": "Cloud & DevOps",
    "ci/cd": "Cloud & DevOps",
    "jenkins": "Cloud & DevOps",
    "terraform": "Cloud & DevOps",
    "ansible": "Cloud & DevOps",
    "linux": "Cloud & DevOps",
    "git": "Cloud & DevOps",
    "github": "Cloud & DevOps",
    "gitlab": "Cloud & DevOps",

    # --- Analytics Tools ---
    "excel": "Analytics",
    "power bi": "Analytics",
    "tableau": "Analytics",
    "looker": "Analytics",
    "google analytics": "Analytics",
    "spark": "Analytics",
    "hadoop": "Analytics",
    "airflow": "Analytics",
    "dbt": "Analytics",

    # --- Soft Skills ---
    "communication": "Soft Skills",
    "teamwork": "Soft Skills",
    "leadership": "Soft Skills",
    "problem solving": "Soft Skills",
    "critical thinking": "Soft Skills",
    "time management": "Soft Skills",
    "project management": "Soft Skills",
    "agile": "Soft Skills",
    "scrum": "Soft Skills",
    "jira": "Soft Skills",
}

# Sorted by descending length so multi-word phrases match before single words
_SORTED_SKILLS = sorted(SKILL_REGISTRY.keys(), key=len, reverse=True)


def _build_pattern(skill: str) -> re.Pattern:
    """Compile a word-boundary regex for a skill (handles c++, c#, etc.)."""
    return re.compile(
        r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])',
        re.IGNORECASE,
    )


# Pre-compile all patterns once at import time for performance
_SKILL_PATTERNS: Dict[str, re.Pattern] = {
    skill: _build_pattern(skill) for skill in _SORTED_SKILLS
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """
    Extract a flat, sorted list of recognised skills from `text`.

    Uses whole-word / whole-phrase regex matching to avoid false positives
    (e.g. "c" matching inside "science" or "certificate").
    """
    text_lower = text.lower()
    found: List[str] = []
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_skills_with_categories(text: str) -> Dict[str, List[str]]:
    """
    Return skills grouped by category dict, e.g.:
        {"Programming": ["python", "java"], "Data & ML": ["pandas", ...], ...}
    """
    found = extract_skills(text)
    grouped: Dict[str, List[str]] = {}
    for skill in found:
        category = SKILL_REGISTRY.get(skill, "Other")
        grouped.setdefault(category, []).append(skill)
    return grouped


def detect_experience_level(resume_text: str) -> str:
    """
    Infer candidate seniority from resume text keywords.

    Returns one of: "Senior", "Mid-Level", "Junior / Entry-Level"
    """
    text = resume_text.lower()

    senior_signals = [
        "senior", "lead", "principal", "staff", "architect",
        "head of", "manager", "director", "vp", "vice president",
        "10 years", "8 years", "7 years",
    ]
    mid_signals = [
        "3 years", "4 years", "5 years", "6 years",
        "mid-level", "mid level", "associate",
    ]
    junior_signals = [
        "fresher", "entry level", "entry-level", "intern",
        "trainee", "junior", "graduate", "recent graduate",
        "0-1 year", "1 year", "2 years",
    ]

    senior_score = sum(1 for kw in senior_signals if kw in text)
    mid_score    = sum(1 for kw in mid_signals if kw in text)
    junior_score = sum(1 for kw in junior_signals if kw in text)

    if senior_score > 0 and senior_score >= mid_score:
        return "Senior"
    if mid_score > 0:
        return "Mid-Level"
    if junior_score > 0:
        return "Junior / Entry-Level"
    return "Not Specified"


def analyze_resume_strength(resume_text: str) -> Tuple[List[str], List[str]]:
    """
    Detect which standard resume sections are present or missing.

    Returns
    -------
    found   : list of section names that were detected
    missing : list of section names that were NOT detected
    """
    text = resume_text.lower()

    sections = {
        "Education": [
            "education", "b.tech", "btech", "bachelor",
            "master", "degree", "university", "college", "b.e", "mba",
        ],
        "Experience": [
            "experience", "worked", "employment",
            "company", "internship", "intern", "work history",
        ],
        "Projects": [
            "project", "projects", "developed",
            "implemented", "built", "deployed",
        ],
        "Certifications": [
            "certificate", "certification",
            "certified", "coursera", "nptel", "udemy", "linkedin learning",
        ],
        "Skills": [
            "skills", "python", "java", "sql",
            "machine learning", "excel", "power bi", "technical skills",
        ],
    }

    found: List[str] = []
    missing: List[str] = []

    for section, keywords in sections.items():
        if any(keyword in text for keyword in keywords):
            found.append(section)
        else:
            missing.append(section)

    return found, missing
