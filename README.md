# TalentScreener AI — v2.0

An AI-powered Resume Screening and Candidate Ranking System built with Python and Streamlit.
Upload multiple PDF resumes against a job description and get ranked, scored, and categorised candidate profiles in seconds.

---

## Features

### Core
- **Multi-resume upload** — batch process any number of PDF resumes at once
- **TF-IDF + Bigram Similarity** — cosine similarity with unigram + bigram tokenisation for better phrase matching
- **Skill Matching** — 100+ skills across 7 categories (Programming, Web, Data & ML, Database, Cloud & DevOps, Analytics, Soft Skills)
- **ATS Score** — keyword coverage (60%) + resume section detection (40%)
- **Experience Level Detection** — automatically classifies candidates as Senior / Mid-Level / Junior

### v2.0 Improvements over original
| Area | Original | v2.0 |
|------|----------|-------|
| Skills list | 25 skills | 100+ skills across 7 categories |
| NLP pipeline | lowercase + regex strip | + contraction expansion, stopword removal, tech-term protection (c++, c#, .net, node.js …) |
| TF-IDF | unigrams only | unigrams + bigrams, sublinear TF |
| Candidate comparison | ❌ | ✅ Side-by-side radar chart |
| Score breakdown | hidden | Similarity % + Skill Match % + ATS % per card |
| Skill gap heatmap | ❌ | ✅ Candidate × skill matrix |
| Score gauges | ❌ | ✅ Gauge charts per candidate |
| Category-wise skills | ❌ | ✅ Breakdown by domain |
| Experience badge | ❌ | ✅ Senior / Mid / Junior |
| "Why this score?" | ❌ | ✅ Top TF-IDF JD keywords |
| PDF robustness | single extraction | dual-pass fallback extraction |
| requirements.txt | unpinned | minimum version pins |

---

## Project Structure

```
TalentScreener-AI/
├── app.py                    # Streamlit UI (main entry point)
├── nlp_utils.py              # Text cleaning pipeline
├── matcher.py                # TF-IDF cosine similarity engine
├── skill_matcher.py          # Skill registry + extraction + experience detection
├── resume_parser.py          # PDF text + name extraction (PyMuPDF)
├── test_app.py               # Functional test suite
├── requirements.txt
├── sample_resumes/           # 5 test PDF resumes
└── sample_job_descriptions/  # 5 test JD text files
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

---

## Running Tests

```bash
python test_app.py
```

---

## How It Works

### Scoring Formula

```
Combined Score = (Similarity Score × 0.4) + (Skill Match % × 0.6)
```

### ATS Score Formula

```
ATS Score = (Matched Keywords / Total JD Keywords × 60) + (Found Sections / 5 × 40)
```

### Recommendation Thresholds

| Score | Recommendation |
|-------|---------------|
| ≥ 75% | ✅ Strong Hire |
| ≥ 55% | 🟦 Hire |
| ≥ 30% | ⚠️ Consider |
| < 30% | ❌ Not Suitable |

---

## Test Mapping

Sample files included for testing (matching and intentional mismatches to validate ranking):

| Resume | Job Description | Expected |
|--------|----------------|----------|
| Sample_Resume_1_DataScientist.pdf | jd_1_data_scientist.txt | ✅ Match |
| Sample_Resume_2_NLP_MLEngineer.pdf | jd_2_ml_engineer.txt | ✅ Match |
| Sample_Resume_3_DataAnalyst.pdf | jd_3_backend_java_developer.txt | ❌ Mismatch (intentional) |
| Sample_Resume_4_ComputerVisionIntern.pdf | jd_4_computer_vision_intern.txt | ✅ Match |
| Sample_Resume_5_JuniorPythonDeveloper.pdf | jd_5_frontend_web_developer.txt | ❌ Mismatch (intentional) |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **scikit-learn** — TF-IDF vectorisation & cosine similarity
- **PyMuPDF (fitz)** — PDF parsing
- **Plotly** — interactive charts
- **pandas** — data handling

---

## Author

Utkarsh Tripathi — PBEL 3.0
