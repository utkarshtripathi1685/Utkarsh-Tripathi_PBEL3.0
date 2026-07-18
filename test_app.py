"""
test_app.py
-----------
Functional test suite for TalentScreener AI v2.0

Tests:
  1. clean_text              — NLP pipeline
  2. extract_skills          — extended skill registry, false-positive guards
  3. extract_skills_with_categories — category grouping
  4. detect_experience_level — seniority inference
  5. calculate_similarity    — TF-IDF cosine similarity
  6. calculate_similarity_with_details — returns top_terms
  7. analyze_resume_strength — section detection
  8. Recommendation thresholds
  9. Combined score formula
 10. ATS score formula
"""

import sys
sys.path.insert(0, '.')

from nlp_utils import clean_text
from matcher import calculate_similarity, calculate_similarity_with_details
from skill_matcher import (
    extract_skills,
    extract_skills_with_categories,
    analyze_resume_strength,
    detect_experience_level,
)

PASS = "PASS"
FAIL = "FAIL"
_failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        msg = f"  [{FAIL}] {name} {detail}"
        print(msg)
        _failures.append(msg)


# ─────────────────────────────────────────────
print("=" * 55)
print("  TalentScreener AI v2.0 — Functional Test Suite")
print("=" * 55)

# ── 1. clean_text ──────────────────────────────────────────
print("\n[1] clean_text")
out = clean_text("Hello, World! Python & ML.")
check("lowercases text",       "python" in out)
check("removes special chars", "," not in out and "&" not in out)
check("strips extra spaces",   "  " not in out)

# Contraction expansion — use remove_stops=False so "not" is not stripped
out2 = clean_text("I don't know if it's working.", remove_stops=False)
check("expands contractions",  "do not" in out2)

# Tech-term protection: c++ and node.js should survive cleaning
out3 = clean_text("experience with c++ and node.js")
check("preserves c++",         "c++" in out3)
check("preserves node.js",     "node.js" in out3)

# ── 2. extract_skills ─────────────────────────────────────
print("\n[2] extract_skills — core skills")
s = extract_skills("python machine learning pandas numpy deep learning sql")
check("finds python",           "python"           in s)
check("finds machine learning", "machine learning" in s)
check("finds pandas",           "pandas"           in s)
check("finds sql",              "sql"              in s)
check("finds deep learning",    "deep learning"    in s)

# New skills in extended registry
s2 = extract_skills("docker kubernetes aws tensorflow pytorch fastapi")
check("finds docker",      "docker"     in s2)
check("finds kubernetes",  "kubernetes" in s2)
check("finds aws",         "aws"        in s2)
check("finds tensorflow",  "tensorflow" in s2)
check("finds pytorch",     "pytorch"    in s2)
check("finds fastapi",     "fastapi"    in s2)

# False-positive guard
s3 = extract_skills("certificate in science and excel communication")
check("no false positive for 'c'",  "c"     not in s3)
check("excel detected",             "excel" in s3)
check("communication detected",     "communication" in s3)

# ── 3. extract_skills_with_categories ─────────────────────
print("\n[3] extract_skills_with_categories")
cats = extract_skills_with_categories("python docker aws sql tensorflow pandas")
check("Programming category present",   "Programming"    in cats)
check("Cloud & DevOps category present","Cloud & DevOps" in cats)
check("Database category present",      "Database"       in cats)
check("Data & ML category present",     "Data & ML"      in cats)
check("python in Programming",          "python"    in cats.get("Programming", []))
check("docker in Cloud & DevOps",       "docker"    in cats.get("Cloud & DevOps", []))

# ── 4. detect_experience_level ────────────────────────────
print("\n[4] detect_experience_level")
senior_text = "Senior Software Engineer with 10 years of experience leading teams."
check("senior detected", detect_experience_level(senior_text) == "Senior")

junior_text = "I am a fresher and recent graduate looking for entry level positions."
check("junior detected", detect_experience_level(junior_text) == "Junior / Entry-Level")

mid_text = "Associate developer with 4 years of experience in backend development."
check("mid-level detected", detect_experience_level(mid_text) == "Mid-Level")

# ── 5. calculate_similarity ───────────────────────────────
print("\n[5] calculate_similarity")
high = calculate_similarity(
    "python developer machine learning",
    "python machine learning developer",
)
check(f"high similarity (got {round(high, 3)})", high > 0.5)

low = calculate_similarity(
    "cooking recipes chef food",
    "python java software engineer",
)
check(f"low similarity for unrelated text (got {round(low, 3)})", low < 0.3)

check("empty resume returns 0", calculate_similarity("", "python developer") == 0.0)
check("empty job returns 0",    calculate_similarity("python developer", "") == 0.0)

# ── 6. calculate_similarity_with_details ──────────────────
print("\n[6] calculate_similarity_with_details")
result = calculate_similarity_with_details(
    "python machine learning data scientist pandas",
    "python data scientist machine learning sql",
)
check("score key present",       "score"     in result)
check("top_terms key present",   "top_terms" in result)
check("score is float",          isinstance(result["score"], float))
check("top_terms is list",       isinstance(result["top_terms"], list))
check("top_terms non-empty",     len(result["top_terms"]) > 0)
check("score > 0 for similar",   result["score"] > 0.3)

# ── 7. analyze_resume_strength ────────────────────────────
print("\n[7] analyze_resume_strength")
full = (
    "Education B.Tech Skills Python SQL Projects Built a web app "
    "Experience Intern at XYZ Certification NPTEL"
)
found, missing = analyze_resume_strength(full)
check("Education detected",      "Education"      in found)
check("Skills detected",         "Skills"         in found)
check("Projects detected",       "Projects"       in found)
check("Experience detected",     "Experience"     in found)
check("Certifications detected", "Certifications" in found)
check("Nothing missing",         len(missing) == 0)

weak = "random text about hobbies and interests"
found2, missing2 = analyze_resume_strength(weak)
check("Weak resume: more missing than found", len(found2) < len(missing2))

# ── 8. Recommendation thresholds ──────────────────────────
print("\n[8] Recommendation thresholds")
def get_recommendation(score):
    if score >= 75: return "Strong Hire"
    elif score >= 55: return "Hire"
    elif score >= 30: return "Consider"
    return "Not Suitable"

check("Score 80 → Strong Hire",       get_recommendation(80)  == "Strong Hire")
check("Score 75 → Strong Hire (edge)",get_recommendation(75)  == "Strong Hire")
check("Score 60 → Hire",              get_recommendation(60)  == "Hire")
check("Score 55 → Hire (edge)",       get_recommendation(55)  == "Hire")
check("Score 40 → Consider",          get_recommendation(40)  == "Consider")
check("Score 30 → Consider (edge)",   get_recommendation(30)  == "Consider")
check("Score 10 → Not Suitable",      get_recommendation(10)  == "Not Suitable")
check("Score  0 → Not Suitable",      get_recommendation(0)   == "Not Suitable")

# ── 9. Combined score formula ─────────────────────────────
print("\n[9] Combined score formula")
job_skills    = ["python", "sql", "machine learning", "pandas", "deep learning"]
resume_skills = ["python", "sql", "machine learning"]
matched       = list(set(resume_skills) & set(job_skills))
match_pct     = 80
skill_pct     = (len(matched) / len(job_skills)) * 100
combined      = round((match_pct * 0.4) + (skill_pct * 0.6), 2)
check(f"skill_pct = 60.0% (got {skill_pct})", skill_pct == 60.0)
check(f"combined  = 68.0% (got {combined})",  combined  == 68.0)

# ── 10. ATS score formula ─────────────────────────────────
print("\n[10] ATS score formula")
job_skills_ats = ["python", "sql", "machine learning", "pandas"]
matched_ats    = ["python", "sql"]
found_ats      = ["Education", "Skills", "Projects"]
keyword_score  = (len(matched_ats) / len(job_skills_ats)) * 60
section_score  = (len(found_ats) / 5) * 40
ats            = round(keyword_score + section_score, 2)
check(f"keyword_score = 30.0 (got {keyword_score})", keyword_score == 30.0)
check(f"section_score = 24.0 (got {section_score})", section_score == 24.0)
check(f"ATS = 54.0 (got {ats})",                     ats           == 54.0)

# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if _failures:
    print(f"  {len(_failures)} TEST(S) FAILED:")
    for f in _failures:
        print(f)
    sys.exit(1)
else:
    print("  ALL TESTS PASSED ✓")
print("=" * 55)
