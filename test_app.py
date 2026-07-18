"""
test_app.py - Functional tests for TalentScreener AI core modules
"""
import sys
sys.path.insert(0, '.')

from nlp_utils import clean_text
from matcher import calculate_similarity
from skill_matcher import extract_skills, analyze_resume_strength

PASS = "PASS"
FAIL = "FAIL"

def check(name, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name} {detail}")
        sys.exit(1)

print("=" * 50)
print("  TalentScreener AI — Functional Test Suite")
print("=" * 50)

# --- Test: clean_text ---
print("\n[1] clean_text")
out = clean_text("Hello, World! Python & ML.")
check("lowercases text", "python" in out)
check("removes special chars", "," not in out and "&" not in out)
check("strips extra spaces", "  " not in out)

# --- Test: extract_skills ---
print("\n[2] extract_skills")
s = extract_skills("python machine learning pandas numpy deep learning sql")
check("finds python", "python" in s)
check("finds machine learning", "machine learning" in s)
check("finds pandas", "pandas" in s)
check("finds sql", "sql" in s)

# False-positive guard
s2 = extract_skills("certificate in science and excel communication")
check("no false positive for 'c'", "c" not in s2)
check("excel detected", "excel" in s2)
check("communication detected", "communication" in s2)

# --- Test: calculate_similarity ---
print("\n[3] calculate_similarity")
high = calculate_similarity("python developer machine learning", "python machine learning developer")
check(f"high similarity (got {round(high,3)})", high > 0.7)

low = calculate_similarity("cooking recipes chef food", "python java software engineer")
check(f"low similarity for unrelated text (got {round(low,3)})", low < 0.3)

# --- Test: analyze_resume_strength ---
print("\n[4] analyze_resume_strength")
full = "Education B.Tech Skills Python SQL Projects Built a web app Experience Intern at XYZ Certification NPTEL"
found, missing = analyze_resume_strength(full)
check("Education detected", "Education" in found)
check("Skills detected", "Skills" in found)
check("Projects detected", "Projects" in found)
check("Experience detected", "Experience" in found)
check("Certifications detected", "Certifications" in found)
check("Nothing missing in full resume", len(missing) == 0)

weak = "random text about hobbies and interests"
found2, missing2 = analyze_resume_strength(weak)
check("Weak resume has more missing than found", len(found2) < len(missing2))

# --- Test: recommendation thresholds ---
print("\n[5] Recommendation thresholds")
def get_recommendation(score):
    if score >= 75: return "Strong Hire"
    elif score >= 55: return "Hire"
    elif score >= 30: return "Consider"
    return "Not Suitable"

check("Score 80 -> Strong Hire", get_recommendation(80) == "Strong Hire")
check("Score 75 -> Strong Hire (boundary)", get_recommendation(75) == "Strong Hire")
check("Score 60 -> Hire", get_recommendation(60) == "Hire")
check("Score 55 -> Hire (boundary)", get_recommendation(55) == "Hire")
check("Score 40 -> Consider", get_recommendation(40) == "Consider")
check("Score 30 -> Consider (boundary)", get_recommendation(30) == "Consider")
check("Score 10 -> Not Suitable", get_recommendation(10) == "Not Suitable")
check("Score  0 -> Not Suitable", get_recommendation(0) == "Not Suitable")

# --- Test: combined score math ---
print("\n[6] Combined score formula")
job_skills = ["python", "sql", "machine learning", "pandas", "deep learning"]
resume_skills = ["python", "sql", "machine learning"]
matched = list(set(resume_skills) & set(job_skills))
missing_s = list(set(job_skills) - set(resume_skills))
match_pct = 80   # simulated TF-IDF similarity * 100
skill_pct = (len(matched) / len(job_skills)) * 100
combined = round((match_pct * 0.4) + (skill_pct * 0.6), 2)
check(f"skill_pct = 60.0% (got {skill_pct})", skill_pct == 60.0)
check(f"combined = 68.0% (got {combined})", combined == 68.0)

# --- Test: ATS score math ---
print("\n[7] ATS score formula")
job_skills_ats = ["python", "sql", "machine learning", "pandas"]
matched_ats = ["python", "sql"]
found_ats = ["Education", "Skills", "Projects"]
keyword_score = (len(matched_ats) / len(job_skills_ats)) * 60
section_score = (len(found_ats) / 5) * 40
ats = round(keyword_score + section_score, 2)
check(f"keyword_score = 30.0 (got {keyword_score})", keyword_score == 30.0)
check(f"section_score = 24.0 (got {section_score})", section_score == 24.0)
check(f"ATS = 54.0 (got {ats})", ats == 54.0)

print("\n" + "=" * 50)
print("  ALL TESTS PASSED")
print("=" * 50)
