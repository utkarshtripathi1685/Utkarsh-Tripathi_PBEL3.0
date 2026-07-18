import streamlit as st
import pandas as pd
import re
from collections import Counter
from resume_parser import extract_text_from_pdf, extract_largest_text
import plotly.graph_objects as go

from nlp_utils import clean_text
from matcher import calculate_similarity
from skill_matcher import extract_skills, analyze_resume_strength

# ================= CONSTANTS =================
STRONG_HIRE_THRESHOLD = 75
HIRE_THRESHOLD = 55
CONSIDER_THRESHOLD = 30
SIMILARITY_WEIGHT = 0.4
SKILL_WEIGHT = 0.6
MAX_RESUME_SECTIONS = 5

# ================= LOCAL FUNCTIONS =================

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"


def extract_phone(text):
    phone_pattern = r'(?:\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}|\b\d{10}\b'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not Found"


def extract_name(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    def is_valid_name_line(line):
        if '@' in line or re.search(r'\d', line) or 'http' in line.lower():
            return False
        word_count = len(line.split())
        return 2 <= word_count <= 5 and len(line) < 60

    for line in lines[:8]:
        if is_valid_name_line(line) and line.isupper():
            return line.title()
    for line in lines[:8]:
        if is_valid_name_line(line):
            return line.title()
    return "Not Found"


def get_recommendation(combined_score):
    """Return recommendation label based on combined score."""
    if combined_score >= STRONG_HIRE_THRESHOLD:
        return "Strong Hire"
    elif combined_score >= HIRE_THRESHOLD:
        return "Hire"
    elif combined_score >= CONSIDER_THRESHOLD:
        return "Consider"
    return "Not Suitable"


def render_recommendation(rec):
    """Render a color-coded recommendation badge using st alerts."""
    if rec == "Strong Hire":
        st.success(f"✅ {rec}")
    elif rec == "Hire":
        st.info(f"🟦 {rec}")
    elif rec == "Consider":
        st.warning(f"⚠️ {rec}")
    else:
        st.error(f"❌ {rec}")


def render_skill_pills(skills, color):
    """Render a list of skills as styled HTML badge pills."""
    if not skills:
        return
    bg = "#1a7a4a" if color == "green" else "#a63030"
    pills = " ".join(
        f'<span style="background:{bg};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;margin:2px;display:inline-block;">'
        f'{s}</span>'
        for s in skills
    )
    st.markdown(pills, unsafe_allow_html=True)


# ================= UI SETUP =================
st.set_page_config(layout="wide", page_title="TalentScreener AI", page_icon="🎯")

# --- SIDEBAR ---
st.sidebar.title("🎯 TalentScreener AI")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Analysis", use_container_width=True):
    for key in ["df", "analysis_done"]:
        st.session_state.pop(key, None)
    st.rerun()

if 'df' in st.session_state and st.session_state.get('analysis_done'):
    df_sidebar = st.session_state['df']
    st.sidebar.markdown("#### 📊 Quick Stats")
    st.sidebar.metric("Candidates Analyzed", len(df_sidebar))
    st.sidebar.metric("Top Score", f"{df_sidebar['Combined Score (%)'].max()}%")
    st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigate",
    ["Evaluation Console", "System Documentation"],
    label_visibility="visible"
)


# ================= PAGE 1: EVALUATION CONSOLE =================
if app_mode == "Evaluation Console":

    st.title("🎯 TalentScreener AI")
    st.caption("AI-Powered Resume Screening Platform")
    st.write("Upload Resumes  →  AI Analysis  →  Candidate Ranking  →  ATS Evaluation")
    st.markdown("---")

    col_input1, col_input2 = st.columns([1, 1], gap="large")

    with col_input1:
        st.subheader("📄 Upload Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload candidate profiles (PDF format supported)",
            type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        st.caption("Only PDF files are accepted.")

    with col_input2:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste Job Description", height=125,
            placeholder="Enter target skills, qualifications, and role responsibilities...",
            label_visibility="collapsed"
        )

    analyze = st.button("🔍 Analyze Candidates", use_container_width=True, type="primary")

    # --- RUN ANALYSIS ---
    if analyze:
        if not job_description and not uploaded_files:
            st.session_state['analysis_done'] = False
            st.error("Please provide a Job Description and upload at least one Resume.")
        elif job_description and not uploaded_files:
            st.session_state['analysis_done'] = False
            st.warning("Job Description provided, but no resumes uploaded.")
        elif not job_description and uploaded_files:
            st.session_state['analysis_done'] = False
            st.warning("Resumes uploaded, but Job Description is empty.")
        else:
            results_list = []
            clean_job = clean_text(job_description)
            job_skills = extract_skills(clean_job)

            with st.spinner("⏳ Analyzing candidates... please wait."):
                for file in uploaded_files:
                    try:
                        resume_text = extract_text_from_pdf(file)
                        candidate_name = extract_largest_text(file)

                        email = extract_email(resume_text)
                        phone = extract_phone(resume_text)
                        if not candidate_name or candidate_name.strip() == "":
                            candidate_name = extract_name(resume_text)

                        clean_resume = clean_text(resume_text)
                        score = calculate_similarity(clean_resume, clean_job)
                        match_percentage = int(round(score * 100, 2))

                        resume_skills = extract_skills(clean_resume)
                        matched_skills = list(set(resume_skills) & set(job_skills))
                        missing_skills = list(set(job_skills) - set(resume_skills))

                        skill_match_pct = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0
                        combined_score = round(
                            (match_percentage * SIMILARITY_WEIGHT) + (skill_match_pct * SKILL_WEIGHT), 2
                        )

                        recommendation = get_recommendation(combined_score)

                        # ATS Score — compute once and store
                        resume_sections_found, resume_sections_missing = analyze_resume_strength(resume_text)
                        keyword_score = (len(matched_skills) / len(job_skills) * 60) if job_skills else 0
                        section_score = (len(resume_sections_found) / MAX_RESUME_SECTIONS) * 40
                        ats_score = round(keyword_score + section_score, 2)

                        results_list.append({
                            "Candidate Name": candidate_name if candidate_name else file.name.replace(".pdf", ""),
                            "Email": email,
                            "Phone": phone,
                            "Combined Score (%)": combined_score,
                            "Matched_Skills": matched_skills,
                            "Missing_Skills": missing_skills,
                            "Recommendation": recommendation,
                            "Resume_Text": resume_text,
                            "ATS Score": ats_score,
                            "Found Sections": resume_sections_found,
                            "Missing Sections List": resume_sections_missing,
                        })
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")

            if results_list:
                df = pd.DataFrame(results_list)
                df = df.sort_values(by="Combined Score (%)", ascending=False).reset_index(drop=True)
                st.session_state['df'] = df
                st.session_state['analysis_done'] = True
            else:
                st.session_state['analysis_done'] = False
                st.error("No resumes could be processed. Please check the uploaded files.")

    # --- DISPLAY RESULTS ---
    if st.session_state.get('analysis_done'):
        df = st.session_state['df']
        best_candidate = df.iloc[0]

        # Count recommendation tiers
        strong_hire = len(df[df["Recommendation"] == "Strong Hire"])
        hire = len(df[df["Recommendation"] == "Hire"])
        consider = len(df[df["Recommendation"] == "Consider"])
        not_suitable = len(df[df["Recommendation"] == "Not Suitable"])
        recommended = strong_hire + hire

        st.markdown("---")
        st.header("📊 Executive Dashboard")

        # --- TOP CANDIDATE ---
        st.subheader("🏆 Top Candidate")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.write(f"**Name:** {best_candidate['Candidate Name']}")
        with t2:
            st.metric("Match Score", f"{best_candidate['Combined Score (%)']}%")
        with t3:
            st.write("**Recommendation:**")
            render_recommendation(best_candidate['Recommendation'])

        # --- RECOMMENDATION SUMMARY ---
        st.subheader("📋 Recommendation Summary")
        success_rate = round((recommended / len(df)) * 100, 2) if len(df) > 0 else 0
        st.progress(success_rate / 100)
        st.caption(f"Overall Selection Rate: {success_rate}%")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.success(f"✅ Strong Hire: {strong_hire}")
        with r2:
            st.info(f"🟦 Hire: {hire}")
        with r3:
            st.warning(f"⚠️ Consider: {consider}")
        with r4:
            st.error(f"❌ Not Suitable: {not_suitable}")   # BUG FIX: was st.info

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Candidate Count", len(df))
        with c2:
            st.metric("Highest Score", f"{df['Combined Score (%)'].max()}%")
        with c3:
            st.metric("Average Score", f"{round(df['Combined Score (%)'].mean(), 2)}%")
        with c4:
            st.metric("Recommended", recommended)

        st.markdown("---")

        # --- SKILL COVERAGE METRICS ---
        total_matched = sum(len(x) for x in df["Matched_Skills"])
        total_missing = sum(len(x) for x in df["Missing_Skills"])
        total_skills = total_matched + total_missing
        coverage = (total_matched / total_skills) if total_skills > 0 else 0

        col_res1, col_res2 = st.columns([11, 9], gap="large")

        # --- LEFT: CANDIDATE RANKING ---
        with col_res1:
            st.header("🏅 Candidate Ranking")
            st.caption("AI-powered ranking based on job description similarity and skill match.")

            for idx, row in df.iterrows():
                with st.container(border=True):
                    left_col, right_col = st.columns([4, 1])
                    with left_col:
                        st.subheader(f"#{idx + 1}  {row['Candidate Name']}")
                        st.write(f"**Email:** {row['Email']}")
                        st.write(f"**Phone:** {row['Phone']}")

                        st.write("**✅ Matched Skills**")
                        if row["Matched_Skills"]:
                            render_skill_pills(row["Matched_Skills"], "green")
                        else:
                            st.info("No matching skills found")

                        st.write("**❌ Missing Skills**")
                        if row["Missing_Skills"]:
                            render_skill_pills(row["Missing_Skills"], "red")
                        else:
                            st.success("No missing skills")

                    with right_col:
                        st.metric(label="Score", value=f"{row['Combined Score (%)']}%")
                        render_recommendation(row["Recommendation"])

            # --- EXPORT ---
            st.subheader("📥 Export Evaluation Data")
            df_export = df.copy()
            if 'Matched_Skills' in df_export.columns:
                df_export['Matched Skills'] = df_export['Matched_Skills'].apply(lambda x: ", ".join(x))
            if 'Missing_Skills' in df_export.columns:
                df_export['Missing Skills'] = df_export['Missing_Skills'].apply(lambda x: ", ".join(x))
            if 'Phone' in df_export.columns:
                df_export['Phone'] = df_export['Phone'].apply(
                    lambda x: f'="{x}"' if x and x != "Not Found" else x
                )
            cols_to_drop = [c for c in ["Matched_Skills", "Missing_Skills", "Resume_Text",
                                         "Found Sections", "Missing Sections List"] if c in df_export.columns]
            df_export = df_export.drop(columns=cols_to_drop)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Full Evaluation Report (CSV)",
                data=csv_data,
                file_name="TalentScreener_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # --- RIGHT: SKILL ANALYSIS ---
        with col_res2:
            st.header("🔬 Skill Analysis")

            k1, k2 = st.columns(2)
            with k1:
                st.metric("Matched Skills", total_matched)
            with k2:
                st.metric("Missing Skills", total_missing)

            st.progress(coverage)

            # Donut chart with custom colors
            fig = go.Figure(data=[go.Pie(
                labels=["Matched Skills", "Missing Skills"],
                values=[total_matched, total_missing],
                hole=0.6,
                marker=dict(colors=["#1a7a4a", "#a63030"])
            )])
            fig.update_layout(
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0),
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Overall Skill Coverage: {round(coverage * 100, 2)}%")

            # Most common matched skill
            all_skills = []
            for skills in df["Matched_Skills"]:
                if isinstance(skills, list):
                    all_skills.extend(skills)
                elif isinstance(skills, str) and skills.strip():
                    all_skills.append(skills)

            if all_skills:
                skill_counter = Counter(all_skills)
                top_skill = skill_counter.most_common(1)[0]
                st.success(f"🥇 Top Skill: **{top_skill[0]}** ({top_skill[1]} candidates)")

                # Skill frequency bar chart (top 8)
                top_skills = skill_counter.most_common(8)
                bar_fig = go.Figure(data=[go.Bar(
                    x=[s[0] for s in top_skills],
                    y=[s[1] for s in top_skills],
                    marker_color="#1a7a4a"
                )])
                bar_fig.update_layout(
                    title="Top Matched Skills",
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=220,
                    yaxis=dict(title="Candidates"),
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.info("No matched skills found.")

        st.markdown("---")

        # --- CANDIDATE DEEP DIVE ---
        st.header("🔍 Candidate Evaluation")
        st.caption("Interactive detailed analysis of any candidate.")

        candidate_labels = [
            f"{row['Candidate Name']} (#{idx + 1})" for idx, row in df.iterrows()
        ]
        selected_label = st.selectbox("Select Candidate for Detailed Analysis:", candidate_labels)

        if selected_label:
            selected_idx = candidate_labels.index(selected_label)
            cand_row = df.iloc[selected_idx]
            score = int(round(float(cand_row["Combined Score (%)"])))

            with st.container(border=True):
                st.subheader("🏷️ Hiring Decision")
                render_recommendation(cand_row["Recommendation"])
                st.metric("Resume Fitment Strength", f"{score}%")
                st.progress(score / 100)

            with st.container(border=True):
                p1, p2 = st.columns(2)
                with p1:
                    st.subheader("👤 Personal Information")
                    st.write(f"**Name:** {cand_row['Candidate Name']}")
                    st.write(f"**Email:** {cand_row['Email']}")
                    st.write(f"**Phone:** {cand_row['Phone']}")
                with p2:
                    st.subheader("📈 Evaluation Metrics")
                    st.metric("Match Score", f"{cand_row['Combined Score (%)']}%")
                    st.metric("ATS Score", f"{cand_row['ATS Score']}%")

            # BUG FIX: use pre-stored values from DataFrame — no redundant function call
            found_sections = cand_row["Found Sections"]
            missing_sections = cand_row["Missing Sections List"]
            strength_score = int((len(found_sections) / MAX_RESUME_SECTIONS) * 100)

            with st.container(border=True):
                st.subheader("📄 Resume Analysis")
                st.progress(strength_score / 100)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Resume Strength", f"{strength_score}%")
                with m2:
                    st.metric("ATS Score", f"{cand_row['ATS Score']}%")

                left, right = st.columns(2)
                with left:
                    st.write("**✅ Detected Sections**")
                    for section in found_sections:
                        st.write(f"- {section}")
                with right:
                    st.write("**🔧 Sections to Improve**")
                    if missing_sections:
                        for section in missing_sections:
                            st.write(f"- {section}")
                    else:
                        st.info("No missing sections")

            # Improvement suggestions
            suggestions = []
            if "Projects" in missing_sections:
                suggestions.append("Add at least 2 academic or personal projects.")
            if "Experience" in missing_sections:
                suggestions.append("Include internships, freelance work, or relevant experience.")
            if "Certifications" in missing_sections:
                suggestions.append("Add certifications such as NPTEL, Coursera, or IBM SkillsBuild.")
            if "Education" in missing_sections:
                suggestions.append("Clearly mention your education details.")
            if "Skills" in missing_sections:
                suggestions.append("Create a dedicated technical skills section.")

            with st.container(border=True):
                st.subheader("💡 Resume Improvement Suggestions")
                if suggestions:
                    for s in suggestions:
                        st.info(s)
                else:
                    st.success("✅ Resume looks complete. No major improvements suggested.")

            # Recommendation distribution bar chart (color-coded)
            st.caption("Overall hiring recommendation distribution")
            dist_fig = go.Figure(data=[go.Bar(
                x=["Strong Hire", "Hire", "Consider", "Not Suitable"],
                y=[strong_hire, hire, consider, not_suitable],
                marker_color=["#1a7a4a", "#1a6aaa", "#b8860b", "#a63030"]
            )])
            dist_fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=250,
                yaxis=dict(title="Candidates"),
            )
            st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander("🔎 Filter Candidates by Skill"):
                search_skill = st.text_input("Search by Skill", placeholder="Example: Python")
                if search_skill:
                    filtered_df = df[df['Matched_Skills'].apply(
                        lambda x: search_skill.lower() in [s.lower() for s in x]
                    )]
                    if not filtered_df.empty:
                        st.success(f"{len(filtered_df)} candidate(s) found with '{search_skill}'.")
                        st.dataframe(
                            filtered_df[["Candidate Name", "Combined Score (%)", "Recommendation"]],
                            use_container_width=True
                        )
                    else:
                        st.warning(f"No candidate found with skill: '{search_skill}'.")

    st.markdown("---")
    st.caption("🎯 TalentScreener AI v1.1 | Smart Resume Screening & Candidate Ranking | Python · Streamlit · NLP")


# ================= PAGE 2: SYSTEM DOCUMENTATION =================
elif app_mode == "System Documentation":
    st.title("📐 Architecture & System Documentation")
    st.caption("Technical specifications and compliance framework of TalentScreener AI")

    st.markdown("---")
    st.subheader("1. Core Processing Engine")
    st.write(
        "The system utilizes a multi-stage Natural Language Processing (NLP) pipeline "
        "to clean text data, remove noisy components, and structure unstructured resume "
        "textual fields."
    )

    st.subheader("2. Vectorization & Similarity Metrics")
    st.write(
        "Candidate profiles are vectorized using TF-IDF and compared against the job "
        "description using cosine similarity to generate normalized match scores. "
        "A weighted hybrid score combines semantic similarity (40%) and skill match (60%)."
    )

    st.subheader("3. ATS Score Computation")
    st.write(
        "ATS Score = Keyword Match Score (60%) + Resume Section Coverage Score (40%). "
        "Sections checked: Education, Experience, Projects, Certifications, Skills."
    )

    st.subheader("4. Enterprise Security & Privacy")
    st.write(
        "All uploads are parsed entirely in volatile runtime memory. No persistent "
        "storage systems or external logs retain any personal data records."
    )
