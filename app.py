"""
app.py
------
TalentScreener AI — main Streamlit application.

Improvements over original:
- Score breakdown radar/gauge chart per candidate
- Category-wise skill analysis using SKILL_REGISTRY
- Experience level badge (Senior / Mid-Level / Junior)
- Side-by-side candidate comparison panel
- Skill gap heatmap across all candidates
- "Why this score?" expandable section with top TF-IDF terms
- Cleaner UI layout with consistent card styling
- Version bumped to v2.0
"""

import streamlit as st
import pandas as pd
import re
from collections import Counter

from resume_parser import extract_text_from_pdf, extract_largest_text
import plotly.graph_objects as go
import plotly.express as px

from nlp_utils import clean_text
from matcher import calculate_similarity, calculate_similarity_with_details
from skill_matcher import (
    extract_skills,
    extract_skills_with_categories,
    analyze_resume_strength,
    detect_experience_level,
    SKILL_REGISTRY,
)

# ================= CONSTANTS =================
STRONG_HIRE_THRESHOLD = 75
HIRE_THRESHOLD = 55
CONSIDER_THRESHOLD = 30
SIMILARITY_WEIGHT = 0.4
SKILL_WEIGHT = 0.6
MAX_RESUME_SECTIONS = 5


# ================= LOCAL HELPER FUNCTIONS =================

def extract_email(text):
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else "Not Found"


def extract_phone(text):
    pattern = r'(?:\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}|\b\d{10}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else "Not Found"


def extract_name(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    def is_valid_name_line(line):
        if '@' in line or re.search(r'\d', line) or 'http' in line.lower():
            return False
        wc = len(line.split())
        return 2 <= wc <= 5 and len(line) < 60

    for line in lines[:8]:
        if is_valid_name_line(line) and line.isupper():
            return line.title()
    for line in lines[:8]:
        if is_valid_name_line(line):
            return line.title()
    return "Not Found"


def get_recommendation(score):
    if score >= STRONG_HIRE_THRESHOLD:
        return "Strong Hire"
    elif score >= HIRE_THRESHOLD:
        return "Hire"
    elif score >= CONSIDER_THRESHOLD:
        return "Consider"
    return "Not Suitable"


def render_recommendation(rec):
    colors = {
        "Strong Hire": ("success", "✅"),
        "Hire": ("info", "🟦"),
        "Consider": ("warning", "⚠️"),
        "Not Suitable": ("error", "❌"),
    }
    kind, icon = colors.get(rec, ("error", "❌"))
    getattr(st, kind)(f"{icon} {rec}")


def render_skill_pills(skills, color):
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


def experience_badge(level: str) -> str:
    badge_map = {
        "Senior": "🔴 Senior",
        "Mid-Level": "🟡 Mid-Level",
        "Junior / Entry-Level": "🟢 Junior / Entry-Level",
        "Not Specified": "⚪ Not Specified",
    }
    return badge_map.get(level, "⚪ Not Specified")


# ================= CHART HELPERS =================

def score_gauge(score: float, title: str = "Score"):
    """Render a gauge chart for a 0–100 score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1a7a4a"},
            "steps": [
                {"range": [0, 30],  "color": "#f8d7da"},
                {"range": [30, 55], "color": "#fff3cd"},
                {"range": [55, 75], "color": "#d1ecf1"},
                {"range": [75, 100],"color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=0, l=10, r=10))
    return fig


def skill_category_bar(category_skills: dict):
    """Bar chart showing skill count per category."""
    if not category_skills:
        return None
    cats = list(category_skills.keys())
    counts = [len(v) for v in category_skills.values()]
    fig = px.bar(
        x=cats, y=counts,
        labels={"x": "Category", "y": "Skills Found"},
        color=counts,
        color_continuous_scale=["#d4edda", "#1a7a4a"],
    )
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        showlegend=False,
    )
    return fig


def skill_gap_heatmap(df: pd.DataFrame, job_skills: list):
    """Binary heatmap: candidate × job skill presence."""
    if not job_skills or df.empty:
        return None
    names = df["Candidate Name"].tolist()
    matrix = []
    for _, row in df.iterrows():
        matched = set(row["Matched_Skills"])
        matrix.append([1 if s in matched else 0 for s in job_skills])

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=job_skills,
        y=names,
        colorscale=[[0, "#f8d7da"], [1, "#1a7a4a"]],
        showscale=False,
        text=[["✓" if v else "✗" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig.update_layout(
        height=max(200, 50 * len(names)),
        margin=dict(t=10, b=60, l=10, r=10),
        xaxis={"tickangle": -35},
    )
    return fig


# ================= UI SETUP =================
st.set_page_config(layout="wide", page_title="TalentScreener AI", page_icon="🎯")

# --- SIDEBAR ---
st.sidebar.title("🎯 TalentScreener AI")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Analysis", use_container_width=True):
    for key in ["df", "analysis_done", "job_skills"]:
        st.session_state.pop(key, None)
    st.rerun()

if st.session_state.get("analysis_done"):
    df_sb = st.session_state["df"]
    st.sidebar.markdown("#### 📊 Quick Stats")
    st.sidebar.metric("Candidates Analyzed", len(df_sb))
    st.sidebar.metric("Top Score", f"{df_sb['Combined Score (%)'].max()}%")
    st.sidebar.metric("Avg Score", f"{round(df_sb['Combined Score (%)'].mean(), 1)}%")
    st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigate",
    ["Evaluation Console", "Candidate Comparison", "System Documentation"],
    label_visibility="visible",
)


# ================= PAGE 1: EVALUATION CONSOLE =================
if app_mode == "Evaluation Console":

    st.title("🎯 TalentScreener AI")
    st.caption("AI-Powered Resume Screening Platform — v2.0")
    st.write("Upload Resumes  →  AI Analysis  →  Candidate Ranking  →  ATS Evaluation")
    st.markdown("---")

    col_input1, col_input2 = st.columns([1, 1], gap="large")

    with col_input1:
        st.subheader("📄 Upload Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload candidate profiles (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("Only PDF files are accepted. Multiple files supported.")

    with col_input2:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste Job Description",
            height=130,
            placeholder="Enter target skills, qualifications, and role responsibilities...",
            label_visibility="collapsed",
        )

    analyze = st.button("🔍 Analyze Candidates", use_container_width=True, type="primary")

    # --- RUN ANALYSIS ---
    if analyze:
        if not job_description and not uploaded_files:
            st.error("Please provide a Job Description and upload at least one Resume.")
        elif not uploaded_files:
            st.warning("Job Description provided, but no resumes uploaded.")
        elif not job_description:
            st.warning("Resumes uploaded, but Job Description is empty.")
        else:
            results_list = []
            clean_job = clean_text(job_description)
            job_skills = extract_skills(clean_job)

            with st.spinner("⏳ Analysing candidates — please wait..."):
                for file in uploaded_files:
                    try:
                        resume_text = extract_text_from_pdf(file)
                        candidate_name = extract_largest_text(file)

                        email = extract_email(resume_text)
                        phone = extract_phone(resume_text)
                        if not candidate_name or candidate_name.strip() == "":
                            candidate_name = extract_name(resume_text)

                        clean_resume = clean_text(resume_text)
                        sim_result = calculate_similarity_with_details(clean_resume, clean_job)
                        match_pct = int(round(sim_result["score"] * 100, 2))
                        top_terms = sim_result["top_terms"]

                        resume_skills = extract_skills(clean_resume)
                        matched_skills = sorted(set(resume_skills) & set(job_skills))
                        missing_skills = sorted(set(job_skills) - set(resume_skills))

                        skill_match_pct = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0
                        combined_score = round(
                            (match_pct * SIMILARITY_WEIGHT) + (skill_match_pct * SKILL_WEIGHT), 2
                        )
                        recommendation = get_recommendation(combined_score)

                        found_sections, missing_sections = analyze_resume_strength(resume_text)
                        keyword_score = (len(matched_skills) / len(job_skills) * 60) if job_skills else 0
                        section_score = (len(found_sections) / MAX_RESUME_SECTIONS) * 40
                        ats_score = round(keyword_score + section_score, 2)

                        exp_level = detect_experience_level(resume_text)
                        skill_categories = extract_skills_with_categories(clean_resume)

                        results_list.append({
                            "Candidate Name": candidate_name or file.name.replace(".pdf", ""),
                            "Email": email,
                            "Phone": phone,
                            "Combined Score (%)": combined_score,
                            "Similarity Score (%)": match_pct,
                            "Skill Match (%)": round(skill_match_pct, 2),
                            "Matched_Skills": matched_skills,
                            "Missing_Skills": missing_skills,
                            "Recommendation": recommendation,
                            "Resume_Text": resume_text,
                            "ATS Score": ats_score,
                            "Found Sections": found_sections,
                            "Missing Sections List": missing_sections,
                            "Experience Level": exp_level,
                            "Skill Categories": skill_categories,
                            "Top TF-IDF Terms": top_terms,
                        })
                    except Exception as e:
                        st.error(f"Error processing **{file.name}**: {str(e)}")

            if results_list:
                df = pd.DataFrame(results_list)
                df = df.sort_values("Combined Score (%)", ascending=False).reset_index(drop=True)
                st.session_state["df"] = df
                st.session_state["job_skills"] = job_skills
                st.session_state["analysis_done"] = True
            else:
                st.error("No resumes could be processed. Please check the uploaded files.")


    # --- DISPLAY RESULTS ---
    if st.session_state.get("analysis_done"):
        df = st.session_state["df"]
        job_skills = st.session_state.get("job_skills", [])
        best = df.iloc[0]

        strong_hire  = len(df[df["Recommendation"] == "Strong Hire"])
        hire         = len(df[df["Recommendation"] == "Hire"])
        consider     = len(df[df["Recommendation"] == "Consider"])
        not_suitable = len(df[df["Recommendation"] == "Not Suitable"])
        recommended  = strong_hire + hire

        st.markdown("---")
        st.header("📊 Executive Dashboard")

        # Top candidate
        st.subheader("🏆 Top Candidate")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.write(f"**Name:** {best['Candidate Name']}")
            st.write(f"**Level:** {experience_badge(best['Experience Level'])}")
        with t2:
            st.metric("Combined Score", f"{best['Combined Score (%)']}%")
        with t3:
            st.metric("ATS Score", f"{best['ATS Score']}%")
        with t4:
            st.write("**Recommendation:**")
            render_recommendation(best["Recommendation"])

        # Recommendation summary
        st.subheader("📋 Recommendation Summary")
        success_rate = round((recommended / len(df)) * 100, 2) if df.shape[0] > 0 else 0
        st.progress(success_rate / 100)
        st.caption(f"Overall Selection Rate: {success_rate}%")

        r1, r2, r3, r4 = st.columns(4)
        with r1: st.success(f"✅ Strong Hire: {strong_hire}")
        with r2: st.info(f"🟦 Hire: {hire}")
        with r3: st.warning(f"⚠️ Consider: {consider}")
        with r4: st.error(f"❌ Not Suitable: {not_suitable}")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Candidates", len(df))
        with c2: st.metric("Highest Score", f"{df['Combined Score (%)'].max()}%")
        with c3: st.metric("Average Score", f"{round(df['Combined Score (%)'].mean(), 2)}%")
        with c4: st.metric("Recommended", recommended)

        st.markdown("---")

        total_matched = sum(len(x) for x in df["Matched_Skills"])
        total_missing = sum(len(x) for x in df["Missing_Skills"])
        total_skills  = total_matched + total_missing
        coverage      = (total_matched / total_skills) if total_skills > 0 else 0

        col_res1, col_res2 = st.columns([11, 9], gap="large")


        # --- LEFT: CANDIDATE RANKING ---
        with col_res1:
            st.header("🏅 Candidate Ranking")
            st.caption("Ranked by combined similarity + skill match score.")

            for idx, row in df.iterrows():
                with st.container(border=True):
                    left_col, right_col = st.columns([4, 1])
                    with left_col:
                        st.subheader(f"#{idx + 1}  {row['Candidate Name']}")
                        st.write(f"**Email:** {row['Email']}  |  **Phone:** {row['Phone']}")
                        st.write(f"**Experience Level:** {experience_badge(row['Experience Level'])}")

                        st.write("**✅ Matched Skills**")
                        render_skill_pills(row["Matched_Skills"], "green") if row["Matched_Skills"] else st.info("No matching skills found")

                        st.write("**❌ Missing Skills**")
                        render_skill_pills(row["Missing_Skills"], "red") if row["Missing_Skills"] else st.success("No missing skills")

                        with st.expander("📊 Score Breakdown"):
                            sb1, sb2, sb3 = st.columns(3)
                            with sb1:
                                st.metric("Combined", f"{row['Combined Score (%)']}%")
                            with sb2:
                                st.metric("Similarity", f"{row['Similarity Score (%)']}%")
                            with sb3:
                                st.metric("Skill Match", f"{row['Skill Match (%)']}%")

                        if row.get("Top TF-IDF Terms"):
                            with st.expander("🔑 Why this score? (Top JD keywords)"):
                                st.write(", ".join(row["Top TF-IDF Terms"]))

                    with right_col:
                        st.metric(label="Score", value=f"{row['Combined Score (%)']}%")
                        render_recommendation(row["Recommendation"])
                        st.metric("ATS", f"{row['ATS Score']}%")

            # Export
            st.subheader("📥 Export Evaluation Data")
            df_export = df.copy()
            df_export["Matched Skills"] = df_export["Matched_Skills"].apply(lambda x: ", ".join(x))
            df_export["Missing Skills"] = df_export["Missing_Skills"].apply(lambda x: ", ".join(x))
            df_export["Phone"] = df_export["Phone"].apply(
                lambda x: f'="{x}"' if x and x != "Not Found" else x
            )
            drop_cols = [c for c in [
                "Matched_Skills", "Missing_Skills", "Resume_Text",
                "Found Sections", "Missing Sections List",
                "Skill Categories", "Top TF-IDF Terms",
            ] if c in df_export.columns]
            df_export = df_export.drop(columns=drop_cols)
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Evaluation Report (CSV)",
                data=csv_data,
                file_name="TalentScreener_Report.csv",
                mime="text/csv",
                use_container_width=True,
            )


        # --- RIGHT: SKILL ANALYSIS ---
        with col_res2:
            st.header("🔬 Skill Analysis")
            k1, k2 = st.columns(2)
            with k1: st.metric("Matched Skills", total_matched)
            with k2: st.metric("Missing Skills", total_missing)
            st.progress(coverage)

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Matched", "Missing"],
                values=[total_matched, total_missing],
                hole=0.6,
                marker=dict(colors=["#1a7a4a", "#a63030"]),
            )])
            fig_donut.update_layout(
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0),
                height=220,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption(f"Overall Skill Coverage: {round(coverage * 100, 2)}%")

            all_skills = []
            for skills in df["Matched_Skills"]:
                all_skills.extend(skills if isinstance(skills, list) else [skills])

            if all_skills:
                skill_counter = Counter(all_skills)
                top_skill = skill_counter.most_common(1)[0]
                st.success(f"🥇 Top Skill: **{top_skill[0]}** ({top_skill[1]} candidates)")

                top_skills = skill_counter.most_common(8)
                bar_fig = go.Figure(data=[go.Bar(
                    x=[s[0] for s in top_skills],
                    y=[s[1] for s in top_skills],
                    marker_color="#1a7a4a",
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

            # Skill gap heatmap
            if job_skills:
                st.subheader("🗺️ Skill Gap Heatmap")
                st.caption("Green = skill present, Red = skill missing")
                heatmap = skill_gap_heatmap(df, job_skills)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("---")

        # --- CANDIDATE DEEP DIVE ---
        st.header("🔍 Candidate Deep Dive")
        st.caption("Select a candidate for detailed evaluation.")

        candidate_labels = [f"{row['Candidate Name']} (#{idx + 1})" for idx, row in df.iterrows()]
        selected_label = st.selectbox("Select Candidate:", candidate_labels)


        if selected_label:
            selected_idx = candidate_labels.index(selected_label)
            cand = df.iloc[selected_idx]
            score = int(round(float(cand["Combined Score (%)"])))

            with st.container(border=True):
                st.subheader("🏷️ Hiring Decision")
                d1, d2, d3 = st.columns(3)
                with d1:
                    render_recommendation(cand["Recommendation"])
                with d2:
                    st.metric("Combined Score", f"{score}%")
                with d3:
                    st.metric("Experience Level", experience_badge(cand["Experience Level"]))
                st.progress(score / 100)

            with st.container(border=True):
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.subheader("👤 Personal Info")
                    st.write(f"**Name:** {cand['Candidate Name']}")
                    st.write(f"**Email:** {cand['Email']}")
                    st.write(f"**Phone:** {cand['Phone']}")
                with p2:
                    st.subheader("📈 Scores")
                    st.plotly_chart(score_gauge(score, "Combined Score"), use_container_width=True)
                with p3:
                    st.subheader("📈 ATS Score")
                    st.plotly_chart(score_gauge(cand["ATS Score"], "ATS Score"), use_container_width=True)

            # Resume section analysis
            found_sections   = cand["Found Sections"]
            missing_sections = cand["Missing Sections List"]
            strength_score   = int((len(found_sections) / MAX_RESUME_SECTIONS) * 100)

            with st.container(border=True):
                st.subheader("📄 Resume Section Analysis")
                st.progress(strength_score / 100)
                ms1, ms2 = st.columns(2)
                with ms1:
                    st.metric("Resume Strength", f"{strength_score}%")
                with ms2:
                    st.metric("ATS Score", f"{cand['ATS Score']}%")

                sec_left, sec_right = st.columns(2)
                with sec_left:
                    st.write("**✅ Detected Sections**")
                    for s in found_sections:
                        st.write(f"- {s}")
                with sec_right:
                    st.write("**🔧 Sections to Improve**")
                    if missing_sections:
                        for s in missing_sections:
                            st.write(f"- {s}")
                    else:
                        st.info("All key sections present.")

            # Category-wise skill breakdown
            skill_cats = cand.get("Skill Categories", {})
            if skill_cats:
                with st.container(border=True):
                    st.subheader("🗂️ Skills by Category")
                    cat_fig = skill_category_bar(skill_cats)
                    if cat_fig:
                        st.plotly_chart(cat_fig, use_container_width=True)
                    for cat, skills in sorted(skill_cats.items()):
                        st.write(f"**{cat}:** {', '.join(skills)}")

            # Improvement suggestions
            suggestions = []
            if "Projects" in missing_sections:
                suggestions.append("Add at least 2 academic or personal projects with tech stack details.")
            if "Experience" in missing_sections:
                suggestions.append("Include internships, freelance work, or relevant experience.")
            if "Certifications" in missing_sections:
                suggestions.append("Add certifications such as NPTEL, Coursera, Google, or AWS.")
            if "Education" in missing_sections:
                suggestions.append("Clearly mention your education details including degree and institution.")
            if "Skills" in missing_sections:
                suggestions.append("Create a dedicated technical skills section.")
            if cand["Missing_Skills"]:
                top_missing = cand["Missing_Skills"][:5]
                suggestions.append(f"Consider learning / highlighting: {', '.join(top_missing)}.")

            with st.container(border=True):
                st.subheader("💡 Resume Improvement Suggestions")
                if suggestions:
                    for s in suggestions:
                        st.info(s)
                else:
                    st.success("✅ Resume looks complete. No major improvements suggested.")

            # Distribution chart
            st.caption("Hiring recommendation distribution across all candidates")
            dist_fig = go.Figure(data=[go.Bar(
                x=["Strong Hire", "Hire", "Consider", "Not Suitable"],
                y=[strong_hire, hire, consider, not_suitable],
                marker_color=["#1a7a4a", "#1a6aaa", "#b8860b", "#a63030"],
            )])
            dist_fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=250,
                yaxis=dict(title="Candidates"),
            )
            st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander("🔎 Filter Candidates by Skill"):
                search_skill = st.text_input("Search by Skill", placeholder="e.g. Python")
                if search_skill:
                    filtered_df = df[df["Matched_Skills"].apply(
                        lambda x: search_skill.lower() in [s.lower() for s in x]
                    )]
                    if not filtered_df.empty:
                        st.success(f"{len(filtered_df)} candidate(s) found with '{search_skill}'.")
                        st.dataframe(
                            filtered_df[["Candidate Name", "Combined Score (%)", "Recommendation"]],
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"No candidate found with skill: '{search_skill}'.")

    st.markdown("---")
    st.caption("🎯 TalentScreener AI v2.0 | Smart Resume Screening & Candidate Ranking | Python · Streamlit · NLP")


# ================= PAGE 2: CANDIDATE COMPARISON =================
elif app_mode == "Candidate Comparison":
    st.title("⚖️ Candidate Comparison")
    st.caption("Compare two candidates side-by-side.")
    st.markdown("---")

    if not st.session_state.get("analysis_done"):
        st.info("Run an analysis first from the **Evaluation Console** to compare candidates.")
    else:
        df = st.session_state["df"]
        candidate_labels = [f"{row['Candidate Name']} (#{idx + 1})" for idx, row in df.iterrows()]

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            cand_a_label = st.selectbox("Candidate A", candidate_labels, index=0)
        with col_sel2:
            default_b = 1 if len(candidate_labels) > 1 else 0
            cand_b_label = st.selectbox("Candidate B", candidate_labels, index=default_b)

        idx_a = candidate_labels.index(cand_a_label)
        idx_b = candidate_labels.index(cand_b_label)
        ca = df.iloc[idx_a]
        cb = df.iloc[idx_b]

        st.markdown("---")
        col_a, col_b = st.columns(2)

        def render_candidate_card(col, cand, label):
            with col:
                with st.container(border=True):
                    st.subheader(label)
                    st.write(f"**Name:** {cand['Candidate Name']}")
                    st.write(f"**Email:** {cand['Email']}")
                    st.write(f"**Level:** {experience_badge(cand['Experience Level'])}")
                    render_recommendation(cand["Recommendation"])

                    st.plotly_chart(
                        score_gauge(float(cand["Combined Score (%)"]), "Combined Score"),
                        use_container_width=True,
                    )

                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Combined", f"{cand['Combined Score (%)']}%")
                    with m2: st.metric("Similarity", f"{cand['Similarity Score (%)']}%")
                    with m3: st.metric("ATS", f"{cand['ATS Score']}%")

                    st.write("**✅ Matched Skills**")
                    render_skill_pills(cand["Matched_Skills"], "green")

                    st.write("**❌ Missing Skills**")
                    render_skill_pills(cand["Missing_Skills"], "red")

                    st.write("**📄 Resume Sections**")
                    for s in cand["Found Sections"]:
                        st.write(f"  ✓ {s}")
                    for s in cand["Missing Sections List"]:
                        st.write(f"  ✗ {s}")

        render_candidate_card(col_a, ca, "Candidate A")
        render_candidate_card(col_b, cb, "Candidate B")

        # Comparison radar chart
        st.markdown("---")
        st.subheader("📡 Score Radar Comparison")
        categories = ["Combined Score", "Similarity Score", "Skill Match", "ATS Score",
                       "Resume Strength"]

        def get_radar_values(cand):
            strength = int((len(cand["Found Sections"]) / MAX_RESUME_SECTIONS) * 100)
            return [
                float(cand["Combined Score (%)"]),
                float(cand["Similarity Score (%)"]),
                float(cand["Skill Match (%)"]),
                float(cand["ATS Score"]),
                float(strength),
            ]

        vals_a = get_radar_values(ca)
        vals_b = get_radar_values(cb)

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=ca["Candidate Name"],
            line_color="#1a7a4a",
        ))
        radar_fig.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=cb["Candidate Name"],
            line_color="#1a6aaa",
            opacity=0.7,
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")
    st.caption("🎯 TalentScreener AI v2.0 | Candidate Comparison")


# ================= PAGE 3: SYSTEM DOCUMENTATION =================
elif app_mode == "System Documentation":
    st.title("📐 Architecture & System Documentation")
    st.caption("Technical specifications and compliance framework of TalentScreener AI v2.0")
    st.markdown("---")

    st.subheader("1. Core Processing Engine")
    st.write(
        "The system uses a multi-stage NLP pipeline: contraction expansion, "
        "protection of special tech terms (c++, c#, .net, node.js, etc.), "
        "stopword removal, and whitespace normalisation — all without requiring "
        "external corpus downloads."
    )

    st.subheader("2. Vectorisation & Similarity Metrics")
    st.write(
        "Candidate profiles are vectorised using TF-IDF with unigram + bigram "
        "tokenisation (ngram_range=(1,2)) and sublinear TF scaling. Cosine "
        "similarity generates normalised match scores. A weighted hybrid score "
        "combines semantic similarity (40%) and skill match (60%)."
    )

    st.subheader("3. Skill Registry")
    st.write(
        f"TalentScreener AI v2.0 recognises **{len(SKILL_REGISTRY)} skills** "
        "across 7 categories: Programming, Web, Data & ML, Database, "
        "Cloud & DevOps, Analytics, and Soft Skills."
    )

    st.subheader("4. ATS Score Computation")
    st.write(
        "ATS Score = Keyword Match Score (60%) + Resume Section Coverage Score (40%). "
        "Sections checked: Education, Experience, Projects, Certifications, Skills."
    )

    st.subheader("5. Experience Level Detection")
    st.write(
        "Resume text is scanned for seniority signals (years of experience, "
        "job titles, keywords like 'senior', 'intern', 'fresher') to classify "
        "candidates as Senior, Mid-Level, or Junior / Entry-Level."
    )

    st.subheader("6. Enterprise Security & Privacy")
    st.write(
        "All uploads are parsed entirely in volatile runtime memory. No persistent "
        "storage systems or external logs retain any personal data records."
    )

    st.markdown("---")
    st.caption("🎯 TalentScreener AI v2.0 | Architecture Documentation")
