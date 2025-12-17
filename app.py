import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -------------------------------
# Load and preprocess CSV
# -------------------------------
df = pd.read_csv("k3_ohe_all_skills.csv")
skill_columns = [col for col in df.columns if col not in ['Job Title', 'Skills']]

def get_job_skills_text(row):
    skills = []
    for col in skill_columns:
        val = row[col]
        if isinstance(val, str):
            skills_in_cell = [s.strip().lower() for s in val.strip("[]").replace("'", "").split(',') if s.strip()]
            skills.extend(skills_in_cell)
        elif isinstance(val, (int, float)) and val == 1:
            skills.append(col.lower())
    return " ".join(skills)

df['skill_text'] = df.apply(get_job_skills_text, axis=1)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['skill_text'].tolist())

# Skill synonyms
skill_synonyms = {
    'ms-excel': ['ms office excel', 'excel'],
    'ms-word': ['ms office word', 'word'],
    'english proficiency (spoken)': ['english speaking', 'english conversation']
}

def normalize_candidate_skills(candidate_skills):
    normalized = []
    for s in candidate_skills:
        s_lower = s.lower()
        matched = False
        for key, syns in skill_synonyms.items():
            if s_lower == key or s_lower in syns:
                normalized.append(key)
                matched = True
                break
        if not matched:
            normalized.append(s_lower)
    return normalized

def recommend_jobs(candidate_skills, top_k=5, threshold=0.1):
    candidate_skills = normalize_candidate_skills(candidate_skills)
    candidate_text = " ".join(candidate_skills)
    candidate_vector = vectorizer.transform([candidate_text])
    cos_sim = cosine_similarity(candidate_vector, tfidf_matrix)[0]

    df['similarity'] = cos_sim
    top_df = df[df['similarity'] >= threshold].sort_values('similarity', ascending=False).head(top_k)

    recommendations = []
    for _, row in top_df.iterrows():
        job_skills = set(row['skill_text'].split())
        candidate_set = set(candidate_skills)
        matched = candidate_set.intersection(job_skills)
        missing = job_skills - candidate_set
        recommendations.append({
            'Job Title': row['Job Title'],
            'Similarity Score': round(row['similarity']*100, 2),
            'Matched Skills': list(matched),
            'Missing Skills': list(missing)
        })
    return recommendations

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💼 Internship / Job Recommendation")

skills_input = st.text_input("Enter your skills (comma-separated):")
top_k = st.number_input("Number of top recommendations:", min_value=1, max_value=20, value=5, step=1)

if st.button("Get Recommendations"):
    candidate_skills = [s.strip() for s in skills_input.split(',') if s.strip()]
    if not candidate_skills:
        st.warning("Please enter at least one skill.")
    else:
        recommendations = recommend_jobs(candidate_skills, top_k=top_k)
        if not recommendations:
            st.info("No matching jobs found. Try adding more skills or lowering the threshold.")
        else:
            st.write(f"### Top {len(recommendations)} Recommended Jobs")
            for rec in recommendations:
                with st.expander(f"**{rec['Job Title']}** - Similarity: {rec['Similarity Score']}%"):
                    st.write(f"**Matched Skills ✅:** {', '.join(rec['Matched Skills']) if rec['Matched Skills'] else 'None'}")
                    st.write(f"**Missing Skills ❌:** {', '.join(rec['Missing Skills']) if rec['Missing Skills'] else 'None'}")
                    st.button(f"Select {rec['Job Title']}", key=rec['Job Title'])
