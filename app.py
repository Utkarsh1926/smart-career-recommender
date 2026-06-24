"""
SMART CAREER PATH RECOMMENDER
Day 5 - Streamlit Web Application
Run this with: streamlit run app.py
Make sure this file is in your CareerProject folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Smart Career Path Recommender",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================
@st.cache_resource
def load_model_and_encoders():
    try:
        # Try loading saved model and encoders
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    except:
        # If files not found, retrain from scratch
        np.random.seed(42)
        random.seed(42)

        career_profiles = {
            "Data Analyst":        dict(stream=["Science","Commerce"], skills=["Analytical","Technical"],
                                        interests=["Coding","Finance"], comm=(4,7), lead=(3,6), tech=(7,10)),
            "Financial Analyst":   dict(stream=["Commerce"], skills=["Analytical","Management"],
                                        interests=["Finance","Business"], comm=(5,8), lead=(4,7), tech=(4,7)),
            "HR Manager":          dict(stream=["Arts","Commerce"], skills=["Communication","Management"],
                                        interests=["People Management","Business"], comm=(7,10), lead=(6,9), tech=(1,4)),
            "Marketing Executive": dict(stream=["Commerce","Arts"], skills=["Creative","Communication"],
                                        interests=["Marketing","Design"], comm=(7,10), lead=(5,8), tech=(2,5)),
            "Entrepreneur":        dict(stream=["Commerce","Science","Arts"], skills=["Management","Creative"],
                                        interests=["Business","Marketing"], comm=(6,9), lead=(7,10), tech=(3,7)),
            "Business Analyst":    dict(stream=["Commerce","Science"], skills=["Analytical","Management"],
                                        interests=["Business","Finance"], comm=(6,9), lead=(5,8), tech=(5,8)),
            "Software Developer":  dict(stream=["Science"], skills=["Technical","Analytical"],
                                        interests=["Coding"], comm=(3,6), lead=(2,5), tech=(8,10)),
        }

        streams_all   = ["Science", "Commerce", "Arts"]
        skills_all    = ["Technical", "Analytical", "Creative", "Communication", "Management"]
        interests_all = ["Coding", "Finance", "Marketing", "People Management", "Business", "Design"]

        rows = []
        for career, p in career_profiles.items():
            for _ in range(120):
                if random.random() < 0.08:
                    row = [random.choice(streams_all), random.choice(skills_all),
                           random.choice(interests_all),
                           np.random.randint(1,11), np.random.randint(1,11), np.random.randint(1,11), career]
                else:
                    row = [random.choice(p["stream"]), random.choice(p["skills"]),
                           random.choice(p["interests"]),
                           np.random.randint(p["comm"][0], p["comm"][1]+1),
                           np.random.randint(p["lead"][0], p["lead"][1]+1),
                           np.random.randint(p["tech"][0], p["tech"][1]+1), career]
                rows.append(row)

        df = pd.DataFrame(rows, columns=["Academic_Stream","Skills","Interests",
                                          "Communication_Skills","Leadership_Skills",
                                          "Technical_Knowledge","Career"])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        encoders = {}
        for col in ["Academic_Stream", "Skills", "Interests"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        target_le = LabelEncoder()
        df["Career"] = target_le.fit_transform(df["Career"])
        encoders["Career"] = target_le

        X = df.drop(columns=["Career"])
        y = df["Career"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, encoders

# ============================================================
# CAREER INFO
# ============================================================
career_info = {
    "Software Developer": {
        "icon": "💻",
        "desc": "Design, build and maintain software applications and systems.",
        "skills": "Python, Java, Problem Solving, Logical Thinking",
        "salary": "₹4L - ₹20L per year",
        "color": "#4ECDC4"
    },
    "Data Analyst": {
        "icon": "📊",
        "desc": "Analyze data to help organizations make better business decisions.",
        "skills": "Excel, SQL, Python, Statistics, Visualization",
        "salary": "₹3.5L - ₹15L per year",
        "color": "#45B7D1"
    },
    "Financial Analyst": {
        "icon": "💰",
        "desc": "Analyze financial data and guide investment decisions.",
        "skills": "Accounting, Excel, Financial Modelling, Economics",
        "salary": "₹4L - ₹18L per year",
        "color": "#96CEB4"
    },
    "HR Manager": {
        "icon": "👥",
        "desc": "Manage employee relations, recruitment, and organizational culture.",
        "skills": "Communication, Empathy, Negotiation, Leadership",
        "salary": "₹3L - ₹12L per year",
        "color": "#FFEAA7"
    },
    "Marketing Executive": {
        "icon": "📢",
        "desc": "Plan and execute marketing campaigns to promote products/services.",
        "skills": "Creativity, Social Media, Communication, Analytics",
        "salary": "₹2.5L - ₹12L per year",
        "color": "#DDA0DD"
    },
    "Business Analyst": {
        "icon": "📋",
        "desc": "Bridge the gap between business needs and technology solutions.",
        "skills": "Analytical Thinking, Documentation, SQL, Communication",
        "salary": "₹4L - ₹16L per year",
        "color": "#F0A500"
    },
    "Entrepreneur": {
        "icon": "🚀",
        "desc": "Start and run your own business venture.",
        "skills": "Leadership, Risk-taking, Innovation, Finance, Networking",
        "salary": "Variable — based on business success",
        "color": "#FF6B6B"
    },
}

# ============================================================
# MAIN APP
# ============================================================
# Header
st.markdown("""
    <h1 style='text-align:center; color:#1F4E79;'>🎓 Smart Career Path Recommender</h1>
    <p style='text-align:center; color:#555; font-size:16px;'>
        Enter your profile below and get a personalized career recommendation using AI & ML
    </p>
    <hr>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI Model..."):
    model, encoders = load_model_and_encoders()

st.success("✅ AI Model Ready!")
st.markdown("---")

# ── INPUT FORM ────────────────────────────────────────────
st.markdown("### 📝 Enter Your Student Profile")

col1, col2 = st.columns(2)

with col1:
    stream = st.selectbox(
        "🏫 Academic Stream",
        ["Science", "Commerce", "Arts"],
        help="Your current or completed academic stream"
    )
    skills = st.selectbox(
        "🛠️ Primary Skills",
        ["Analytical", "Technical", "Creative", "Communication", "Management"],
        help="Your strongest skill area"
    )
    interests = st.selectbox(
        "❤️ Area of Interest",
        ["Business", "Coding", "Design", "Finance", "Marketing", "People Management"],
        help="What you enjoy doing the most"
    )

with col2:
    comm = st.slider(
        "🗣️ Communication Skills",
        min_value=1, max_value=10, value=5,
        help="Rate your communication skills from 1 (low) to 10 (excellent)"
    )
    lead = st.slider(
        "👑 Leadership Skills",
        min_value=1, max_value=10, value=5,
        help="Rate your leadership abilities from 1 (low) to 10 (excellent)"
    )
    tech = st.slider(
        "💡 Technical Knowledge",
        min_value=1, max_value=10, value=5,
        help="Rate your technical knowledge from 1 (low) to 10 (excellent)"
    )

st.markdown("---")

# ── PREDICT BUTTON ────────────────────────────────────────
if st.button("🔍 Predict My Career", use_container_width=True):

    # Encode inputs
    stream_enc   = encoders["Academic_Stream"].transform([stream])[0]
    skills_enc   = encoders["Skills"].transform([skills])[0]
    interest_enc = encoders["Interests"].transform([interests])[0]

    input_data = [[stream_enc, skills_enc, interest_enc, comm, lead, tech]]

    # Predict
    prediction      = model.predict(input_data)[0]
    career_result   = encoders["Career"].inverse_transform([prediction])[0]
    probabilities   = model.predict_proba(input_data)[0]
    confidence      = round(max(probabilities) * 100, 2)

    # Get career details
    info = career_info.get(career_result, {})
    icon = info.get("icon", "🎯")

    st.markdown("---")
    st.markdown("## 🎯 Your Career Recommendation")

    # Result box
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1F4E79, #2E74B5);
                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
            <h1 style='color:white; font-size:48px; margin:0;'>{icon}</h1>
            <h2 style='color:white; margin:10px 0;'>{career_result}</h2>
            <h3 style='color:#90CAF9; margin:0;'>Confidence: {confidence}%</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Career details
    if info:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📌 About this Career:**")
            st.info(info.get("desc", ""))
            st.markdown("**💰 Expected Salary:**")
            st.success(info.get("salary", ""))
        with c2:
            st.markdown("**🛠️ Key Skills Required:**")
            st.warning(info.get("skills", ""))

    st.markdown("---")

    # Top 3 career probabilities chart
    st.markdown("### 📊 Top Career Matches")
    career_names   = encoders["Career"].classes_
    prob_df        = pd.DataFrame({"Career": career_names, "Probability": probabilities})
    prob_df        = prob_df.sort_values("Probability", ascending=False).head(3)

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#1F4E79", "#2E74B5", "#90CAF9"]
    bars = ax.barh(prob_df["Career"], prob_df["Probability"] * 100,
                   color=colors, edgecolor="white")
    ax.set_xlabel("Match Probability (%)")
    ax.set_title("Top 3 Career Matches for Your Profile")
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, prob_df["Probability"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    # Input summary
    st.markdown("---")
    st.markdown("### 📋 Your Profile Summary")
    summary_data = {
        "Field": ["Academic Stream", "Skills", "Interests",
                  "Communication Skills", "Leadership Skills", "Technical Knowledge"],
        "Your Input": [stream, skills, interests,
                       f"{comm}/10", f"{lead}/10", f"{tech}/10"]
    }
    st.table(pd.DataFrame(summary_data))

# ── FOOTER ───────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:#888; font-size:13px;'>
        Smart Career Path Recommender | AI & ML Project | Powered by Random Forest Classifier
    </p>
""", unsafe_allow_html=True)
