import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="★",
    layout="wide"
)

# ================= CLEAN LIGHT THEME CSS - HIGH CONTRAST =================
st.markdown("""
<style>
/* ----- Base Layout - Clean White Background ----- */
.stApp {
    background: #ffffff;
    min-height: 100vh;
}

/* ----- Typography ----- */
* {
    font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* ----- Sidebar Styling - Light Gray ----- */
[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 1px solid #e9ecef;
}

[data-testid="stSidebar"] * {
    color: #212529;
}

[data-testid="stSidebar"] .st-emotion-cache-1dtejcf {
    background: #e9ecef;
    border-radius: 12px;
}

[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stSlider {
    background-color: #ffffff;
    border-radius: 10px;
    color: #212529;
    border: 1px solid #dee2e6;
}

[data-testid="stSidebar"] .st-emotion-cache-ue6h4q {
    font-weight: 600;
    color: #495057;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    background-color: #dee2e6;
}

/* ----- Main Headers ----- */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1a3c5e;
    text-align: center;
    margin-bottom: 8px;
    letter-spacing: -0.3px;
}

.subtitle {
    text-align: center;
    color: #6c757d;
    font-size: 17px;
    margin-bottom: 40px;
    font-weight: 500;
}

/* ----- Cards - Clean White with Shadow ----- */
.card {
    background: #ffffff;
    padding: 24px 28px;
    border-radius: 20px;
    border: 1px solid #e9ecef;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #1a3c5e;
    margin-bottom: 18px;
    border-left: 4px solid #2c7da0;
    padding-left: 14px;
}

/* ----- Result Cards - Colored Gradient ----- */
.result-card {
    background: linear-gradient(135deg, #1a3c5e 0%, #2c7da0 100%);
    color: white;
    padding: 22px 12px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
}

.result-value {
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
}

.result-label {
    font-size: 13px;
    color: rgba(255,255,255,0.85);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ----- Risk Level Indicators - High Contrast ----- */
.risk-low {
    background: #d1fae5;
    color: #065f46;
    padding: 16px 22px;
    border-radius: 16px;
    border-left: 6px solid #10b981;
    font-weight: 600;
    margin-top: 20px;
}

.risk-medium {
    background: #fed7aa;
    color: #9a3412;
    padding: 16px 22px;
    border-radius: 16px;
    border-left: 6px solid #f59e0b;
    font-weight: 600;
    margin-top: 20px;
}

.risk-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 16px 22px;
    border-radius: 16px;
    border-left: 6px solid #ef4444;
    font-weight: 600;
    margin-top: 20px;
}

.risk-low strong, .risk-medium strong, .risk-high strong {
    font-size: 15px;
    display: block;
    margin-bottom: 6px;
}

/* ----- Buttons ----- */
.stButton button {
    background: linear-gradient(135deg, #2c7da0, #1a3c5e);
    color: white;
    border: none;
    padding: 12px 28px;
    border-radius: 40px;
    font-weight: 700;
    font-size: 16px;
    width: 100%;
    transition: 0.2s;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.stButton button:hover {
    background: linear-gradient(135deg, #1a5a78, #0f2c46);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* ----- Metrics ----- */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
    color: #1a3c5e;
}

[data-testid="stMetricLabel"] {
    font-size: 13px;
    font-weight: 600;
    color: #6c757d;
}

/* ----- Dataframe ----- */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e9ecef;
}

.stDataFrame table {
    background: #ffffff;
    color: #212529;
}

.stDataFrame th {
    background: #f8f9fa;
    color: #1a3c5e;
    font-weight: 600;
}

/* ----- Info Boxes (Suggestions) - Clean Blue ----- */
.stAlert {
    border-radius: 14px;
    background: #eff6ff;
    border-left: 4px solid #2c7da0;
    color: #1a3c5e;
    margin-bottom: 12px;
    padding: 12px 16px;
}

/* ----- Chart Container ----- */
.stChart canvas {
    border-radius: 14px;
    background: #ffffff;
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# ================= UI Layout =================
st.markdown('<div class="main-title">Student Performance Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict final grade | Detect academic risk | Receive tailored improvement strategies</div>',
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("Student Profile Data")

cgpa = st.sidebar.number_input("Current CGPA", 0.0, 10.0, 8.0, step=0.1)
attendance = st.sidebar.number_input("Attendance Percentage", 0.0, 100.0, 80.0, step=1.0)
backlogs = st.sidebar.number_input("Number of Backlogs", 0, 10, 0)
study_hours = st.sidebar.number_input("Study Hours per Day", 0.0, 12.0, 3.0, step=0.5)
coding_hours = st.sidebar.number_input("Coding Practice Hours per Week", 0.0, 40.0, 5.0, step=1.0)
sleep_hours = st.sidebar.number_input("Sleep Hours per Day", 0.0, 12.0, 7.0, step=0.5)
social_media = st.sidebar.number_input("Social Media Usage per Day", 0.0, 12.0, 2.0, step=0.5)
stress = st.sidebar.slider("Stress Level (Semester)", 1, 5, 3)
motivation = st.sidebar.slider("Motivation Level", 1, 5, 4)

placement_ready = st.sidebar.selectbox(
    "Placement Readiness Status",
    ["Yes", "No"]
)

assessment_style = st.sidebar.selectbox(
    "Assessment Completion Timing",
    [
        "Immediately after assigned",
        "Last minute",
        "On deadline day"
    ]
)

exam_style = st.sidebar.selectbox(
    "Exam Preparation Pattern",
    [
        "Regular study",
        "One week before exams",
        "One day before exams"
    ]
)

# Data preparation
placement_ready_value = 1 if placement_ready == "Yes" else 0

input_data = pd.DataFrame({
    "Current CGPA": [cgpa],
    "Attendance percentage": [attendance],
    "No of backlogs(if any)": [backlogs],
    "Study hours per day": [study_hours],
    "Coding practice hours per week": [coding_hours],
    "Sleep hours per day": [sleep_hours],
    "Social media usage per day": [social_media],
    "Stress level during semester": [stress],
    "Motivation level": [motivation],
    "Do you feel placement ready?": [placement_ready_value],

    "When do you usually complete assessments?_Immediately after assigned": [
        1 if assessment_style == "Immediately after assigned" else 0
    ],
    "When do you usually complete assessments?_Last minute": [
        1 if assessment_style == "Last minute" else 0
    ],
    "When do you usually complete assessments?_On deadline day": [
        1 if assessment_style == "On deadline day" else 0
    ],

    "Exam preparation style_One week before exams": [
        1 if exam_style == "One week before exams" else 0
    ],
    "Exam preparation style_Regular study": [
        1 if exam_style == "Regular study" else 0
    ],
})

# Helper function
def get_risk_level(prediction):
    if prediction >= 8.5:
        return "Low Risk", "risk-low", "Student is performing excellently with a strong academic profile. Keep up the momentum."
    elif prediction >= 7:
        return "Medium Risk", "risk-medium", "Student is on a stable track but targeted improvements can elevate performance further."
    else:
        return "High Risk", "risk-high", "Student requires immediate academic intervention and guided mentorship to improve outcomes."

# Two column layout (removed unnecessary containers)
left, right = st.columns([1.15, 0.85], gap="medium")

with left:
    # Evaluation Data Input section - direct card without extra container
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Evaluation Data Input</div>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # Key Academic Indicators section - direct card without extra container
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Key Academic Indicators</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    c1.metric("Current CGPA", f"{cgpa:.2f}")
    c2.metric("Attendance Rate", f"{attendance:.1f}%")
    
    c3, c4 = st.columns(2)
    c3.metric("Daily Study Load", f"{study_hours:.1f} hrs")
    c4.metric("Active Backlogs", backlogs)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
if st.button("Generate Performance Forecast", use_container_width=True):
    try:
        model = joblib.load("student_performance_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        risk, risk_class, message = get_risk_level(prediction)
        
        # Forecast Results section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Forecast Results</div>', unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3, gap="small")
        
        with r1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-value">{round(prediction, 2)}</div>
                    <div class="result-label">Predicted Final Grade</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with r2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-value">{risk.split()[0]}</div>
                    <div class="result-label">Academic Risk</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with r3:
            performance_gap = round(10 - prediction, 2)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-value">{performance_gap}</div>
                    <div class="result-label">Improvement Potential</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(f'<div class="{risk_class}"><strong>★ Diagnostic Insight</strong><br>{message}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Personalized Suggestions section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Customized Improvement Roadmap</div>', unsafe_allow_html=True)
        
        suggestions = []
        
        if attendance < 75:
            suggestions.append("Boost attendance consistency: low attendance disrupts learning continuity and exam readiness.")
        if study_hours < 2:
            suggestions.append("Increase daily focused study to at least 2-3 hours to reinforce core concepts.")
        if coding_hours < 4:
            suggestions.append("Amplify coding practice: 4+ hours weekly builds technical fluency and problem-solving speed.")
        if social_media > 4:
            suggestions.append("Reduce social media usage: excessive screen time correlates with lower retention and focus.")
        if sleep_hours < 6:
            suggestions.append("Optimize sleep routine: 7-8 hours of rest improves memory consolidation and cognitive performance.")
        if backlogs > 0:
            suggestions.append("Prioritize backlog clearance: unresolved backlogs create cumulative academic drag.")
        if motivation <= 2:
            suggestions.append("Set micro-goals and reward milestones to rebuild intrinsic motivation and consistency.")
        if stress >= 4:
            suggestions.append("Adopt structured planning and mindful breaks to lower stress and enhance productivity.")
        
        if not suggestions:
            suggestions.append("Academic profile is balanced. Maintain discipline and aim for progressive improvement.")
        
        for suggestion in suggestions:
            st.info(suggestion)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Factor Analysis section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Factor Impact Visualization</div>', unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            "Factor": [
                "CGPA",
                "Attendance",
                "Study Hours",
                "Coding Hours",
                "Sleep Hours",
                "Social Media",
                "Stress",
                "Motivation"
            ],
            "Value": [
                cgpa,
                attendance / 10,
                study_hours,
                coding_hours / 4,
                sleep_hours,
                social_media,
                stress,
                motivation
            ]
        })
        
        st.bar_chart(chart_data.set_index("Factor"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error("Model files missing. Please ensure 'student_performance_model.pkl' and 'scaler.pkl' are in the same directory as this application.")
    except ValueError as e:
        st.error("Feature mismatch detected. The input data structure does not align with the training configuration.")
        st.code(str(e))