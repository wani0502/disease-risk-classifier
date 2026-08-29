import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Classifier",
    
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LIGHT THEME
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       MAIN PAGE
       ======================================================== */

    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: #12355B !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #174A6E !important;
        font-weight: 650 !important;
    }

    p {
        color: #52667A;
    }


    /* ========================================================
       INPUT FIELDS
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    label {
        color: #334E68 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        background-color: #DBEAFE;
        color: #1E3A5F;
        border: 1px solid #93C5FD;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #BFDBFE;
        color: #172554;
        border-color: #60A5FA;
    }

    .stButton > button:active {
        background-color: #93C5FD;
        color: #172554;
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    .stSlider [data-baseweb="slider"] {
        color: #176B87;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    .stAlert {
        border-radius: 8px;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #E2E8F0;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetricValue"] {
        color: #176B87;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: #52667A;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    .stProgress > div > div > div > div {
        background-color: #176B87;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    .streamlit-expanderHeader {
        color: #174A6E !important;
        font-weight: 600;
    }


    /* ========================================================
       HTML TABLES
       ======================================================== */

    .reference-table,
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }

    .reference-table th,
    .comparison-table th {
        background-color: #E8F1F8;
        color: #19324D;
        font-weight: 600;
        padding: 11px 14px;
        text-align: left;
        border-bottom: 1px solid #CBD5E1;
    }

    .reference-table td,
    .comparison-table td {
        padding: 10px 14px;
        color: #334155;
        border-bottom: 1px solid #E2E8F0;
    }

    .reference-table tr:last-child td,
    .comparison-table tr:last-child td {
        border-bottom: none;
    }

    .reference-table tr:hover,
    .comparison-table tr:hover {
        background-color: #F8FAFC;
    }


    /* ========================================================
       MODEL RESULT
       ======================================================== */

    .result-box {
        background-color: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    .risk-number {
        font-size: 2.3rem;
        font-weight: 700;
        color: #176B87;
    }

    .small-note {
        color: #64748B;
        font-size: 0.85rem;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background-color: #FFF8E7;
        border-left: 4px solid #E6A817;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1.5rem;
    }

    .disclaimer-title {
        color: #805B00;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .disclaimer-text {
        color: #6B5A32;
        font-size: 0.85rem;
        line-height: 1.5;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_xgb_model.joblib"

# IMPORTANT:
# This is the threshold selected during model evaluation.
# The model itself predicts probabilities; this threshold
# determines whether probability becomes class 0 or class 1.
THRESHOLD = 0.65


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title(" Diabetes Risk Classifier")

st.write(
    "Enter the health information below to estimate the "
    "model-predicted risk of the At Risk classification."
)

st.info(
    "This tool is based on the CDC Diabetes Health Indicators "
    "dataset and is intended for educational purposes."
)


# ============================================================
# HEALTH INFORMATION
# ============================================================

st.header(" Health Information")

col1, col2 = st.columns(2)

with col1:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (kg/m²)."
    )

    age_labels = {
        1: "18–24",
        2: "25–29",
        3: "30–34",
        4: "35–39",
        5: "40–44",
        6: "45–49",
        7: "50–54",
        8: "55–59",
        9: "60–64",
        10: "65–69",
        11: "70–74",
        12: "75–79",
        13: "80+"
    }

    age = st.selectbox(
        "Age Group",
        list(age_labels.keys()),
        format_func=lambda x: age_labels[x],
        help="Age group used in the original dataset."
    )

    genhlth_labels = {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    }

    genhlth = st.selectbox(
        "General Health",
        list(genhlth_labels.keys()),
        format_func=lambda x: genhlth_labels[x],
        help="Self-reported general health."
    )


with col2:

    phys_hlth = st.slider(
        "Physical Health — days affected",
        0,
        30,
        0,
        help=(
            "Number of days during the past 30 days when "
            "physical health was not good."
        )
    )

    ment_hlth = st.slider(
        "Mental Health — days affected",
        0,
        30,
        0,
        help=(
            "Number of days during the past 30 days when "
            "mental health was not good."
        )
    )

    diffwalk = st.selectbox(
        "Difficulty Walking",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    physactivity = st.selectbox(
        "Physical Activity",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# MEDICAL HISTORY
# ============================================================

st.header(" Medical History")

col1, col2 = st.columns(2)

with col1:

    highbp = st.selectbox(
        "High Blood Pressure",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    highchol = st.selectbox(
        "High Cholesterol",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart = st.selectbox(
        "Heart Disease / Heart Attack",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


with col2:

    stroke = st.selectbox(
        "History of Stroke",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    cholcheck = st.selectbox(
        "Cholesterol Check",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    smoker = st.selectbox(
        "Smoker",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# LIFESTYLE
# ============================================================

st.header(" Lifestyle")

col1, col2 = st.columns(2)

with col1:

    fruits = st.selectbox(
        "Regularly Eat Fruits",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    veggies = st.selectbox(
        "Regularly Eat Vegetables",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    alcohol = st.selectbox(
        "Heavy Alcohol Consumption",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


with col2:

    healthcare = st.selectbox(
        "Healthcare Coverage",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    nodoc = st.selectbox(
        "Could Not See Doctor Due to Cost",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# DEMOGRAPHICS
# ============================================================

st.header(" Demographics")

col1, col2 = st.columns(2)

with col1:

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    education_labels = {
        1: "Never attended school / kindergarten",
        2: "Grades 1–8",
        3: "Grades 9–11",
        4: "High school graduate",
        5: "Some college / technical school",
        6: "College graduate"
    }

    education = st.selectbox(
        "Education Level",
        list(education_labels.keys()),
        format_func=lambda x: education_labels[x]
    )


with col2:

    income_labels = {
        1: "Less than $10,000",
        2: "$10,000–$14,999",
        3: "$15,000–$19,999",
        4: "$20,000–$24,999",
        5: "$25,000–$34,999",
        6: "$35,000–$49,999",
        7: "$50,000–$74,999",
        8: "$75,000+"
    }

    income = st.selectbox(
        "Income Level",
        list(income_labels.keys()),
        format_func=lambda x: income_labels[x]
    )


# ============================================================
# MODEL INPUT
# ============================================================

input_data = pd.DataFrame([{
    "BMI": bmi,
    "MentHlth": ment_hlth,
    "PhysHlth": phys_hlth,
    "Age": age,
    "HighBP": highbp,
    "HighChol": highchol,
    "CholCheck": cholcheck,
    "Smoker": smoker,
    "Stroke": stroke,
    "HeartDiseaseorAttack": heart,
    "PhysActivity": physactivity,
    "Fruits": fruits,
    "Veggies": veggies,
    "HvyAlcoholConsump": alcohol,
    "AnyHealthcare": healthcare,
    "NoDocbcCost": nodoc,
    "DiffWalk": diffwalk,
    "Sex": sex,
    "GenHlth": genhlth,
    "Education": education,
    "Income": income
}])


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button("Calculate Risk"):

    # Model probability
    probability = model.predict_proba(input_data)[0, 1]

    # IMPORTANT:
    # Do NOT use model.predict() here because that uses the
    # default 0.50 threshold.
    prediction = int(probability >= THRESHOLD)

    st.header("Risk Assessment")

st.markdown(
f"""
<div class="result-box">
<div class="small-note">
Model-predicted probability
</div>
<div class="risk-number">
{probability * 100:.1f}%
</div>
<div class="small-note">
Classification threshold: {THRESHOLD:.2f}
</div>
</div>
""",
unsafe_allow_html=True
)

    st.progress(float(probability))


    # --------------------------------------------------------
    # CLASSIFICATION RESULT
    # --------------------------------------------------------

    if prediction == 1:

        st.warning(
            "The model predicts the **At Risk** classification."
        )

        st.caption(
            f"The predicted probability ({probability:.3f}) "
            f"is at or above the optimized threshold of {THRESHOLD:.2f}."
        )

    else:

        st.success(
            "✓ The model predicts the **No Diabetes** classification."
        )

        st.caption(
            f"The predicted probability ({probability:.3f}) "
            f"is below the optimized threshold of {THRESHOLD:.2f}."
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(" Main Factors Influencing This Prediction")

    try:

        explainer = shap.TreeExplainer(model)

        person_shap = explainer.shap_values(input_data)

        shap_array = np.asarray(person_shap)

        # Handle different SHAP output formats
        if shap_array.ndim == 3:
            shap_array = shap_array[0, :, 1]

        elif shap_array.ndim == 2:
            shap_array = shap_array[0]

        explanation = pd.DataFrame({
            "Feature": input_data.columns,
            "SHAP": shap_array
        })

        explanation["Absolute SHAP"] = (
            explanation["SHAP"].abs()
        )

        explanation = explanation.sort_values(
            "Absolute SHAP",
            ascending=False
        )

        top3 = explanation.head(3)

        for _, row in top3.iterrows():

            if row["SHAP"] > 0:
                direction = "increased"
            else:
                direction = "decreased"

            st.write(
                f"**{row['Feature']}** — this feature "
                f"{direction} the model's predicted risk."
            )

    except Exception as e:

        st.info(
            "Feature-level explanations are currently unavailable."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("Model Performance")

st.markdown("### Final Model")

st.write(
    "The final model is an XGBoost classifier trained on the "
    "CDC Diabetes Health Indicators dataset."
)

st.markdown("#### Final Test Performance")

final_metrics = pd.DataFrame({
    "Metric": [
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "Classification Threshold"
    ],
    "Score": [
        "0.8253",
        "0.4650",
        "0.4121",
        "0.6089",
        "0.4915",
        "0.65"
    ]
})

st.markdown(
    final_metrics.to_html(
        index=False,
        classes="comparison-table"
    ),
    unsafe_allow_html=True
)

st.markdown("### Model Comparison")

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Mean ROC-AUC": [
        0.8177,
        0.8222,
        0.8224
    ],
    "Std": [
        0.0023,
        0.0017,
        0.0023
    ]
})

model_comparison_display = model_comparison.copy()

model_comparison_display["Mean ROC-AUC"] = (
    model_comparison_display["Mean ROC-AUC"]
    .map(lambda x: f"{x:.4f}")
)

model_comparison_display["Std"] = (
    model_comparison_display["Std"]
    .map(lambda x: f"{x:.4f}")
)

st.markdown(
    model_comparison_display.to_html(
        index=False,
        classes="comparison-table"
    ),
    unsafe_allow_html=True
)

st.write(
    "XGBoost achieved the highest mean ROC-AUC among the three "
    "evaluated models and was selected as the final model."
)


# ============================================================
# DATASET REFERENCE
# ============================================================

st.header("Dataset Reference")

st.markdown("### Age Groups")

age_table = pd.DataFrame({
    "Code": list(age_labels.keys()),
    "Age Group": list(age_labels.values())
})

st.markdown(
    age_table.to_html(
        index=False,
        classes="reference-table"
    ),
    unsafe_allow_html=True
)

st.markdown("### General Health")

health_table = pd.DataFrame({
    "Code": list(genhlth_labels.keys()),
    "Description": list(genhlth_labels.values())
})

st.markdown(
    health_table.to_html(
        index=False,
        classes="reference-table"
    ),
    unsafe_allow_html=True
)

st.markdown("### Education")

education_table = pd.DataFrame({
    "Code": list(education_labels.keys()),
    "Description": list(education_labels.values())
})

st.markdown(
    education_table.to_html(
        index=False,
        classes="reference-table"
    ),
    unsafe_allow_html=True
)

st.markdown("### Income")

income_table = pd.DataFrame({
    "Code": list(income_labels.keys()),
    "Income Level": list(income_labels.values())
})

st.markdown(
    income_table.to_html(
        index=False,
        classes="reference-table"
    ),
    unsafe_allow_html=True
)

st.markdown("### Binary Variables")

st.markdown(
    """
    <table class="reference-table">
        <tr>
            <th>Value</th>
            <th>Meaning</th>
        </tr>
        <tr>
            <td>0</td>
            <td>No</td>
        </tr>
        <tr>
            <td>1</td>
            <td>Yes</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Binary variables such as HighBP, HighChol, Stroke, Smoker, "
    "PhysActivity and DiffWalk use 0 = No and 1 = Yes."
)


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.header("About the Project")

st.write(
    "This project uses machine learning to estimate the "
    "model-predicted probability of the At Risk classification "
    "using health, lifestyle and demographic indicators."
)

st.write(
    "The complete workflow includes exploratory data analysis, "
    "preprocessing, class-imbalance handling, model comparison, "
    "hyperparameter tuning, threshold optimization and SHAP "
    "explainability."
)


# ============================================================
# ABOUT THE CREATOR
# ============================================================

st.header("About the Creator")

st.write(
    "This project was designed and developed by Wani Rathaur, "
    "with a focus on Machine Learning, Data Science and "
    "Artificial Intelligence."
)

st.write(
    "The project demonstrates an end-to-end machine learning "
    "workflow applied to a real-world healthcare classification "
    "problem."
)

col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <a href="https://github.com/wani0502/disease-risk-classifier"
           target="_blank"
           style="
               display:block;
               text-align:center;
               text-decoration:none;
               background:#243B53;
               color:white;
               padding:10px 20px;
               border-radius:8px;
               font-weight:600;
               font-size:15px;
               border:1px solid #243B53;
               transition:0.2s;
           ">
           GitHub
        </a>
        """,
        unsafe_allow_html=True
    )

# with col2:
#     st.markdown(
#         """
#         <a href="https://www.linkedin.com/in/wani-rathaur-096a54343/"
#            target="_blank"
#            style="
#                display:block;
#                text-align:center;
#                text-decoration:none;
#                background:#243B53;
#                color:white;
#                padding:10px 20px;
#                border-radius:8px;
#                font-weight:600;
#                font-size:15px;
#                border:1px solid #243B53;
#                transition:0.2s;
#            ">
#            LinkedIn
#         </a>
#         """,
#         unsafe_allow_html=True
#     )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
<div class="disclaimer">

<div class="disclaimer-title">
Medical Disclaimer
</div>

<div class="disclaimer-text">
This application is intended for educational and
demonstration purposes only. It is not a medical
diagnostic tool and should not be used to make
healthcare decisions. Predictions are generated
by a machine-learning model and may be inaccurate.
Please consult a qualified healthcare professional
for medical advice.
</div>

</div>
    """,
    unsafe_allow_html=True
)
