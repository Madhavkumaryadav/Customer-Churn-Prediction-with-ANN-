import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #060A18;
    --panel: rgba(255, 255, 255, 0.045);
    --panel-border: rgba(140, 170, 255, 0.16);
    --text: #E8ECF8;
    --muted: #8E98B8;
    --cyan: #4FD8FF;
    --violet: #8B7CFF;
    --ok: #3EE6B0;
    --warn: #FFC857;
    --bad: #FF5C7A;
}

html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(900px 500px at 8% -10%, rgba(139, 124, 255, 0.22), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(79, 216, 255, 0.16), transparent 60%),
        var(--bg);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* Hero */
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(90deg, #FFFFFF 0%, var(--cyan) 55%, var(--violet) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0 0 1.6rem 0; max-width: 620px; }

/* Glass panels (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--panel-border) !important;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 0.6rem 0.9rem;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--cyan);
    margin: 0.2rem 0 0.6rem 0;
}

/* Inputs */
label, .stSlider label, .stSelectbox label, .stNumberInput label, .stToggle label {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
.stNumberInput button { background: transparent !important; color: var(--muted) !important; }
div[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(79, 216, 255, 0.2) !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(79, 216, 255, 0.7);
}

/* Button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #050816;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    box-shadow: 0 8px 28px rgba(99, 140, 255, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 34px rgba(99, 140, 255, 0.5);
    color: #050816;
}
.stButton > button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

/* Result card */
.result-wrap { text-align: center; padding: 0.4rem 0 0.2rem 0; }
.gauge-value { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; margin-top: -3.2rem; }
.gauge-caption { color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem; }
.badge {
    display: inline-block;
    padding: 0.4rem 0.95rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    border: 1px solid currentColor;
}
.verdict { margin-top: 1rem; font-size: 1.05rem; font-weight: 500; }
.empty-state { text-align: center; color: var(--muted); padding: 2.4rem 0.6rem; line-height: 1.6; }
.empty-state .ring {
    width: 74px; height: 74px; margin: 0 auto 1rem auto; border-radius: 50%;
    border: 2px dashed rgba(79, 216, 255, 0.5);
    box-shadow: 0 0 24px rgba(79, 216, 255, 0.18) inset;
}

/* Summary chips */
.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; justify-content: center; margin-top: 1.1rem; }
.chip {
    font-size: 0.75rem; color: var(--muted);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--panel-border);
    padding: 0.28rem 0.65rem; border-radius: 999px;
}

@media (prefers-reduced-motion: reduce) {
    .stButton > button { transition: none; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load the trained model, encoders and scaler (unchanged)
# ------------------------------------------------------------------
model = tf.keras.models.load_model('model.h5')

with open('level_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('geo_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
<h1>Customer Churn Prediction</h1>
<p>Enter a customer's details to estimate how likely they are to leave the bank.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], gap="large")

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        with c2:
            gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 35)

    with st.container(border=True):
        st.markdown('<div class="section-title">Financials</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input('Credit score', min_value=0.0, step=1.0)
        with c2:
            balance = st.number_input('Balance', min_value=0.0, step=100.0)
        with c3:
            estimated_salary = st.number_input('Estimated salary', min_value=0.0, step=100.0)

    with st.container(border=True):
        st.markdown('<div class="section-title">Relationship with the bank</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider('Tenure (years)', 0, 10, 3)
        with c2:
            num_of_products = st.slider('Number of products', 1, 4, 1)
        t1, t2 = st.columns(2)
        with t1:
            has_cr_card = st.toggle('Has credit card')
        with t2:
            is_active_member = st.toggle('Is active member')

    predict_clicked = st.button('Predict churn')

# ------------------------------------------------------------------
# Prediction (same logic as before)
# ------------------------------------------------------------------
if predict_clicked:
    ## Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_cr_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary]
    })

    ## One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    ## Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    ## Scale the input data
    input_data_scaled = scaler.transform(input_data)

    ## Predict churn
    prediction = model.predict(input_data_scaled)
    st.session_state['proba'] = float(prediction[0][0])
    st.session_state['summary'] = [
        geography,
        str(gender),
        f"Age {age}",
        f"{tenure} yr tenure",
        f"{num_of_products} product{'s' if num_of_products > 1 else ''}",
        "Card" if has_cr_card else "No card",
        "Active" if is_active_member else "Inactive",
    ]

# ------------------------------------------------------------------
# Result panel
# ------------------------------------------------------------------
with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        if 'proba' not in st.session_state:
            st.markdown(
                """
<div class="empty-state">
<div class="ring"></div>
No prediction yet.<br>Fill in the details and select <b>Predict churn</b>.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            p = st.session_state['proba']

            if p > 0.5:
                color, level = "#FF5C7A", "High risk"
                verdict = "The customer is likely to churn."
            elif p > 0.3:
                color, level = "#FFC857", "Moderate risk"
                verdict = "The customer is not likely to churn, but is worth watching."
            else:
                color, level = "#3EE6B0", "Low risk"
                verdict = "The customer is not likely to churn."

            arc_len = 3.14159 * 90
            filled = max(0.0, min(1.0, p)) * arc_len
            chips = "".join(f'<span class="chip">{c}</span>' for c in st.session_state['summary'])

            st.markdown(
                f"""
<div class="result-wrap">
<svg viewBox="0 0 220 125" width="100%" style="max-width:320px">
<defs>
<linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#4FD8FF"/>
<stop offset="100%" stop-color="{color}"/>
</linearGradient>
</defs>
<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="14" stroke-linecap="round"/>
<path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="url(#g)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{filled:.1f} {arc_len:.1f}"/>
</svg>
<div class="gauge-value" style="color:{color}">{p*100:.0f}%</div>
<div class="gauge-caption">Churn probability ({p:.2f})</div>
<span class="badge" style="color:{color}; background:{color}1F">{level}</span>
<div class="verdict">{verdict}</div>
<div class="chips">{chips}</div>
</div>
""",
                unsafe_allow_html=True,
            )