import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLES (CLEAN + PROFESSIONAL)
# --------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main {
    padding: 2rem;
}

h1, h2, h3 {
    font-weight: 600;
    color: #0f172a;
}

.subtitle {
    color: #64748b;
    font-size: 1rem;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
}

.metric-card {
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2563eb;
}

.badge-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 600;
}

.badge-low {
    background: #dcfce7;
    color: #166534;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL & ENCODERS
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encoder_gender.pkl", "rb") as f:
        gender_enc = pickle.load(f)
    with open("onehot_encoder_geo.pkl", "rb") as f:
        geo_enc = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, gender_enc, geo_enc, scaler

model, gender_enc, geo_enc, scaler = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📊 Churn AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Analytics", "About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Neural Network · Accuracy ~86%")

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.title("Customer Churn Prediction")
    st.markdown(
        "<p class='subtitle'>Predict customer churn using Machine Learning & explainable insights</p>",
        unsafe_allow_html=True
    )

    st.markdown("### 🔑 Key Capabilities")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card metric-card">
            <h3>🎯 Accurate Model</h3>
            <p>Neural Network trained on 10k+ customers</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card metric-card">
            <h3>📈 Business Insights</h3>
            <p>Identify high-risk customers early</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card metric-card">
            <h3>🧠 Explainable AI</h3>
            <p>Understand why churn happens</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. Enter customer details  
    2. Model predicts churn probability  
    3. Visual insights guide retention actions  
    """)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
elif page == "Prediction":
    st.title("Churn Prediction")
    st.markdown("<p class='subtitle'>Enter customer details below</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            geography = st.selectbox("Geography", geo_enc.categories_[0])
            gender = st.selectbox("Gender", gender_enc.classes_)
            age = st.slider("Age", 18, 92, 35)

        with col2:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
            salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

        with col3:
            tenure = st.slider("Tenure (Years)", 0, 10, 5)
            products = st.slider("Number of Products", 1, 4, 2)
            active = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        input_df = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_enc.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [products],
            "HasCrCard": [1],
            "IsActiveMember": [active],
            "EstimatedSalary": [salary]
        })

        geo_encoded = geo_enc.transform([[geography]]).toarray()
        geo_df = pd.DataFrame(geo_encoded, columns=geo_enc.get_feature_names_out())

        X = pd.concat([input_df, geo_df], axis=1)
        X_scaled = scaler.transform(X)

        prob = model.predict(X_scaled, verbose=0)[0][0]

        st.markdown("### 📌 Prediction Result")
        if prob > 0.5:
            st.markdown(f"<span class='badge-high'>High Churn Risk — {prob:.1%}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-low'>Low Churn Risk — {prob:.1%}</span>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------
elif page == "Analytics":
    st.title("Customer Analytics")

    df = pd.read_csv("Churn_Modelling.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(df))
    col2.metric("Churn Rate", f"{df['Exited'].mean()*100:.1f}%")
    col3.metric("Avg Age", f"{df['Age'].mean():.1f}")

    st.markdown("### Churn by Geography")
    geo = df.groupby("Geography")["Exited"].mean().reset_index()
    st.plotly_chart(px.bar(geo, x="Geography", y="Exited"), use_container_width=True)

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    **Customer Churn Prediction System**  
    Built using Deep Learning to help businesses retain customers.

    **Tech Stack**
    - TensorFlow / Keras  
    - Streamlit  
    - Scikit-learn  
    - Plotly  

    Perfect for **banking, telecom, SaaS & retail** use cases.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
Customer Churn Prediction System · 2026
</div>
""", unsafe_allow_html=True)
