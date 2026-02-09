import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="ChurnPredict - AI Customer Retention",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. LOAD ASSETS (Model, Encoders, Scaler)
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('model.h5')
        with open('label_encoder_gender.pkl', 'rb') as f:
            le_gender = pickle.load(f)
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            ohe_geo = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        return model, le_gender, ohe_geo, sc
    except Exception as e:
        st.error(f"Error loading model files: {e}. Make sure .h5 and .pkl files are in the same folder.")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# 3. GLOBAL CSS FIXES (For smooth scrolling and hiding code visibility error)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html { scroll-behavior: smooth !important; }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Metric Card styling for visual cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# 4. HEADER & HERO SECTION (With Fixed JavaScript Navigation)
# Yahan 'window.parent' use kiya gaya hai taaki Streamlit iframe se bahar scroll command bhej sake
hero_html = """
<div id="home"></div>
<nav style="display: flex; justify-content: space-between; align-items: center; position: fixed; top: 0; left: 0; right: 0; background: rgba(10, 14, 39, 0.95); padding: 15px 60px; z-index: 9999; border-bottom: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);">
    <div style="font-size: 24px; font-weight: 800; color: white;">🎯 Churn<span style="color:#4F46E5">Predict</span></div>
    <div style="display: flex; gap: 35px;">
        <a href="javascript:void(0)" onclick="window.parent.document.getElementById('home').scrollIntoView({behavior: 'smooth'})" style="color: #cbd5e1; text-decoration: none; font-weight: 500;">Home</a>
        <a href="javascript:void(0)" onclick="window.parent.document.getElementById('predict-section').scrollIntoView({behavior: 'smooth'})" style="color: #cbd5e1; text-decoration: none; font-weight: 500;">Predict</a>
        <a href="javascript:void(0)" onclick="window.parent.document.getElementById('analytics-section').scrollIntoView({behavior: 'smooth'})" style="color: #cbd5e1; text-decoration: none; font-weight: 500;">Analytics</a>
    </div>
    <button onclick="window.parent.document.getElementById('predict-section').scrollIntoView({behavior: 'smooth'})" style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: white; border: none; padding: 10px 22px; border-radius: 10px; cursor: pointer; font-weight: 600;">Get Started</button>
</nav>

<div style="padding: 180px 20px 100px; text-align: center; color: white;">
    <h1 style="font-size: 65px; font-weight: 900; margin-bottom: 20px; letter-spacing: -2px;">AI-Powered Customer <span style="background: linear-gradient(135deg, #60A5FA 0%, #4F46E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Retention</span></h1>
    <p style="font-size: 20px; color: #94a3b8; max-width: 750px; margin: 0 auto 40px;">Deep Learning models to predict churn probability with 86.4% precision. Turn data into actionable loyalty strategies.</p>
    <div style="display: flex; justify-content: center; gap: 30px;">
        <div style="text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #4F46E5;">86%</div>
            <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Accuracy</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #4F46E5;">10k+</div>
            <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Analyzed</div>
        </div>
    </div>
</div>
"""
components.html(hero_html, height=700)

# 5. PREDICTION WORKSPACE (Target for Scrolling)
st.markdown('<div id="predict-section"></div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; margin-top: 50px;'>🔍 Machine Learning Workspace</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Run Prediction", "📊 System Analytics", "ℹ️ Documentation"])

with tab1:
    if model is None:
        st.error("Assets Loading Error: Check if model.h5, scaler.pkl, and encoders are in the root directory.")
    else:
        col_input, col_output = st.columns([1.8, 1], gap="large")
        
        with col_input:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Customer Profile Details")
            c1, c2 = st.columns(2)
            with c1:
                geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
                gender = st.selectbox('Gender', label_encoder_gender.classes_)
                age = st.slider('Age', 18, 92, 35)
                tenure = st.slider('Tenure (Years)', 0, 10, 5)
            with c2:
                credit_score = st.number_input('Credit Score', 300, 850, 650)
                balance = st.number_input('Account Balance ($)', 0.0, 250000.0, 50000.0)
                estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 75000.0)
                num_of_products = st.selectbox('Products Owned', [1, 2, 3, 4])
            
            is_active = st.checkbox('Active Member Status', value=True)
            has_card = st.checkbox('Has Credit Card', value=True)
            
            predict_button = st.button("🚀 Analyze Churn Risk", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_output:
            if predict_button:
                # Prepare data for prediction
                input_df = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Gender': [label_encoder_gender.transform([gender])[0]],
                    'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
                    'NumOfProducts': [num_of_products],
                    'HasCrCard': [1 if has_card else 0],
                    'IsActiveMember': [1 if is_active else 0],
                    'EstimatedSalary': [estimated_salary]
                })

                # Geo Encoding
                geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
                geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
                final_input = pd.concat([input_df, geo_df], axis=1)
                
                # Scaling & Prediction
                final_scaled = scaler.transform(final_input)
                prediction_proba = model.predict(final_scaled, verbose=0)[0][0]
                
                res_color = "#ef4444" if prediction_proba > 0.5 else "#22c55e"
                
                st.markdown(f"""
                    <div style="background: {res_color}20; border: 2px solid {res_color}; padding: 30px; border-radius: 20px; text-align: center;">
                        <h4 style="color: #94a3b8; margin: 0;">CHURN PROBABILITY</h4>
                        <h1 style="font-size: 55px; color: {res_color}; margin: 15px 0;">{prediction_proba:.1%}</h1>
                        <p style="font-weight: 700; color: white;">{"⚠️ HIGH RISK" if prediction_proba > 0.5 else "✅ LOW RISK"}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Small Gauge Visual
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction_proba*100,
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': res_color}},
                    title={'text': "Confidence Meter", 'font': {'color': 'white'}}
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=250)
                st.plotly_chart(fig, use_container_width=True)

# 6. ANALYTICS SECTION (Target for Scrolling)
st.markdown('<div id="analytics-section"></div>', unsafe_allow_html=True)
with tab2:
    st.subheader("Global Churn Trends & Statistics")
    try:
        df_stats = pd.read_csv('Churn_Modelling.csv')
        c_a, c_b = st.columns(2)
        with c_a:
            fig_age = px.histogram(df_stats, x="Age", color="Exited", nbins=30, barmode="overlay", 
                                  title="Churn Correlation by Age Group", color_discrete_map={0: "#22c55e", 1: "#ef4444"})
            fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_age, use_container_width=True)
        with c_b:
            fig_geo = px.pie(df_stats, names='Geography', title="Customer Base by Geography", hole=0.4)
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_geo, use_container_width=True)
    except:
        st.warning("Churn_Modelling.csv data file is missing for analytics display.")

# 7. DOCUMENTATION
with tab3:
    st.markdown("""
    ### 🧠 AI Model Logic
    Model uses a **Feed-Forward Neural Network (ANN)** to classify customer risk.
    - **Preprocessing**: Label Encoding (Gender), One-Hot Encoding (Geography), Standard Scaling.
    - **Accuracy**: Optimized on historical banking data for maximum precision.
    """)

# 8. FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 40px; border-top: 1px solid rgba(255,255,255,0.05); color: #64748b;">
    <p style="font-weight: 600;">ChurnPredict v3.2 | Powered by TensorFlow</p>
    <p style="font-size: 13px;">© 2026 Built by Ujjwal Ray | AI & Retention Intelligence</p>
</div>
""", unsafe_allow_html=True)
