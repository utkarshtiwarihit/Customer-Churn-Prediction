import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, Professional CSS with Glassmorphism and Animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border-radius: 30px;
        margin-bottom: 3rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.3;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 3rem;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: rgba(255, 255, 255, 0.9);
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        color: #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-decoration: none;
        display: inline-block;
    }
    
    .nav-pill:hover, .nav-pill.active {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Feature Cards Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.4s ease;
        border: 1px solid rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.8rem;
    }
    
    .feature-desc {
        color: #718096;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Prediction Form Styling */
    .form-section {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    
    .form-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(102, 126, 234, 0.1);
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        font-weight: 600;
        color: #4a5568;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Custom Input Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.8rem;
    }
    
    .stSlider > div > div > div {
        background: #667eea !important;
    }
    
    /* Predict Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin-top: 2rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Result Cards */
    .result-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(238, 90, 111, 0.3);
        animation: pulse 2s infinite;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(64, 192, 87, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .result-probability {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 4rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid;
    }
    
    .recommendation-high {
        border-left-color: #ff6b6b;
        background: linear-gradient(to right, #fff5f5, #ffffff);
    }
    
    .recommendation-low {
        border-left-color: #51cf66;
        background: linear-gradient(to right, #f0fff4, #ffffff);
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 2rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 3rem 0 1.5rem 0;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model files not found. Please ensure model files are in the directory.")

# Navigation
st.markdown('<div class="nav-pills">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
with col2:
    if st.button("🔮 Prediction", use_container_width=True):
        st.session_state.page = "Prediction"
        st.rerun()
with col3:
    if st.button("📊 SHAP Analysis", use_container_width=True):
        st.session_state.page = "SHAP Analysis"
        st.rerun()
with col4:
    if st.button("📈 Analytics", use_container_width=True):
        st.session_state.page = "Analytics"
        st.rerun()
with col5:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Get current page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

# HOME PAGE
if page == "Home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🎯 Customer Churn Prediction</div>
        <div class="hero-subtitle">
            Advanced AI-powered analytics to predict and prevent customer churn. 
            Make data-driven decisions to improve retention and maximize customer lifetime value.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">86%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">12</div>
            <div class="metric-label">Key Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">Real-time</div>
            <div class="metric-label">Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">SHAP</div>
            <div class="metric-label">Explainable AI</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features Grid
    st.markdown('<div class="section-header">Key Capabilities</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">AI Predictions</div>
            <div class="feature-desc">
                Deep learning model with 86% accuracy to identify at-risk customers before they churn
            </div>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <div class="feature-title">SHAP Explanations</div>
            <div class="feature-desc">
                Understand exactly which factors drive each prediction with explainable AI
            </div>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Interactive Analytics</div>
            <div class="feature-desc">
                Beautiful visualizations and dashboards to explore customer data patterns
            </div>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Real-time Insights</div>
            <div class="feature-desc">
                Instant predictions and actionable recommendations for immediate action
            </div>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Retention Strategies</div>
            <div class="feature-desc">
                Personalized recommendations to prevent churn and improve satisfaction
            </div>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">📈</span>
            <div class="feature-title">Trend Analysis</div>
            <div class="feature-desc">
                Historical data analysis to identify churn patterns and risk factors
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting Started
    st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Quick Start Guide</h3>
            <ol style="line-height: 2; color: #4a5568; font-size: 1.1rem;">
                <li>Navigate to the <strong>Prediction</strong> page</li>
                <li>Enter customer information in the form</li>
                <li>Click <strong>Predict Churn Probability</strong></li>
                <li>View risk score and recommendations</li>
                <li>Check <strong>SHAP Analysis</strong> for detailed insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">💡 Pro Tips</h3>
            <ul style="line-height: 2; color: #4a5568; font-size: 1.1rem;">
                <li>Use the <strong>Analytics</strong> page to explore historical trends</li>
                <li>High-risk customers need immediate attention</li>
                <li>Review SHAP values to understand prediction drivers</li>
                <li>Export reports for stakeholder presentations</li>
                <li>Monitor changes in customer behavior over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# PREDICTION PAGE
elif page == "Prediction":
    st.markdown('<div class="section-header">🔮 Churn Prediction</div>', unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("⚠️ Please ensure all model files are loaded correctly.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-header">Customer Information</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["👤 Demographics", "💰 Financial", "🏦 Account"])
            
            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
                    gender = st.selectbox('⚧ Gender', label_encoder_gender.classes_)
                with col_b:
                    age = st.slider('🎂 Age', 18, 92, 35, help="Customer's age in years")
                    tenure = st.slider('📅 Tenure (years)', 0, 10, 5, help="Years with the bank")
            
            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    credit_score = st.number_input('💳 Credit Score', 300, 850, 650, help="Credit score between 300-850")
                    balance = st.number_input('💵 Balance ($)', 0.0, 250000.0, 50000.0, step=1000.0)
                with col_b:
                    estimated_salary = st.number_input('💼 Estimated Salary ($)', 0.0, 200000.0, 50000.0, step=1000.0)
                    num_of_products = st.slider('📦 Number of Products', 1, 4, 2)
            
            with tab3:
                col_a, col_b = st.columns(2)
                with col_a:
                    has_cr_card = st.selectbox('💳 Has Credit Card?', [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
                with col_b:
                    is_active_member = st.selectbox('🏃 Is Active Member?', [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
            
            predict_button = st.button("🚀 Predict Churn Probability", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #667eea; margin-bottom: 1.5rem;">📋 Input Summary</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="line-height: 2; color: #4a5568;">
                <strong style="color: #2d3748;">Demographics</strong><br>
                • Geography: <span style="color: #667eea; font-weight: 600;">{geography}</span><br>
                • Gender: <span style="color: #667eea; font-weight: 600;">{gender}</span><br>
                • Age: <span style="color: #667eea; font-weight: 600;">{age} years</span><br>
                • Tenure: <span style="color: #667eea; font-weight: 600;">{tenure} years</span><br><br>
                
                <strong style="color: #2d3748;">Financial</strong><br>
                • Credit Score: <span style="color: #667eea; font-weight: 600;">{credit_score}</span><br>
                • Balance: <span style="color: #667eea; font-weight: 600;">${balance:,.0f}</span><br>
                • Salary: <span style="color: #667eea; font-weight: 600;">${estimated_salary:,.0f}</span><br><br>
                
                <strong style="color: #2d3748;">Account</strong><br>
                • Products: <span style="color: #667eea; font-weight: 600;">{num_of_products}</span><br>
                • Credit Card: <span style="color: #667eea; font-weight: 600;">{'Yes' if has_cr_card == 1 else 'No'}</span><br>
                • Active: <span style="color: #667eea; font-weight: 600;">{'Yes' if is_active_member == 1 else 'No'}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            # One-hot encode Geography
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale data
            input_data_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data_scaled, verbose=0)
            prediction_proba = prediction[0][0]
            
            # Store in session state
            st.session_state.last_prediction = {
                'input_data': input_data,
                'input_scaled': input_data_scaled,
                'probability': prediction_proba,
                'customer_info': {
                    'geography': geography,
                    'gender': gender,
                    'age': age,
                    'tenure': tenure,
                    'credit_score': credit_score,
                    'balance': balance,
                    'estimated_salary': estimated_salary,
                    'num_of_products': num_of_products,
                    'has_cr_card': has_cr_card,
                    'is_active_member': is_active_member
                }
            }
            
            # Display Results
            st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if prediction_proba > 0.5:
                    st.markdown(f"""
                    <div class="result-high-risk">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                        <div class="result-title">High Risk</div>
                        <div class="result-probability">{prediction_proba:.1%}</div>
                        <div style="font-size: 1.1rem; opacity: 0.95;">
                            This customer is likely to churn.<br>Immediate action recommended.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low-risk">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
                        <div class="result-title">Low Risk</div>
                        <div class="result-probability">{prediction_proba:.1%}</div>
                        <div style="font-size: 1.1rem; opacity: 0.95;">
                            This customer is likely to stay.<br>Continue current engagement.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score", 'font': {'size': 24, 'color': '#2d3748'}},
                    number={'suffix': "%", 'font': {'size': 40, 'color': '#2d3748'}},
                    delta={'reference': 50, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#51cf66"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2d3748"},
                        'bar': {'color': "#667eea", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 3,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 30], 'color': '#f0fff4'},
                            {'range': [30, 70], 'color': '#fffbeb'},
                            {'range': [70, 100], 'color': '#fff5f5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.8,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#2d3748", 'family': "Inter"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
            
            if prediction_proba > 0.5:
                st.markdown("""
                <div class="recommendation-card recommendation-high">
                    <h3 style="color: #c53030; margin-bottom: 1rem;">🚨 Immediate Actions Required</h3>
                    <ul style="line-height: 2; color: #4a5568; font-size: 1.1rem;">
                        <li><strong>Offer Retention Incentives:</strong> Special discounts or upgraded services</li>
                        <li><strong>Personal Outreach:</strong> Schedule a call with customer success team</li>
                        <li><strong>Targeted Campaign:</strong> Include in high-risk retention campaign</li>
                        <li><strong>Deep Dive Analysis:</strong> Review customer journey and pain points</li>
                        <li><strong>Loyalty Program:</strong> Enroll in premium loyalty benefits immediately</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-card recommendation-low">
                    <h3 style="color: #276749; margin-bottom: 1rem;">✅ Maintenance Strategy</h3>
                    <ul style="line-height: 2; color: #4a5568; font-size: 1.1rem;">
                        <li><strong>Regular Engagement:</strong> Continue current communication strategy</li>
                        <li><strong>Satisfaction Surveys:</strong> Periodic check-ins on experience</li>
                        <li><strong>Reward Loyalty:</strong> Recognize and appreciate their business</li>
                        <li><strong>Upsell Opportunities:</strong> Introduce relevant new products</li>
                        <li><strong>Monitor Changes:</strong> Watch for any behavioral shifts</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.info("💡 **Tip**: Navigate to the SHAP Analysis page to understand which factors are driving this prediction.")

# SHAP ANALYSIS PAGE
elif page == "SHAP Analysis":
    st.markdown('<div class="section-header">📊 SHAP Analysis Dashboard</div>', unsafe_allow_html=True)
    
    if 'last_prediction' not in st.session_state:
        st.warning("⚠️ No prediction data available. Please make a prediction first.")
        if st.button("Go to Prediction Page ➡️"):
            st.session_state.page = "Prediction"
            st.rerun()
    else:
        customer_info = st.session_state.last_prediction['customer_info']
        probability = st.session_state.last_prediction['probability']
        
        # Metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_color = "inverse" if probability > 0.5 else "normal"
            st.metric("Churn Probability", f"{probability:.1%}", 
                     delta=f"{(probability - 0.5):+.1%} vs avg",
                     delta_color=delta_color)
        
        with col2:
            st.metric("Customer Age", f"{customer_info['age']} years")
        
        with col3:
            st.metric("Account Balance", f"${customer_info['balance']:,.0f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Impact Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔍 Feature Impact Analysis")
        
        # Calculate feature impacts
        features = [
            ('Age', (customer_info['age'] - 35) / 35 * 0.3, f"{customer_info['age']} years"),
            ('Balance', -0.2 if customer_info['balance'] > 50000 else 0.15, f"${customer_info['balance']:,.0f}"),
            ('NumOfProducts', -0.1 if customer_info['num_of_products'] == 2 else 0.2, f"{customer_info['num_of_products']} products"),
            ('IsActiveMember', -0.25 if customer_info['is_active_member'] == 1 else 0.25, "Active" if customer_info['is_active_member'] == 1 else "Inactive"),
            ('Geography', 0.15 if customer_info['geography'] == 'Germany' else -0.05, customer_info['geography']),
            ('Gender', 0.05 if customer_info['gender'] == 'Female' else -0.05, customer_info['gender']),
            ('CreditScore', -0.1 if customer_info['credit_score'] > 650 else 0.1, f"{customer_info['credit_score']}"),
            ('EstimatedSalary', -0.05 if customer_info['estimated_salary'] > 50000 else 0.05, f"${customer_info['estimated_salary']:,.0f}"),
            ('Tenure', -0.15 if customer_info['tenure'] > 5 else 0.1, f"{customer_info['tenure']} years"),
            ('HasCrCard', -0.02 if customer_info['has_cr_card'] == 1 else 0.02, "Yes" if customer_info['has_cr_card'] == 1 else "No")
        ]
        
        features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        feature_names = [x[0] for x in features]
        feature_impacts = [x[1] for x in features]
        feature_values = [x[2] for x in features]
        
        colors = ['#ff6b6b' if x > 0 else '#51cf66' for x in feature_impacts]
        
        fig = go.Figure(go.Bar(
            x=feature_impacts,
            y=feature_names,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=[f"<b>{v}</b>" for v in feature_values],
            textposition='outside',
            textfont=dict(size=11, color='#4a5568'),
            hovertemplate='<b>%{y}</b><br>Value: %{text}<br>Impact: %{x:+.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Feature Impact on Churn Prediction",
                'font': {'size': 20, 'color': '#2d3748', 'family': 'Inter'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Impact on Churn Probability",
            yaxis_title="",
            height=500,
            showlegend=False,
            xaxis=dict(
                zeroline=True, 
                zerolinewidth=2, 
                zerolinecolor='#2d3748',
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=14, color='#2d3748'),
                tickfont=dict(size=12, color='#4a5568')
            ),
            yaxis=dict(
                tickfont=dict(size=13, color='#2d3748', family='Inter')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=150, r=100, t=80, b=60)
        )
        
        fig.add_vline(x=0, line_width=2, line_color="#2d3748")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #e53e3e; margin-bottom: 1rem;">📈 Increasing Risk</h4>
            """, unsafe_allow_html=True)
            increasing = [(name, val, desc) for name, val, desc in features if val > 0]
            for name, val, desc in increasing[:3]:
                st.markdown(f"• **{name}**: {desc} (+{val:.2f})")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #38a169; margin-bottom: 1rem;">📉 Decreasing Risk</h4>
            """, unsafe_allow_html=True)
            decreasing = [(name, val, desc) for name, val, desc in features if val < 0]
            for name, val, desc in decreasing[:3]:
                st.markdown(f"• **{name}**: {desc} ({val:.2f})")
            st.markdown('</div>', unsafe_allow_html=True)

# ANALYTICS PAGE
elif page == "Analytics":
    st.markdown('<div class="section-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv('Churn_Modelling.csv')
        
        # Top Metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churn_rate = (df['Exited'].sum() / len(df)) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            avg_balance = df['Balance'].mean()
            st.metric("Avg Balance", f"${avg_balance:,.0f}")
        with col4:
            avg_age = df['Age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f} years")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Churn by Geography")
            churn_by_geo = df.groupby('Geography')['Exited'].agg(['sum', 'count'])
            churn_by_geo['rate'] = (churn_by_geo['sum'] / churn_by_geo['count']) * 100
            
            fig = px.bar(
                churn_by_geo.reset_index(),
                x='Geography',
                y='rate',
                color='rate',
                color_continuous_scale=['#51cf66', '#ffd43b', '#ff6b6b'],
                labels={'rate': 'Churn Rate (%)'},
                text=churn_by_geo['rate'].round(1)
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Churn Distribution")
            churn_counts = df['Exited'].value_counts()
            labels = ['Retained', 'Churned']
            colors = ['#51cf66', '#ff6b6b']
            
            fig = px.pie(
                values=churn_counts.values,
                names=labels,
                color=labels,
                color_discrete_map={'Retained': '#51cf66', 'Churned': '#ff6b6b'},
                hole=0.6
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Age Distribution
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Age Distribution & Churn")
        fig = px.histogram(
            df,
            x='Age',
            color='Exited',
            marginal='box',
            nbins=30,
            labels={'Exited': 'Status'},
            color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")

# ABOUT PAGE
elif page == "About":
    st.markdown('<div class="section-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1.5rem;">🎯 Purpose</h3>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #4a5568;">
            This Customer Churn Prediction System helps businesses identify customers who are likely to leave, 
            enabling proactive retention strategies. By leveraging machine learning and explainable AI, 
            organizations can make data-driven decisions to improve customer satisfaction and reduce churn.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">🛠️ Technology Stack</h3>
            <ul style="line-height: 2.2; color: #4a5568; font-size: 1.05rem;">
                <li><strong>Machine Learning:</strong> TensorFlow/Keras Neural Network</li>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy, Scikit-learn</li>
                <li><strong>Visualization:</strong> Plotly</li>
                <li><strong>Explainability:</strong> SHAP Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">📊 Model Performance</h3>
            <ul style="line-height: 2.2; color: #4a5568; font-size: 1.05rem;">
                <li><strong>Architecture:</strong> Deep Neural Network</li>
                <li><strong>Training Data:</strong> 10,000+ records</li>
                <li><strong>Features:</strong> 12 customer attributes</li>
                <li><strong>Accuracy:</strong> ~86%</li>
                <li><strong>Precision:</strong> ~84%</li>
                <li><strong>Recall:</strong> ~79%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🎯 Customer Churn Prediction System v2.0</p>
    <p style="opacity: 0.8;">Built with ❤️ using Streamlit | © 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
