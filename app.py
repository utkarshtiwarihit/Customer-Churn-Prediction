import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="ChurnPredict AI | Customer Retention",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced CSS for Reference Site Look
st.markdown("""
    <style>
    /* Main Background and Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stText, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9 !important;
    }

    /* Custom Cards */
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Prediction Result Styling */
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
    }
    
    .high-risk {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    
    .low-risk {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white !important;
        border: none;
        padding: 12px;
        font-weight: 700;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model & Scaler Loading (Optimized)
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_geo = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_gender = pickle.load(file)
    # Scaler load karna (agar scaler.pkl file hai, varna aapne jaisa use kiya waisa hi)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, onehot_geo, label_gender, scaler

try:
    model, onehot_encoder_geo, label_encoder_gender, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 4. Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
    st.title("ChurnPredict AI")
    st.markdown("---")
    page = st.radio("MAIN MENU", ["Dashboard", "Risk Predictor", "About Model"], index=1)
    st.markdown("---")
    st.info("System Status: Online 🟢")

# 5. Page: Risk Predictor
if page == "Risk Predictor":
    st.title("🔍 Customer Risk Analysis")
    st.write("Fill in the customer profile details to analyze the probability of churn.")
    
    # Layout with Columns
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Customer Demographics & Activity")
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
            gender = st.selectbox('Gender', label_encoder_gender.classes_)
            age = st.slider('Age', 18, 92, 35)
            tenure = st.slider('Tenure (Years)', 0, 10, 5)
        
        with sub_col2:
            credit_score = st.number_input('Credit Score', 300, 850, 650)
            balance = st.number_input('Account Balance', 0.0, 250000.0, 50000.0)
            estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 75000.0)
            num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
            
        st.markdown("---")
        c1, c2 = st.columns(2)
        has_cr_card = c1.radio('Has Credit Card?', ['Yes', 'No'])
        is_active_member = c2.radio('Is Active Member?', ['Yes', 'No'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Retention Insights")
        predict_button = st.button('Analyze Churn Risk ⚡')
        
        if predict_button:
            # Data Transformation
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
                'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
                'EstimatedSalary': [estimated_salary]
            })

            # Geography One-hot encoding
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine and Scale
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            input_scaled = scaler.transform(input_data)
            
            # Prediction
            prediction_proba = model.predict(input_scaled)[0][0]
            is_churn = prediction_proba > 0.5
            
            # Result Card
            if is_churn:
                st.markdown(f"""
                    <div class="result-card high-risk">
                        <h3>High Churn Risk!</h3>
                        <h1 style='font-size: 3rem;'>{prediction_proba:.1%}</h1>
                        <p>This customer is likely to leave the bank.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card low-risk">
                        <h3>Low Churn Risk</h3>
                        <h1 style='font-size: 3rem;'>{prediction_proba:.1%}</h1>
                        <p>This customer is stable and loyal.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4f46e5"},
                    'steps' : [
                        {'range': [0, 30], 'color': "#dcfce7"},
                        {'range': [30, 70], 'color': "#fef9c3"},
                        {'range': [70, 100], 'color': "#fee2e2"}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

# 6. Page: Dashboard (Placeholder for overall stats)
elif page == "Dashboard":
    st.title("📊 Enterprise Dashboard")
    st.write("Real-time summary of customer churn across the organization.")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><h4>Avg Churn Rate</h4><h2 style="color:#ef4444">24.2%</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h4>Monitored Users</h4><h2 style="color:#4f46e5">10,000+</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h4>Retention Goal</h4><h2 style="color:#22c55e">90.0%</h2></div>', unsafe_allow_html=True)
    
    st.image("https://raw.githubusercontent.com/streamlit/template-churn-prediction/main/images/churn_prediction_app.png")

# Footer
st.markdown("---")
st.caption("ChurnPredict AI Platform v2.0 | Powered by TensorFlow and Streamlit")
