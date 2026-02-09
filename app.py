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

# Page configuration
st.set_page_config(
    page_title="ChurnPredict - AI Customer Retention",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = tf.keras.models.load_model('model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# Complete Hero Section with Animation
hero_html = """
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    color: #ffffff;
    overflow-x: hidden;
}

#network-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
}

.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background: rgba(10, 14, 39, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.2rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 1.5rem;
    font-weight: 700;
}

.logo-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.logo-text {
    color: #ffffff;
}

.logo-text span {
    color: #4F46E5;
}

.nav-links {
    display: flex;
    gap: 2.5rem;
    align-items: center;
}

.nav-link {
    color: #a0aec0;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.95rem;
    transition: color 0.3s ease;
    cursor: pointer;
}

.nav-link:hover {
    color: #ffffff;
}

.cta-button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    padding: 0.7rem 1.5rem;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4);
}

.hero-section {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    padding: 8rem 3rem 4rem 3rem;
    z-index: 1;
}

.hero-content {
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    align-items: center;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(79, 70, 229, 0.15);
    border: 1px solid rgba(79, 70, 229, 0.3);
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #a78bfa;
    margin-bottom: 2rem;
}

.badge-dot {
    width: 8px;
    height: 8px;
    background: #4F46E5;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 1.5rem;
}

.white-text {
    color: #ffffff;
}

.blue-text {
    background: linear-gradient(135deg, #60A5FA 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.orange-text {
    background: linear-gradient(135deg, #FB923C 0%, #F97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.15rem;
    line-height: 1.8;
    color: #94a3b8;
    margin-bottom: 2.5rem;
    max-width: 600px;
}

.hero-buttons {
    display: flex;
    gap: 1rem;
    margin-bottom: 3rem;
}

.btn-primary {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(79, 70, 229, 0.4);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.05);
    color: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
    max-width: 700px;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.stat-label {
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 500;
}

.dashboard-preview {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
}

.dashboard-header {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.metric-title {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60A5FA;
}

.customer-data-list {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 1rem;
    line-height: 1.6;
    text-align: left;
}

.low-risk {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.5rem;
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.chart-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.chart-box {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 200px;
}

.chart-title {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}

.scroll-indicator {
    position: absolute;
    bottom: 3rem;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
}

.scroll-text {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.scroll-arrow {
    font-size: 1.5rem;
    color: #4F46E5;
    animation: bounce 2s infinite;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    60% { transform: translateY(-5px); }
}

@media (max-width: 1024px) {
    .hero-content {
        grid-template-columns: 1fr;
    }
    
    .hero-title {
        font-size: 3rem;
    }
    
    .nav-links {
        display: none;
    }
}
</style>
</head>
<body>
<canvas id="network-canvas"></canvas>

<div class="navbar">
    <div class="logo">
        <div class="logo-icon">🎯</div>
        <div class="logo-text">Churn<span>Predict</span></div>
    </div>
    <div class="nav-links">
        <a class="nav-link" href="#home">Home</a>
        <a class="nav-link" href="#features">Features</a>
        <a class="nav-link" href="#predict">Predict</a>
        <a class="nav-link" href="#demo">Demo</a>
        <a class="nav-link" href="#analytics">Analytics</a>
        <a class="nav-link" href="#tech">Tech Stack</a>
    </div>
    <button class="cta-button">✨ Try Predict</button>
</div>

<div class="hero-section">
    <div class="hero-content">
        <div class="hero-left">
            <div class="badge">
                <div class="badge-dot"></div>
                AI-Powered Analytics
            </div>
            
            <h1 class="hero-title">
                <div class="white-text">Predict Customer</div>
                <div class="blue-text">Churn</div>
                <div class="white-text">with</div>
                <div class="orange-text">AI Precision</div>
            </h1>
            
            <p class="hero-subtitle">
                Transform your customer data into actionable retention strategies.
                Our neural network-powered platform delivers <strong style="color: #fff;">86% accuracy</strong> in 
                predicting churn risk.
            </p>
            
            <div class="hero-buttons">
                <button class="btn-primary">✨ Try Prediction</button>
                <button class="btn-secondary">View Features →</button>
            </div>
            
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">10,000+</div>
                    <div class="stat-label">Customers Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">86%</div>
                    <div class="stat-label">Prediction Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">&lt;100ms</div>
                    <div class="stat-label">Response Time</div>
                </div>
            </div>
        </div>
        
        <div class="hero-right">
            <div class="dashboard-preview">
                <div class="dashboard-header">
                    <div class="metric-card">
                        <div class="metric-title">Customer Data</div>
                        <div class="customer-data-list">
                            • Credit Score<br>
                            • Geography<br>
                            • Age & Tenure<br>
                            • Account Balance<br>
                            • Products Used<br>
                            • Activity Status
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Churn Rate</div>
                        <div class="metric-value">12.5%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Prediction Accuracy</div>
                        <div class="metric-value">91.8%</div>
                        <div class="low-risk">Low Risk</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-box">
                        <div class="chart-title">MONTHLY CHURN TREND</div>
                        <svg width="100%" height="150" style="margin-top: 1rem;">
                            <path d="M 10 140 Q 40 100, 70 90 T 130 60 T 190 80 T 250 40" 
                                  stroke="#4F46E5" stroke-width="3" fill="none" 
                                  stroke-linecap="round"/>
                            <path d="M 10 140 Q 40 100, 70 90 T 130 60 T 190 80 T 250 40 L 250 150 L 10 150 Z" 
                                  fill="url(#gradient)" opacity="0.3"/>
                            <defs>
                                <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" style="stop-color:#4F46E5;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#4F46E5;stop-opacity:0" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    
                    <div class="chart-box">
                        <div class="chart-title">CONFIDENCE SCORE OVER TIME</div>
                        <svg width="100%" height="150" style="margin-top: 1rem;">
                            <circle cx="130" cy="75" r="60" fill="none" stroke="rgba(79, 70, 229, 0.2)" stroke-width="12"/>
                            <circle cx="130" cy="75" r="60" fill="none" stroke="#4F46E5" stroke-width="12"
                                    stroke-dasharray="377" stroke-dashoffset="75"
                                    transform="rotate(-90 130 75)"/>
                            <text x="130" y="85" text-anchor="middle" fill="#ffffff" font-size="24" font-weight="700">88%</text>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="scroll-indicator">
        <div class="scroll-text">Scroll to Explore</div>
        <div class="scroll-arrow">↓</div>
    </div>
</div>

<script>
const canvas = document.getElementById('network-canvas');
const ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.radius = Math.random() * 2 + 1;
        this.isOrange = Math.random() > 0.85;
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        
        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }
    
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.isOrange ? 'rgba(249, 115, 22, 0.6)' : 'rgba(79, 70, 229, 0.6)';
        ctx.fill();
    }
}

const particles = [];
const particleCount = 80;

for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
}

function connectParticles() {
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 150) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(79, 70, 229, ${0.2 * (1 - distance / 150)})`;
                ctx.lineWidth = 0.5;
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    particles.forEach(particle => {
        particle.update();
        particle.draw();
    });
    
    connectParticles();
    requestAnimationFrame(animate);
}

animate();
</script>
</body>
</html>
"""

# Render hero section
components.html(hero_html, height=1000, scrolling=False)

# Custom CSS for prediction form
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
}

.section-title {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin: 3rem 0;
    background: linear-gradient(135deg, #60A5FA 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 35px rgba(79, 70, 229, 0.4) !important;
}

label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Prediction Section
st.markdown('<h2 class="section-title">🎯 Predict Customer Churn</h2>', unsafe_allow_html=True)

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.slider('Age', 18, 92, 35)
        credit_score = st.slider('Credit Score', 300, 850, 650)
        tenure = st.slider('Tenure (years)', 0, 10, 5)
    
    with col2:
        balance = st.number_input('Account Balance ($)', min_value=0.0, value=50000.0)
        estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has_cr_card = st.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        is_active = st.selectbox('Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    if st.button('🔮 Predict Churn Probability'):
        # Prepare data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active],
            'EstimatedSalary': [estimated_salary]
        })
        
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled, verbose=0)
        prediction_proba = prediction[0][0]
        
        risk_color = "#ef4444" if prediction_proba > 0.5 else "#22c55e"
        risk_text = "High Risk ⚠️" if prediction_proba > 0.5 else "Low Risk ✅"
        
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); 
             border-radius: 20px; padding: 3rem; text-align: center; margin-top: 2rem;">
            <h3 style="color: #94a3b8; margin-bottom: 1rem;">Churn Probability</h3>
            <div style="font-size: 5rem; font-weight: 900; color: {risk_color}; margin: 2rem 0;">
                {prediction_proba:.1%}
            </div>
            <p style="font-size: 1.2rem; color: #94a3b8;">
                This customer is classified as <strong style="color: {risk_color};">{risk_text}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction_proba > 0.5:
            st.error("🚨 **High Risk Customer** - Immediate action recommended!")
            st.markdown("""
            - Schedule personalized retention call within 48 hours
            - Offer exclusive loyalty rewards or account upgrades
            - Review recent interactions for pain points
            - Consider special pricing or product bundles
            """)
        else:
            st.success("✅ **Low Risk Customer** - Continue excellent service!")
            st.markdown("""
            - Maintain regular engagement
            - Consider cross-selling opportunities
            - Leverage as brand advocate
            - Periodic relationship check-ins
            """)
else:
    st.error("⚠️ Model files not found. Please ensure all model files are in the directory.")
