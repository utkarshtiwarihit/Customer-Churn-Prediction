# Customer Churn Prediction 🔮

A machine learning web application built with Streamlit and TensorFlow to predict customer churn probability for banking customers.

## 🌐 Live Demo
### **[🚀 Try the App Here: https://customer-ann.streamlit.app/](https://customer-ann.streamlit.app/)**

## 👤 Owner
https://github.com/utkarshtiwarihit

---

## 📋 Overview

This project implements an Artificial Neural Network (ANN) to predict whether a bank customer is likely to churn based on various features such as credit score, geography, gender, age, balance, and more. The model is deployed as an interactive web application using Streamlit.

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for real-time predictions
- **Deep Learning Model**: TensorFlow/Keras-based ANN for accurate churn prediction
- **Preprocessing Pipeline**: Includes label encoding, one-hot encoding, and feature scaling
- **Real-time Predictions**: Get instant churn probability and classification

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow 2.15.0
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Model Serialization**: Pickle

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Model training notebook
├── prediction.ipynb                # Prediction testing notebook
├── model.h5                        # Trained neural network model
├── label_encoder_gender.pkl        # Gender label encoder
├── onehot_encoder_geo.pkl          # Geography one-hot encoder
├── scaler.pkl                      # Feature scaler
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ann_classification_churn
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Input Features

The application accepts the following customer information:

- **Geography**: Customer's country (France, Germany, Spain)
- **Gender**: Male or Female
- **Age**: Customer age (18-92 years)
- **Credit Score**: Customer's credit score
- **Balance**: Account balance
- **Tenure**: Number of years with the bank (0-10)
- **Number of Products**: Number of bank products used (1-4)
- **Has Credit Card**: Whether customer has a credit card (0/1)
- **Is Active Member**: Whether customer is an active member (0/1)
- **Estimated Salary**: Customer's estimated salary

### Output

- **Churn Probability**: A value between 0 and 1 indicating the likelihood of churn
- **Prediction**: Classification as "likely to churn" (>0.5) or "not likely to churn" (≤0.5)

## 🧠 Model Details

The project uses an Artificial Neural Network (ANN) trained on customer banking data. The model pipeline includes:

1. **Label Encoding**: Converts gender categories to numerical values
2. **One-Hot Encoding**: Transforms geography into binary columns
3. **Standard Scaling**: Normalizes numerical features for better model performance
4. **Neural Network**: Multi-layer perceptron for binary classification

## 📊 Model Training

Model training and experimentation details can be found in the Jupyter notebooks:
- `experiments.ipynb`: Contains model architecture, training process, and evaluation
- `prediction.ipynb`: Demonstrates prediction workflow with example data

## 🔧 Requirements

```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit==1.28.0
altair==4.2.2
scikeras
```

## 🐛 Troubleshooting

### Altair Module Error
If you encounter `ModuleNotFoundError: No module named 'altair.vegalite.v4'`, ensure you're using compatible versions:
```bash
pip install streamlit==1.28.0 altair==4.2.2
```

### Model Loading Issues
Ensure all `.pkl` and `.h5` files are in the same directory as `app.py`


## 📧 Contact

**Ujjwal Ray** - https://github.com/utkarshtiwarihit

---

⭐ If you found this project helpful, please consider giving it a star!


