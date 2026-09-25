# 🌾 Smart Crop Advisory System

> An AI-powered agricultural decision support system that recommends suitable crops, predicts yield, estimates profit, and provides crop rotation suggestions using Machine Learning and agronomic rules.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 📖 Overview

Agriculture is highly dependent on soil quality, nutrient availability, climatic conditions, and economic factors. Selecting the wrong crop can result in low yield, financial losses, and soil degradation.

The **Smart Crop Advisory System** helps farmers and agricultural stakeholders make informed decisions by:

- Recommending the **Top-3 suitable crops**
- Predicting **crop yield**
- Estimating **revenue and profit**
- Suggesting **crop rotation plans**
- Applying **state and climate-specific validation rules**


The system combines **Machine Learning** with **Agronomic Rule-Based Intelligence** to provide realistic and practical recommendations.


---

## 🚀 Features

### 🌱 Crop Recommendation
- Predicts Top-3 suitable crops.
- Uses soil, nutrient, climate, and economic inputs.
- Employs a stacked ensemble classification model.

### 📈 Yield Prediction
- Predicts expected crop yield (kg/ha).
- Uses ensemble regression models.

### 💰 Revenue & Profit Estimation
Calculates:

```text
Revenue = (Yield / 1000) × Market Price

Profit = Revenue − Input Cost
```

### 🔄 Crop Rotation Planner
- Suggests suitable rotation crops.
- Helps maintain soil fertility.
- Supports sustainable farming practices.

### 📊 Interactive Dashboard
- Multi-page Streamlit application.
- Interactive charts and comparisons.
- Downloadable results.

---

## 🏗️ System Architecture

```text
User Inputs
     │
     ▼
Feature Engineering
     │
     ▼
Preprocessing
(Encoding + Scaling)
     │
     ▼
Stacked Ensemble Classifier
(RF + XGBoost + Meta Classifier)
     │
     ▼
Top-3 Crop Recommendation
     │
     ▼
Rule-Based Validation Layer
     │
     ▼
Yield Prediction
(RF Regressor + XGBoost Regressor)
     │
     ▼
Revenue & Profit Calculation
     │
     ▼
Crop Rotation Planner
```

---

## 🤖 Machine Learning Models

### Classification Models

#### Random Forest Classifier
- Reduces variance.
- Handles non-linear relationships.
- Provides stable predictions.

#### XGBoost Classifier
- Reduces bias.
- Improves predictive accuracy.
- Captures complex interactions.

#### Stacked Ensemble Meta-Classifier
Combines predictions from:

- Random Forest Classifier
- XGBoost Classifier

Benefits:
- Increased robustness
- Better generalization
- Improved crop recommendation performance

---

### Regression Models

#### Random Forest Regressor
Predicts crop yield based on:

- Soil conditions
- Climate parameters
- Economic factors

#### XGBoost Regressor
Provides additional yield estimation using gradient boosting.

#### Ensemble Yield Prediction

```text
Final Yield =
(Random Forest Prediction + XGBoost Prediction) / 2
```

---

## ⚙️ Feature Engineering

The following engineered features are used:

### Nutrient Features

```text
NPK Sum = N + P + K
```

### Nutrient Ratios

```text
N Ratio = N / (N + P + K)

P Ratio = P / (N + P + K)

K Ratio = K / (N + P + K)
```

### Economic Feature

```text
Cost-to-Price Ratio
```

These features improve model performance and agronomic relevance.

---

## 🧠 Data Preprocessing

### Data Cleaning
- Duplicate removal
- Missing value handling
- Data standardization
- Invalid value correction

### Outlier Handling
- Interquartile Range (IQR) Method
- Domain-based validation

### Encoding
- OneHotEncoder for categorical variables
- LabelEncoder for crop classes

### Scaling
- Feature normalization using Scalers
- Improved numerical stability

---

## 📊 Input Parameters

The system takes the following inputs:

| Parameter | Description |
|------------|-------------|
| State | Geographical location |
| Soil Type | Type of agricultural soil |
| Rainfall | Seasonal rainfall (mm) |
| Temperature | Average temperature (°C) |
| Humidity | Relative humidity (%) |
| Soil pH | Acidity/alkalinity of soil |
| Nitrogen (N) | Nitrogen content |
| Phosphorus (P) | Phosphorus content |
| Potassium (K) | Potassium content |
| Input Cost | Cultivation cost (₹/ha) |
| Market Price | Crop market price (₹/ton) |

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### Visualization
- Altair

### Web Application
- Streamlit

### Model Persistence
- Joblib

### Rule Engine
- JSON

---

## 📂 Project Structure

```text
FINAL YEAR AOT/
│
├── data/
│   ├── cleaned_dataset.csv
│   └── final_yield_complete.csv
│
├── models/
│   ├── rf_classifier.pkl
│   ├── xgb_classifier.model
│   ├── meta_classifier.pkl
│   ├── rf_yield_reg.pkl
│   ├── xgb_yield_reg.model
│   ├── ohe.pkl
│   ├── ohe_crop.pkl
│   ├── scaler.pkl
│   ├── scaler_reg.pkl
│   ├── labelencoder_crop.pkl
│   ├── rotation_rules.json
│   └── crop_means.json
│
├── src/
│   ├── 01_data_prep.py
│   ├── 02_feature_engineer.py
│   ├── 03_train_classifiers.py
│   ├── 04_train_regressors.py
│   ├── 05_ensemble_meta.py
│   ├── 06_rotation_rules.py
│   ├── rules_state_climate.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/smart-crop-advisory-system.git
cd smart-crop-advisory-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run src/app.py
```

---

## 🎯 Real-World Impact

This system helps:

- Farmers
- Agricultural Consultants
- Researchers
- Government Agencies
- Agri-Tech Startups

Benefits:

- Better crop selection
- Improved profitability
- Reduced cultivation risks
- Sustainable farming practices
- Data-driven decision making

---

## 🔮 Future Enhancements

- Weather API Integration
- Soil Health Card Integration
- Explainable AI (XAI)
- Mobile Application
- Cloud Deployment
- IoT-Based Soil Monitoring
- Multi-Language Support

---

## 👨‍🎓 Academic Project

**Project Title:** Smart Crop Advisory System

**Project Type:** B.Tech Final Year Project

**Domain:** Machine Learning | Agriculture Technology | Decision Support Systems

---

## 📜 License

This project is developed for academic and educational purposes.

---
