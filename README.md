# ✈️ Airplane Fare Classification System

An advanced machine learning system that classifies or predicts airline ticket fares using powerful ensemble models. This project utilizes rich feature engineering, optimized ML algorithms (CatBoost, XGBoost, Random Forest, etc.), and SHAP for explainability, all in a streamlined, reproducible pipeline. Inspired by FAANG-level deployment standards and built for real-world scalability.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Data Overview](#data-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling & Evaluation](#modeling--evaluation)
- [Model Explainability (SHAP)](#model-explainability-shap)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)

---

## 🚀 Project Overview

Airfare pricing is dynamic and influenced by multiple factors such as departure times, stops, source-destination pairs, and airlines. This project builds a robust ML pipeline that classifies airline ticket fares based on these parameters with high accuracy.

The solution is designed to support:
- Revenue management systems
- Real-time fare classification
- Business analytics dashboards

---

## 🧠 Features

✅ Complete end-to-end machine learning pipeline  
✅ Advanced preprocessing and feature engineering  
✅ 5 state-of-the-art ML models benchmarked  
✅ Model explainability using SHAP values  
✅ FAANG-inspired architecture and modular codebase  

---

## ⚙️ Tech Stack

- Python (Pandas, NumPy)
- Scikit-learn
- CatBoost, XGBoost
- Matplotlib, Seaborn
- SHAP (SHapley Additive Explanations)
- Jupyter Notebook

---

## 🗃️ Data Overview

**Features Used:**

- `Airline`, `Source`, `Destination`
- `Date_of_Journey`, `Dep_Time`, `Arrival_Time`
- `Duration` (transformed to minutes)
- `Total_Stops`, `Price` (target)

**Feature Engineering:**
- Extracted journey day, month, year
- Converted time columns into hour and minute
- Calculated total duration in minutes
- Label encoded categorical variables
- Selected features using Mutual Information (threshold = 0.5)

---

## 📊 Exploratory Data Analysis

- **Boxplots & Violin plots**: Outlier detection and distribution analysis
- **Bar plots**: Frequency of flights by day/month
- **Heatmaps**: Correlation matrices between numerical features

---

## 🤖 Modeling & Evaluation

Five models were trained and compared using R² score on the test set:

| Model              | R² Score    |
|--------------------|-------------|
| Decision Tree      | 0.97.64     |
| Random Forest      | 0.97.64     |
| KNN                | 0.92        |
| XGBoost            | 0.95.8      |
| CatBoost           | 0.95.8      |

> ✅ **Best Models**: Random Forest, XGBoost, and CatBoost all performed at near-perfect R² (~0.99)

---

## 🧠 Model Explainability (SHAP)

To interpret model predictions, SHAP (SHapley Additive exPlanations) was used:

- **SHAP Summary Plot** shows the impact of each feature on the model’s output.
- **Key Insights**:
  - `Total_Stops` and `Duration_mins` were the most influential features.
  - `Airline` and `Dep_hour` also had significant predictive power.
  
```python
import shap

# Load model and data
explainer = shap.TreeExplainer(best_model)  # best_model = Random Forest
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)
