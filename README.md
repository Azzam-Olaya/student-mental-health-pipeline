# student-mental-health-pipeline
# 🎓 Student Wellbeing Risk Classifier

A complete Data Science pipeline to analyze the factors associated with student
wellbeing and mental health risk, and to build a supervised classification prototype.

> ⚠️ **Disclaimer**: this project is strictly educational. The model is in no way
> a diagnostic tool, a clinical screening tool, a basis for disciplinary decisions,
> or a student selection mechanism.

## 🎯 Context and Objectives

An educational institution wants to use a student dataset in order to:

- **Detect** profiles with an increased risk of psychological difficulties
- **Anticipate** support needs (prevention)
- **Understand** the factors associated with psychological risk
- **Guide** students toward appropriate resources
- **Prioritize** student support actions

## 🔄 Data Science Workflow Covered

| Step | Content |
|------|---------|
| 1. EDA | Data structure, missing values, duplicates, outliers, descriptive statistics, target distribution, correlations |
| 2. Preparation | 80/20 split, imputation, outlier handling (IQR/Z-score), encoding, SMOTE, scaling, leakage-free pipeline |
| 3. Modeling | Comparison of at least 3 classifiers (accuracy, precision, recall, F1, confusion matrix, ROC-AUC) |
| 4. Optimization | GridSearchCV / RandomizedSearchCV, model selection, error analysis, pipeline export (`.joblib`) |
| 5. Deployment | Airflow DAG (scheduled orchestration) + interactive Streamlit app |

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Matplotlib / Seaborn ·
Joblib · Apache Airflow · Streamlit

## 📁 Project Structure

​```
├── data/              # Raw and cleaned data
├── notebooks/         # EDA and experiments
├── src/               # Preprocessing, training, and evaluation scripts
├── dags/              # Airflow DAG
├── app/               # Streamlit application
├── models/            # Saved pipeline (.joblib)
├── reports/           # Figures and results
└── requirements.txt
​```

## 🚀 Quick Start

​```bash
pip install -r requirements.txt
streamlit run app/app.py
​```

## ⚖️ Limitations and Ethics

Predictions are statistical indicators derived from a limited dataset. They never
replace the judgment of a healthcare professional or a qualified counselor.